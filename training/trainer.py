"""Live training runner with loss tracking and simple generation check."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.device_utils import get_best_device


class TokenBudgetSampler(torch.utils.data.Sampler):
    """Batches of roughly constant TOKEN count, grouped by length.

    ``collate_pad`` pads to the longest sequence in a batch, so a fixed batch
    size over a wide length distribution is the worst case: with p50 826 and
    p95 16,009 tokens, one long trace drags fifteen short ones to its length.
    Grouping by length instead lets short traces ride in large batches and
    long ones in small batches, both near ``max_tokens``.

    This matters most for MoE, whose per-step Python loop over experts is
    amortized by batch size: measured 6,149 tok/s at batch 2 x 832 against
    36,169 at batch 16 x 1024 - a 5.9x difference from batching alone.

    Sorting is done inside shuffled megabatches so the order still varies
    between epochs rather than being a fixed length ramp.
    """

    def __init__(self, lengths, max_tokens, max_batch=64, shuffle=True,
                 seed=0, mega=4096, drop_last=True):
        self.lengths = list(lengths)
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.shuffle = shuffle
        self.seed = seed
        self.mega = mega
        self.drop_last = drop_last
        self.epoch = 0
        self._batches = self._build()

    def _build(self):
        import random as _random
        order = list(range(len(self.lengths)))
        if self.shuffle:
            _random.Random(self.seed + self.epoch).shuffle(order)
        batches = []
        for start in range(0, len(order), self.mega):
            chunk = sorted(order[start:start + self.mega],
                           key=lambda i: self.lengths[i])
            cur, cur_max = [], 0
            for i in chunk:
                nxt_max = max(cur_max, self.lengths[i])
                if cur and (nxt_max * (len(cur) + 1) > self.max_tokens
                            or len(cur) >= self.max_batch):
                    batches.append(cur)
                    cur, cur_max = [i], self.lengths[i]
                else:
                    cur, cur_max = cur + [i], nxt_max
            if cur:
                batches.append(cur)
        if self.shuffle:
            _random.Random(self.seed + 977 + self.epoch).shuffle(batches)
        return batches

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._batches = self._build()

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class Trainer:
    """Train a tiny language model on sample stories.

    Supports AMP, gradient accumulation, a validation split, and cosine LR.
    """

    def __init__(self, model, dataset, batch_size: int = 4,
                 lr: float = 1e-3, device: str | None = None,
                 max_iters: int = 500,
                 val_dataset=None,
                 grad_accum_steps: int = 1,
                 warmup_steps: int = 200,
                 use_amp: bool = True,
                 amp_dtype: str = "bf16",
                 weight_decay: float = 0.1,
                 betas: tuple = (0.9, 0.95),
                 z_loss: float = 1e-4,
                 num_workers: int = 4,
                 checkpoint_path: str = None,
                 save_every: int = 0,
                 notify_every_min: float = 30.0,
                 resume: bool = False,
                 ddp: bool = False,
                 eval_fn=None,
                 eval_every: int = 0,
                 eval_patience: int = 4,
                 max_minutes: float = 0.0,
                 val_batches: int = 0,
                 max_tokens: int = 0,
                 max_batch: int = 64):
        if device is None:
            device = get_best_device()
        self.model = model.to(device)
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.device = device
        self.max_iters = max_iters
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.warmup_steps = max(1, warmup_steps)
        self.use_amp = use_amp and torch.cuda.is_available()
        # bf16 needs no loss scaling and cannot overflow the way fp16 does.
        # Both an RTX 4090 (sm_89) and a 5090 (sm_120) support it natively.
        if amp_dtype == "bf16" and torch.cuda.is_available() \
                and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype is torch.float16)
        self.z_loss = z_loss

        # Decay only matrices. Biases, norm gains and embeddings are excluded,
        # which is standard practice and matters more at small scale.
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or "embedding" in name or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=betas, eps=1e-8, fused=True)
        self._base_lr = lr
        self.optimizer.zero_grad()
        if hasattr(dataset, "collate_pad"):
            collator = dataset.collate_pad
        elif hasattr(dataset, "dataset"):
            collator = dataset.dataset.collate_pad
        else:
            collator = None
        loader_kw = dict(num_workers=num_workers, pin_memory=True,
                         persistent_workers=num_workers > 0,
                         drop_last=True) if num_workers > 0 else {}
        self.sampler = None
        self.batch_sampler = None
        if max_tokens and not ddp:
            base = dataset.dataset if hasattr(dataset, "dataset") else dataset
            idx = dataset.indices if hasattr(dataset, "indices") else range(len(base))
            lengths = [len(base.samples[i][0]) for i in idx]
            self.batch_sampler = TokenBudgetSampler(lengths, max_tokens, max_batch,
                                                    shuffle=True, seed=1234)
            self.loader = DataLoader(dataset, batch_sampler=self.batch_sampler,
                                     collate_fn=collator,
                                     **{k: v for k, v in loader_kw.items()
                                        if k != "drop_last"})
            tok_est = sum(lengths) / max(1, len(self.batch_sampler))
            print(f"token-budget batching: {len(self.batch_sampler):,} batches/epoch, "
                  f"~{tok_est:,.0f} real tokens each (budget {max_tokens:,})")
        elif ddp:
            from torch.utils.data.distributed import DistributedSampler
            # Each rank sees a disjoint shard, so the effective batch is
            # batch_size * grad_accum * world_size.
            self.sampler = DistributedSampler(dataset, shuffle=True,
                                              drop_last=True)
            self.loader = DataLoader(dataset, batch_size=batch_size,
                                     sampler=self.sampler,
                                     collate_fn=collator, **loader_kw)
        else:
            self.loader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True, collate_fn=collator,
                                     **loader_kw)
        if val_dataset is not None:
            if max_tokens:
                # The validation loader must respect the same token budget.
                # With a fixed batch size it is the first thing to OOM: at
                # batch 32 and a 19k-token trace the cross-entropy alone wants
                # ~20 GiB, and val runs at step 0, so the run dies before it
                # has trained anything.
                vbase = (val_dataset.dataset if hasattr(val_dataset, "dataset")
                         else val_dataset)
                vidx = (val_dataset.indices if hasattr(val_dataset, "indices")
                        else range(len(vbase)))
                vlens = [len(vbase.samples[i][0]) for i in vidx]
                self.val_loader = DataLoader(
                    val_dataset, collate_fn=collator,
                    batch_sampler=TokenBudgetSampler(vlens, max_tokens, max_batch,
                                                     shuffle=False, seed=0))
            else:
                self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=False, collate_fn=collator)
        else:
            self.val_loader = None
        self.ddp = ddp
        self.lr = lr
        # Task accuracy, not validation loss, is the signal that matters.
        # Loss keeps falling while the model memorizes; accuracy is what turns
        # over when it starts overfitting.
        self.eval_fn = eval_fn
        self.eval_every = eval_every
        self.eval_patience = eval_patience
        self.eval_history = []
        # Wall-clock budget for one segment. Training is split into short runs
        # because sustained dual-GPU draw trips the breaker; a segment stops on
        # time, writes resume state and exits 0, and the next one continues.
        self.max_minutes = max_minutes
        self.stopped_on_time = False
        # A full validation pass is 12.5k forward passes and costs minutes. It
        # runs 24 times in a 12k-iteration run, which is longer than the
        # training itself. Cap it; a fixed subset is still a valid comparison
        # because val_loader does not shuffle.
        self.val_batches = val_batches
        self.best_acc = -1.0
        self.best_acc_step = -1
        self._best_acc_state = None
        self.stopped_early = False
        self.checkpoint_path = checkpoint_path
        self.save_every = save_every
        self._report_horizon(max_iters, grad_accum_steps)
        # Minutes between Discord progress posts; 0 disables them.
        self.notify_every_min = notify_every_min
        self.start_step = 0
        self.history = []
        self.val_history = []
        self.best_loss = float("inf")
        self.best_val_loss = float("inf")
        self.best_val_step = -1
        self._best_model_state = None
        self.last_grad_norm = 0.0
        if resume and checkpoint_path:
            self._resume_from(checkpoint_path)

    def _report_horizon(self, max_iters: int, grad_accum: int):
        """Say how many epochs --iters actually is, and complain if it is wild.

        Epochs are batches_per_epoch / grad_accum, which is easy to misjudge:
        150,000 iters was described in this repo's own notes as "2.5 epochs"
        and is nearer 18. The LR schedule is warmup then cosine to 0.1x across
        max_iters, so a horizon that is wrong at launch cannot be fixed by
        stopping the run early - the weights never anneal.
        """
        try:
            batches = len(self.batch_sampler) if self.batch_sampler is not None \
                else len(self.loader)
        except TypeError:
            return
        steps_per_epoch = max(1, batches // max(1, grad_accum))
        epochs = max_iters / steps_per_epoch
        print(f"horizon: {max_iters:,} iters = {epochs:.1f} epochs "
              f"({steps_per_epoch:,} optimizer steps/epoch)")
        if epochs > 10:
            print(f"  WARNING: {epochs:.0f} epochs over one corpus is deep into "
                  f"memorisation for most sizes. The cosine schedule runs to "
                  f"--iters, so stopping early will NOT anneal the weights - "
                  f"set --iters to the horizon you actually want.", flush=True)

    def _resume_from(self, path: str):
        """Restore model, optimizer and step from a periodic checkpoint."""
        import os
        resume_path = path + ".resume"
        if not os.path.exists(resume_path):
            print(f"no resume state at {resume_path}; starting from scratch")
            return
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        # Saved from the unwrapped module, so it loads whether or not this
        # process is running under DDP. Older resume files were written from
        # the DDP wrapper and carry a "module." prefix; strip it.
        weights = state["model"]
        if any(k.startswith("module.") for k in weights):
            weights = {k.removeprefix("module."): v for k, v in weights.items()}
        self._unwrapped().load_state_dict(weights)
        self.model.to(self.device)
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler") and self.scaler.is_enabled():
            self.scaler.load_state_dict(state["scaler"])
        self.start_step = state.get("step", 0) + 1
        self.history = state.get("history", [])
        self.val_history = state.get("val_history", [])
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        self.best_val_step = state.get("best_val_step", -1)
        self.best_loss = state.get("best_loss", float("inf"))
        self._best_model_state = state.get("best_model_state")
        # Without these the next segment starts at best_acc = -1, so its first
        # evaluation always counts as an improvement and overwrites a better
        # _bestacc.pt from an earlier segment - and early stopping restarts its
        # patience count from zero every time.
        self.best_acc = state.get("best_acc", -1.0)
        self.best_acc_step = state.get("best_acc_step", -1)
        self.eval_history = state.get("eval_history", [])
        self._best_acc_state = state.get("best_acc_state")
        print(f"resumed from step {self.start_step} "
              f"(best_val={self.best_val_loss:.4f}"
              + (f", best_acc={self.best_acc:.1%} @ {self.best_acc_step}"
                 if self.best_acc >= 0 else "") + ")")

    def _run_eval(self, step: int, pbar=None):
        """Score the task battery; keep the best weights. Returns True to stop.

        Early stopping is on accuracy rather than loss because the two diverge
        exactly when it matters: an overfitting model's training loss keeps
        improving while its accuracy stalls or declines.
        """
        model = self._unwrapped()
        was_training = model.training
        model.eval()
        try:
            acc = float(self.eval_fn(model))
        except Exception as exc:                              # noqa: BLE001
            print(f"\n  [eval @ {step}] failed: {type(exc).__name__}: {exc}")
            if was_training:
                model.train()
            return False
        if was_training:
            model.train()

        self.eval_history.append((step, acc))
        improved = acc > self.best_acc
        if improved:
            self.best_acc = acc
            self.best_acc_step = step
            self._best_acc_state = {k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()}
            if self.checkpoint_path:
                best = self.checkpoint_path.replace(".pt", "_bestacc.pt")
                tmp = best + ".tmp"
                torch.save({"model": self._best_acc_state,
                            "config": model._get_config()
                            if hasattr(model, "_get_config") else {},
                            "accuracy": acc, "step": step}, tmp)
                os.replace(tmp, best)

        since = sum(1 for st, a in self.eval_history
                    if st > self.best_acc_step)
        marker = "BEST" if improved else f"no gain x{since}"
        print(f"\n  [eval @ {step:5d}] accuracy {acc:.1%}  "
              f"(best {self.best_acc:.1%} @ {self.best_acc_step}) {marker}",
              flush=True)

        if since >= self.eval_patience:
            print(f"  early stop: {since} evaluations without improvement; "
                  f"restoring step {self.best_acc_step} "
                  f"({self.best_acc:.1%})", flush=True)
            self.stopped_early = True
            return True
        return False

    def _unwrapped(self):
        return getattr(self.model, "module", self.model)

    def _save_resume_state(self, step: int):
        """Atomically write resume state so a crash cannot corrupt it.

        Written to a temporary file and renamed, because a power loss during a
        300MB write would otherwise leave a truncated checkpoint that is worse
        than having none.
        """
        import os
        if not self.checkpoint_path:
            return
        target = self.checkpoint_path + ".resume"
        tmp = target + ".tmp"
        torch.save({
            "model": self._unwrapped().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "step": step,
            "history": self.history,
            "val_history": self.val_history,
            "best_val_loss": self.best_val_loss,
            "best_val_step": self.best_val_step,
            "best_loss": self.best_loss,
            "best_model_state": self._best_model_state,
            "best_acc": self.best_acc,
            "best_acc_step": self.best_acc_step,
            "eval_history": self.eval_history,
            "best_acc_state": self._best_acc_state,
            "config": self._unwrapped()._get_config()
            if hasattr(self._unwrapped(), "_get_config") else {},
        }, tmp)
        os.replace(tmp, target)

    def _lr_for_step(self, step: int, lr: float):
        """Linear warmup then cosine decay to 0.1 * lr."""
        if step < self.warmup_steps:
            return lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.max_iters - self.warmup_steps)
        return 0.1 * lr + 0.9 * lr * (1 + float(torch.cos(torch.tensor(progress * 3.141592653589793)))) / 2

    def _set_lr(self, step: int):
        base_lr = self._base_lr
        new_lr = self._lr_for_step(step, base_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

    def _causal_mask(self, size: int):
        return torch.tril(torch.ones(size, size, device=self.device)).unsqueeze(0).unsqueeze(0)

    def _compute_loss(self, out, y, mask):
        aux_loss = 0.0
        if isinstance(out, tuple):
            logits, aux_loss = out
        else:
            logits = out
        if logits.dim() == 4:
            logits = logits[:, :, 0, :]
        B, S, V = logits.shape
        flat_logits = logits.reshape(-1, V)
        flat_mask = mask.reshape(-1)
        denom = flat_mask.sum().clamp(min=1.0)
        loss = F.cross_entropy(flat_logits, y.reshape(-1), reduction="none")
        loss = (loss * flat_mask).sum() / denom
        # Router z-loss (PaLM / ST-MoE): keeps logits from drifting to large
        # magnitudes, which is the main source of low-precision instability.
        if self.z_loss:
            log_z = torch.logsumexp(flat_logits.float(), dim=-1)
            loss = loss + self.z_loss * ((log_z.pow(2) * flat_mask).sum() / denom)
        if isinstance(aux_loss, torch.Tensor):
            loss = loss + 1.0 * aux_loss
        return loss

    def train_step(self, x, y, mask, step: int):
        self._set_lr(step)
        x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
        self.model.train()

        with torch.amp.autocast("cuda", enabled=self.use_amp,
                                dtype=self.amp_dtype):
            out = self.model(x)
            loss = self._compute_loss(out, y, mask)
            loss = loss / self.grad_accum_steps

        self.scaler.scale(loss).backward()

        if (step + 1) % self.grad_accum_steps == 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0)
            self.last_grad_norm = float(grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return loss.item() * self.grad_accum_steps

    @torch.no_grad()
    def val_loss(self):
        if self.val_loader is None:
            return None
        self.model.eval()
        total = 0.0
        count = 0
        for i, (x, y, mask) in enumerate(self.val_loader):
            if self.val_batches and i >= self.val_batches:
                break
            x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp,
                                    dtype=self.amp_dtype):
                out = self.model(x)
                loss = self._compute_loss(out, y, mask)
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / count if count else None

    def train(self):
        import time
        from agent.notify import notify
        t_start = time.time()
        last_ping = t_start
        # Only rank 0 speaks: eight DDP ranks narrating the same run would
        # make the channel useless.
        rank = 0
        if self.ddp:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
        speak = notify if rank == 0 else (lambda *a, **k: False)
        speak(f"training started: {self.max_iters:,} iters, "
              f"lr {self.lr:g}", tag="train")
        budget = self.max_minutes * 60 if self.max_minutes else 0
        data_iter = iter(self.loader)
        pbar = tqdm(range(self.start_step, self.max_iters), desc="training",
                    initial=self.start_step, total=self.max_iters)
        for step in pbar:
            try:
                x, y, mask = next(data_iter)
            except StopIteration:
                if self.sampler is not None:
                    # Reshuffle differently each epoch across ranks.
                    self.sampler.set_epoch(step)
                if self.batch_sampler is not None:
                    self.batch_sampler.set_epoch(step)
                data_iter = iter(self.loader)
                x, y, mask = next(data_iter)
            loss = self.train_step(x, y, mask, step)
            self.history.append(loss)
            self.best_loss = min(self.best_loss, loss)
            postfix = {"loss": f"{loss:.4f}", "best": f"{self.best_loss:.4f}"}
            if step % 500 == 0 and self.val_loader is not None:
                vl = self.val_loss()
                if vl is not None:
                    self.val_history.append(vl)
                    if vl < self.best_val_loss:
                        self.best_val_loss = vl
                        self.best_val_step = step
                        # Capture from the UNWRAPPED model: under DDP
                        # self.model.state_dict() carries a "module." prefix
                        # that will not load back into the bare Transformer.
                        self._best_model_state = {
                            k: v.detach().cpu().clone()
                            for k, v in self._unwrapped().state_dict().items()
                        }
                    postfix["val"] = f"{vl:.4f}"
            pbar.set_postfix(postfix)
            # Paced by minutes, not steps: a step-count interval reports every
            # few seconds early on and once an hour late on.
            if self.notify_every_min and time.time() - last_ping >= self.notify_every_min * 60:
                last_ping = time.time()
                done = step - self.start_step + 1
                total = max(1, self.max_iters - self.start_step)
                rate = done / max(1e-9, time.time() - t_start)
                eta_min = (total - done) / rate / 60 if rate else 0
                speak(f"step {step:,}/{self.max_iters:,} ({100.0*step/self.max_iters:.1f}%) "
                      f"loss {loss:.4f} best {self.best_loss:.4f} "
                      f"val {self.best_val_loss:.4f} ETA {eta_min/60:.1f}h", tag="train")
            if self.save_every and step and step % self.save_every == 0:
                self._save_resume_state(step)

            if self.eval_fn and self.eval_every and step \
                    and step % self.eval_every == 0:
                if self._run_eval(step, pbar):
                    break

            # Stop on the wall-clock budget rather than being killed, so the
            # resume state on disk is the step we actually reached.
            if budget and time.time() - t_start >= budget:
                self.stopped_on_time = True
                self._save_resume_state(step)
                elapsed = (time.time() - t_start) / 60
                print(f"\n  time budget reached at step {step} "
                      f"({elapsed:.1f} min); resume state written",
                      flush=True)
                speak(f"time budget reached at step {step:,} "
                      f"({elapsed:.1f} min); resume state written", tag="train")
                break
        # Prefer the best *accuracy* weights when a task eval was running;
        # fall back to best validation loss otherwise.
        target = self._unwrapped()
        if self.stopped_on_time:
            return self.history
        if self._best_acc_state is not None:
            target.load_state_dict(self._best_acc_state)
            print(f"restored best-accuracy weights: {self.best_acc:.1%} "
                  f"@ step {self.best_acc_step}")
        elif self._best_model_state is not None:
            target.load_state_dict(self._best_model_state)
        return self.history


    def save_best_val(self, path: str):
        """Save the tracked best-validation weights to disk."""
        if self._best_model_state is None:
            self._best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in self._unwrapped().state_dict().items()
            }
        torch.save({
            "model": self._best_model_state,
            "config": self.model._get_config()
            if hasattr(self.model, "_get_config") else {},
        }, path)

    def generate(self, prompt: str, max_new: int = 20,
                 tokenizer=None, detokenizer=None,
                 use_reasoning: bool = False,
                 think_steps: int = 5,
                 sampler=None) -> str:
        self.model.eval()
        if tokenizer is None:
            tokenizer = lambda text: [min(ord(c), 255) for c in text]
        if detokenizer is None:
            detokenizer = lambda tokens: "".join(chr(min(t, 255)) for t in tokens)
        tokens = tokenizer(prompt)
        if use_reasoning:
            from reasoning import ReasoningTimeHelper
            helper = ReasoningTimeHelper(self.model, think_steps=think_steps)
            tokens = helper.think_then_answer(
                torch.tensor([tokens], dtype=torch.long, device=self.device)
            )[0].tolist()
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            with torch.no_grad():
                return detokenizer(input_ids[0].tolist())
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        generated = tokens[:]
        if sampler is None:
            from sampling import Sampler
            sampler = Sampler(temperature=1.0, top_k=0, top_p=1.0,
                              repetition_penalty=1.0)
        with torch.no_grad():
            for _ in range(max_new):
                out = self.model(input_ids)
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out
                if logits.dim() == 4:
                    logits = logits[:, :, 0, :]
                next_logits = logits[:, -1, :]
                gen_tensor = torch.tensor(generated, dtype=torch.long, device=self.device)
                next_token = sampler.sample(next_logits, gen_tensor)
                generated.append(next_token)
                input_ids = torch.cat([input_ids,
                                       torch.tensor([[next_token]], device=self.device)], dim=1)
        return detokenizer(generated)
