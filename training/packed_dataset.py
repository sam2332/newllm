"""Fixed-length windows over the packed pretraining shards.

``scripts/build_pretrain_shards.py`` writes flat uint16 token streams with EOT
between documents. Pretraining wants every token to be a training token, so
this cuts the stream into fixed-length windows rather than padding documents:
at sequence 2048 a padded corpus of short web pages wastes most of the batch.

The mask is all ones. That is the whole difference between pretraining and the
instruct stage - there is no prompt to exclude, every position is supervised -
which is why this needs no change to ``training/trainer.py``: it yields the
same ``(x, y, mask)`` triple ``PrebuiltDataset`` does.

Shards are memory-mapped, so a 18 GB corpus costs no RAM and the page cache
does the work. Windows are indexed globally across shards, and a window never
straddles two shards - the seam would join unrelated documents mid-sentence.
"""

import json
import os

import numpy as np
import torch


class PackedShardDataset(torch.utils.data.Dataset):
    @staticmethod
    def collate_pad(batch):
        """Every window is exactly seq_len, so this only stacks.

        The Trainer looks up ``collate_pad`` on the dataset; the instruct path
        needs real padding, this path never does - which is the point of
        packing the corpus in the first place.
        """
        xs, ys, ms = zip(*batch)
        return torch.stack(xs), torch.stack(ys), torch.stack(ms)

    def __init__(self, path: str, seq_len: int = 2048, limit_tokens: int = 0):
        manifest_path = os.path.join(path, "manifest.json")
        with open(manifest_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        self.seq_len = seq_len
        self.dir = path
        self.shards, self.starts, total = [], [], 0
        for entry in meta["shards"]:
            n = int(entry["tokens"])
            if limit_tokens and total >= limit_tokens:
                break
            self.shards.append((entry["path"], n))
            # One extra token is needed for the shifted target, so the last
            # partial window is dropped rather than wrapping into the next
            # shard.
            self.starts.append((total, max(0, (n - 1) // seq_len)))
            total += n
        self.tokens = total
        self.windows = sum(w for _, w in self.starts)
        self._maps = {}

    def __len__(self):
        return self.windows

    def _map(self, idx):
        # Opened lazily and per worker: a memmap cannot be inherited across a
        # fork safely, and DataLoader workers each need their own handle.
        m = self._maps.get(idx)
        if m is None:
            name, _ = self.shards[idx]
            m = np.memmap(os.path.join(self.dir, name), dtype=np.uint16, mode="r")
            self._maps[idx] = m
        return m

    def __getitem__(self, index):
        for shard_idx, (_, n_windows) in enumerate(self.starts):
            if index < n_windows:
                break
            index -= n_windows
        m = self._map(shard_idx)
        lo = index * self.seq_len
        chunk = np.asarray(m[lo:lo + self.seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y, torch.ones(self.seq_len, dtype=torch.float32)

    def describe(self) -> str:
        return (f"{self.tokens/1e9:.2f}B tokens, {len(self.shards)} shards, "
                f"{self.windows:,} windows of {self.seq_len}")
