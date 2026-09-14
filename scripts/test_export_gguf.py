"""Prove a GGUF export is the same model: load it into the Ollama docker
instance and compare next-token DISTRIBUTIONS with PyTorch.

Why distributions and not greedy text: greedy tokens diverge on rounding
noise whenever the logits are flat, and once one token differs the rest is a
different sequence. Logprobs on the same prefix are directly comparable and
Ollama returns them (``logprobs`` / ``top_logprobs``, Ollama >= 0.12).

Positions whose argmax is a CONTROL token (``<think>`` right after the
assistant tag) cannot be compared through Ollama: its thinking parser
consumes the token and returns no logprob entry for it. So every compared
position is placed just *inside* the thought, where the next token is an
ordinary word, and the ``<think>`` prediction itself is checked indirectly -
PyTorch's argmax must be ``<think>`` and Ollama must report a non-empty
``thinking`` field for the same prompt.

Two such positions, because different bugs show at different places:
  * a short prompt (~30 tokens);
  * after a ~300-token rendered conversation - a RoPE layout or frequency
    mismatch is invisible at 30 tokens and unmistakable at 300.

Pass criteria per prompt: same argmax, and the top-3 logprobs agree within
``--tol`` (f16 KV cache and kernel order account for a few hundredths).

    .venv/bin/python scripts/test_export_gguf.py checkpoints_X/agent_best.pt \\
        --gguf checkpoints_X/newllm.gguf --container ollama-4090 --name newllm-test
"""

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request

import torch

sys.path.insert(0, "/home/lmeadows/llm")

from agent.chat import load_checkpoint, load_tokenizer_for
from agent.chatml import render

OLLAMA_TEMPLATE = "/home/lmeadows/llm/data/templates/qwen3.ollama.tmpl"


def sh(*cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"$ {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return r.stdout.strip()


def install(gguf: str, container: str, name: str, num_ctx: int) -> None:
    """docker cp the GGUF + a Modelfile into the container and create the model."""
    remote_dir = "/root/newllm"
    sh("docker", "exec", container, "mkdir", "-p", remote_dir)
    sh("docker", "cp", gguf, f"{container}:{remote_dir}/{name}.gguf")
    template = open(OLLAMA_TEMPLATE).read()
    modelfile = (f"FROM {remote_dir}/{name}.gguf\n"
                 f'TEMPLATE """{template}"""\n'
                 'PARAMETER stop "<|im_start|>"\nPARAMETER stop "<|im_end|>"\n'
                 f"PARAMETER num_ctx {num_ctx}\n")
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as fh:
        fh.write(modelfile)
        local = fh.name
    sh("docker", "cp", local, f"{container}:{remote_dir}/{name}.Modelfile")
    out = sh("docker", "exec", container, "ollama", "create", name, "-f",
             f"{remote_dir}/{name}.Modelfile")
    print(out.splitlines()[-1] if out else "created")


def llama_top(host, name, prompt, num_ctx, k=10):
    """(prompt_eval_count, [(bytes, logprob), ...]) for the next token."""
    body = json.dumps({"model": name, "prompt": prompt, "raw": True, "stream": False,
                       "logprobs": True, "top_logprobs": k,
                       "options": {"temperature": 0, "num_predict": 1, "seed": 0,
                                   "num_ctx": num_ctx}}).encode()
    req = urllib.request.Request(host + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    if "error" in d:
        sys.exit(f"ollama: {d['error']}")
    entries = (d.get("logprobs") or [{}])[0].get("top_logprobs") or []
    return d.get("prompt_eval_count"), [(bytes(e.get("bytes") or []), float(e["logprob"]))
                                        for e in entries]


def torch_top(model, tok, ids, device, k=10):
    with torch.no_grad():
        out = model(torch.tensor([ids], device=device))
        logits = out[0] if isinstance(out, tuple) else out
        logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
    top = torch.topk(logp, k)
    rows = []
    for v, i in zip(top.values, top.indices):
        i = int(i)
        piece = tok.decode([i])
        # llama.cpp emits CONTROL tokens as empty pieces.
        b = b"" if tok.is_special(i) else piece.encode("utf-8")
        rows.append((b, piece, float(v)))
    return rows


def compare(label, ours, theirs, tol):
    theirs_map = dict(theirs)
    top1_ok = bool(theirs) and ours[0][0] == theirs[0][0]
    diffs = []
    for b, piece, v in ours[:3]:
        if b in theirs_map:
            diffs.append(abs(v - theirs_map[b]))
        else:
            diffs.append(float("inf"))
    worst = max(diffs)
    overlap = sum(1 for b, _, _ in ours if b in theirs_map)
    status = "OK " if top1_ok and worst < tol else "BAD"
    print(f"  [{status}] {label}: argmax {'same' if top1_ok else 'DIFFERENT'}, "
          f"top-3 max |dlogprob| {worst:.3f}, top-10 overlap {overlap}/10")
    for b, piece, v in ours[:4]:
        t = theirs_map.get(b)
        print(f"        {piece!r:18s} torch {v:8.4f}   llama.cpp "
              f"{t if t is None else round(t, 4)}")
    return top1_ok and worst < tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--container", default="ollama-4090")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--name", default="newllm-test")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--tol", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    if not args.skip_install:
        install(args.gguf, args.container, args.name, args.num_ctx)
    model = load_checkpoint(args.checkpoint, device=args.device).float()
    tok = load_tokenizer_for(model)

    short = [{"role": "system", "content": "You are Zed."},
             {"role": "user", "content": "What is 12 + 8?"}]
    long_conv = [{"role": "system", "content": "You are Zed, a gruff engineer."}]
    for i in range(6):
        long_conv += [{"role": "user", "content": f"Question {i}: what is {i * 7} + {i * 3}? "
                                                  "Explain briefly and then give the number."},
                      {"role": "assistant", "content": f"That is {i * 10}.",
                       "reasoning_content": "Adding the two terms."}]
    long_conv.append({"role": "user", "content": "And now, what is 12 + 8?"})

    prompts = [
        ("short, inside the thought", render(short, None, True)[0] + "<think>\nI should"),
        ("long conversation, inside the thought",
         render(long_conv, None, True)[0] + "<think>\nI should"),
    ]
    ok_all = True
    # The <think> prediction itself. Ollama consumes a generated CONTROL token
    # and reports no logprob entry for it (eval_count is one more than the
    # number of entries), so the check is: PyTorch's argmax is <think>, Ollama
    # shows exactly that signature, and Ollama's first *reported* token equals
    # PyTorch's argmax at the position right after <think>.
    for label, conv in (("short", short), ("long", long_conv)):
        prompt = render(conv, None, True)[0]
        ids = tok.encode(prompt)
        ours = torch_top(model, tok, ids, args.device)
        after = torch_top(model, tok, ids + [tok.special_id("<think>")], args.device)
        body = json.dumps({"model": args.name, "prompt": prompt, "raw": True,
                           "stream": False, "logprobs": True, "top_logprobs": 1,
                           "options": {"temperature": 0, "seed": 0,
                                       "num_predict": 2, "num_ctx": args.num_ctx}}).encode()
        req = urllib.request.Request(args.host + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        entries = d.get("logprobs") or []
        consumed = d.get("eval_count", 0) - len(entries)
        first = bytes(entries[0].get("bytes") or []) if entries else None
        think_ok = (ours[0][1] == "<think>" and consumed == 1
                    and first is not None and first == after[0][0])
        print(f"  [{'OK ' if think_ok else 'BAD'}] {label}: torch argmax {ours[0][1]!r} "
              f"({ours[0][2]:.4f}); ollama consumed {consumed} control token, first "
              f"reported {first!r} vs torch-after-<think> {after[0][1]!r}")
        ok_all &= think_ok
    for label, prompt in prompts:
        ids = tok.encode(prompt)
        n, theirs = llama_top(args.host, args.name, prompt, args.num_ctx)
        tok_ok = (n == len(ids))
        print(f"{label}: {len(ids)} tokens (llama.cpp counted {n}) "
              f"{'OK' if tok_ok else 'TOKENIZER MISMATCH'}")
        ok = compare(label, torch_top(model, tok, ids, args.device), theirs, args.tol)
        ok_all &= ok and tok_ok
    print("PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
