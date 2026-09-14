"""Prove a GGUF export is the same model: load it into the Ollama docker
instance and compare greedy output with PyTorch on the identical prompt.

Three things are checked, each of which has broken real exports:
  1. ``ollama create`` accepts the file (metadata / tensor layout is valid);
  2. the prompt tokenizes to the same count in llama.cpp and here
     (``prompt_eval_count`` vs our tokenizer) - catches pre-tokenizer drift;
  3. greedy continuations agree for a prefix of tokens - catches RoPE layout,
     norm, bias and MoE routing mismatches, which all show up within a few
     tokens.

    .venv/bin/python scripts/test_export_gguf.py checkpoints_X/agent_best.pt \
        --gguf checkpoints_X/newllm.gguf --container ollama-4090 --name newllm-test
"""

import argparse
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, "/home/lmeadows/llm")

from agent.chat import load_checkpoint, load_tokenizer_for
from agent.chatml import render
from agent.generate import generate_tokens
from sampling import Sampler

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--container", default="ollama-4090")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--name", default="newllm-test")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--min-match", type=int, default=8)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    install(args.gguf, args.container, args.name, args.num_ctx)

    model = load_checkpoint(args.checkpoint, device=args.device).float()
    tok = load_tokenizer_for(model)
    messages = [{"role": "system", "content": "You are Zed."},
                {"role": "user", "content": "What is 12 + 8?"}]
    prompt, _ = render(messages, None, add_generation_prompt=True)
    ids = tok.encode(prompt)

    # PyTorch greedy, f32.
    sampler = Sampler(temperature=0.01, top_k=1, top_p=1.0)
    ours = list(generate_tokens(model, ids, sampler, max_new=args.tokens,
                                device=args.device, tokenizer=tok,
                                stop_on_assistant_close=False, stop_texts=()))
    ours_text = tok.decode(ours)

    # llama.cpp greedy through Ollama, raw prompt (no template applied).
    import ollama
    client = ollama.Client(host=args.host)
    r = client.generate(model=args.name, prompt=prompt, raw=True, stream=False,
                        options={"temperature": 0, "num_predict": args.tokens,
                                 "seed": 0, "top_k": 1, "num_ctx": args.num_ctx})
    theirs_text = r.response or ""
    theirs = tok.encode(theirs_text)

    ok_tok = (r.prompt_eval_count == len(ids))
    print(f"prompt tokens: ours {len(ids)}, llama.cpp {r.prompt_eval_count}  "
          f"{'OK' if ok_tok else 'MISMATCH'}")
    match = 0
    for a, b in zip(ours, theirs):
        if a != b:
            break
        match += 1
    print(f"greedy agreement: {match}/{min(len(ours), len(theirs))} tokens")
    print(f"  torch : {ours_text[:120]!r}")
    print(f"  gguf  : {theirs_text[:120]!r}")
    ok = ok_tok and match >= args.min_match
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
