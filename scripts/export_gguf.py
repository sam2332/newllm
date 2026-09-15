"""Export an ``arch_version=3`` checkpoint to GGUF for llama.cpp / Ollama.

The v3 architecture is isomorphic to Qwen3 (dense) or Qwen3-MoE, so the export
is a tensor rename plus metadata - no weight transforms:

    encoder.embedding.weight              -> token_embd.weight
    layers.N.norm1.weight                 -> blk.N.attn_norm.weight
    layers.N.attn.W_{q,k,v,o}.weight      -> blk.N.attn_{q,k,v,output}.weight
    layers.N.attn.{q,k}_norm.weight       -> blk.N.attn_{q,k}_norm.weight
    layers.N.norm2.weight                 -> blk.N.ffn_norm.weight
    layers.N.ff.w_{gate,up,down}.weight   -> blk.N.ffn_{gate,up,down}.weight
    layers.N.ff.router.weight             -> blk.N.ffn_gate_inp.weight
    layers.N.ff.experts.*.w_gate.weight   -> blk.N.ffn_gate_exps.weight  (stacked)
    norm.weight                           -> output_norm.weight
    decoder.hidden_to_logits.weight       -> output.weight

The tokenizer is written as GPT-2 byte-level BPE with the ``qwen2``
pre-tokenizer, which is exactly how it was trained, and the Qwen3 Jinja chat
template is embedded so llama.cpp's server renders tools the way training did.

    .venv/bin/python scripts/export_gguf.py checkpoints_X/agent_best.pt --out newllm.gguf
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/lmeadows/llm")
import gguf

from agent.tokenizer_registry import load_tokenizer

TEMPLATE_PATH = "/home/lmeadows/llm/data/templates/qwen3.jinja"


def tokenizer_tables(tok_path: str):
    """Tokens, types and merges from an HF tokenizer.json."""
    hf = json.load(open(tok_path))
    vocab = hf["model"]["vocab"]
    merges = hf["model"]["merges"]
    added = {a["content"]: a["id"] for a in hf.get("added_tokens", [])}
    size = max(max(vocab.values()), max(added.values(), default=-1)) + 1
    tokens = [""] * size
    types = [gguf.TokenType.NORMAL] * size
    for t, i in vocab.items():
        tokens[i] = t
    for t, i in added.items():
        tokens[i] = t
        types[i] = gguf.TokenType.CONTROL
    merges = [m if isinstance(m, str) else " ".join(m) for m in merges]
    return tokens, types, merges, added


def export(checkpoint: str, out: str, dtype: str = "f16", name: str = "newllm"):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg, sd = ckpt["config"], ckpt["model"]
    if cfg.get("arch_version", 1) < 3:
        sys.exit("export needs arch_version>=3 (no attention bias, half-split "
                 "RoPE, no embedding scale); this checkpoint is "
                 f"arch_version={cfg.get('arch_version')}")
    spec = cfg.get("tokenizer")
    if not spec or spec.get("kind") != "bpe":
        sys.exit("export needs a BPE checkpoint; the byte tokenizer has no GGUF form")
    tok = load_tokenizer(spec, os.path.dirname(os.path.abspath(checkpoint)))

    moe = bool(cfg.get("use_moe"))
    arch = "qwen3moe" if moe else "qwen3"
    n_layers = cfg["n_layers"]
    d_model, n_heads = cfg["d_model"], cfg["n_heads"]
    n_kv = cfg.get("n_kv_heads") or n_heads
    head_dim = d_model // n_heads
    # feed_forward_length is the SwiGLU hidden width actually used.
    ff_key = "layers.0.ff.experts.0.w_gate.weight" if moe else "layers.0.ff.w_gate.weight"
    n_ff = sd[ff_key].shape[0]

    w = gguf.GGUFWriter(out, arch)
    w.add_name(name)
    w.add_context_length(int(cfg.get("export_context_length", 65536)))
    w.add_embedding_length(d_model)
    w.add_block_count(n_layers)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_heads)
    w.add_head_count_kv(n_kv)
    w.add_key_length(head_dim)
    w.add_value_length(head_dim)
    w.add_rope_dimension_count(head_dim)
    w.add_rope_freq_base(float(cfg.get("rope_base", 10000.0)))
    w.add_layer_norm_rms_eps(float(cfg.get("norm_eps", 1e-6)))
    if moe:
        w.add_expert_count(cfg["num_experts"])
        w.add_expert_used_count(cfg["top_k"])
        w.add_expert_feed_forward_length(n_ff)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if dtype == "f16"
                    else gguf.LlamaFileType.ALL_F32)

    tokens, types, merges, added = tokenizer_tables(tok.path)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_eos_token_id(added["<|im_end|>"])
    w.add_pad_token_id(added["<|endoftext|>"])
    w.add_bos_token_id(added["<|endoftext|>"])
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)
    w.add_chat_template(open(TEMPLATE_PATH).read())

    np_dtype = np.float16 if dtype == "f16" else np.float32

    def put(gname, tensor, keep_f32=False):
        arr = tensor.detach().to(torch.float32).numpy()
        w.add_tensor(gname, arr.astype(np.float32 if keep_f32 else np_dtype))

    put("token_embd.weight", sd["encoder.embedding.weight"])
    for i in range(n_layers):
        p = f"layers.{i}."
        b = f"blk.{i}."
        put(b + "attn_norm.weight", sd[p + "norm1.weight"], keep_f32=True)
        put(b + "attn_q.weight", sd[p + "attn.W_q.weight"])
        put(b + "attn_k.weight", sd[p + "attn.W_k.weight"])
        put(b + "attn_v.weight", sd[p + "attn.W_v.weight"])
        put(b + "attn_output.weight", sd[p + "attn.W_o.weight"])
        put(b + "attn_q_norm.weight", sd[p + "attn.q_norm.weight"], keep_f32=True)
        put(b + "attn_k_norm.weight", sd[p + "attn.k_norm.weight"], keep_f32=True)
        put(b + "ffn_norm.weight", sd[p + "norm2.weight"], keep_f32=True)
        if moe:
            put(b + "ffn_gate_inp.weight", sd[p + "ff.router.weight"], keep_f32=True)
            for kind in ("gate", "up", "down"):
                stacked = torch.stack([sd[f"{p}ff.experts.{e}.w_{kind}.weight"]
                                       for e in range(cfg["num_experts"])])
                put(b + f"ffn_{kind}_exps.weight", stacked)
        else:
            put(b + "ffn_gate.weight", sd[p + "ff.w_gate.weight"])
            put(b + "ffn_up.weight", sd[p + "ff.w_up.weight"])
            put(b + "ffn_down.weight", sd[p + "ff.w_down.weight"])
    put("output_norm.weight", sd["norm.weight"], keep_f32=True)
    put("output.weight", sd["decoder.hidden_to_logits.weight"])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    size = os.path.getsize(out)
    print(f"wrote {out}: {arch}, {n_layers} layers, d_model {d_model}, "
          f"n_ff {n_ff}, vocab {len(tokens)}, {size/2**20:.0f} MiB")
    from agent.notify import notify
    notify(f":package: **GGUF exported** `{os.path.basename(out)}`\n"
           f"{arch}, {n_layers} layers, d_model {d_model}, vocab {len(tokens)}, "
           f"{size/2**20:.0f} MiB", tag="export", blocking=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    ap.add_argument("--name", default="newllm")
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.checkpoint) or ".",
                                   f"{args.name}.gguf")
    export(args.checkpoint, out, args.dtype, args.name)


if __name__ == "__main__":
    main()
