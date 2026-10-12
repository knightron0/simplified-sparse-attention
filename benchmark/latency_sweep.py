"""End-to-end latency sweep (TTFT / TPOT vs context length) for Llama-3.2-1B.

Reproduces the setup of the paper's Figure 3 (end-to-end latency scaling) on a
smaller model: dense FlashAttention baseline vs SSA (single-level) vs H-SSA
(hierarchical), batch 1, bf16, greedy decoding of a fixed number of tokens.

One model per invocation; rows are appended to a CSV so the three configs can be
run separately (and resumed) and then plotted with benchmark/plot_latency.py:

    python benchmark/latency_sweep.py --config dense --out results/latency.csv
    python benchmark/latency_sweep.py --config ssa   --out results/latency.csv
    python benchmark/latency_sweep.py --config hssa  --out results/latency.csv

`context_len` is the number of raw document tokens (before gist insertion); the
gist models see `input_len` = context + inserted gist tokens + question suffix.
Contexts are built from the bundled NQ-open passages (benchmark/RAG_exp).

Timing: CUDA events recorded at every generate step (same method as
longbench_pred_qwen.py): TTFT = first step (prefill), TPOT = mean of the rest.
Each length gets `--warmup` untimed runs (CUDA-extension build, flex compile)
then `--repeats` timed runs; the CSV keeps every timed run.
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
)

CONFIGS = {
    # name: (checkpoint, chunk_size, meta_gist)
    # dense: the gist checkpoint loaded as a vanilla LlamaForCausalLM on plain text
    # (no gist tokens) -- same architecture/latency as meta-llama/Llama-3.2-1B (+3
    # vocab rows) without needing gated-repo access. Override with --ckpt_path.
    "dense": ("gist-sparse-attention/GSA-FT-Llama-3.2-1B-chunk16", None, None),
    "ssa": ("gist-sparse-attention/GSA-FT-Llama-3.2-1B-chunk16", [16], 0),
    "hssa": ("gist-sparse-attention/GSA-FT-Llama-3.2-1B-chunk4-chunk4", [4, 4], 1),
}

DEFAULT_LENGTHS = [4096, 8192, 12288, 16384, 24576, 32768]
DATA_PATH = os.path.join(REPO_ROOT, "benchmark", "RAG_exp", "nq-open-10_0.jsonl")


class CudaTimingCriteria:
    """Records a CUDA event per generate step (copied from longbench_pred_qwen.py)."""

    def __init__(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.events = [e]

    def __call__(self, input_ids, scores, **kwargs):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.events.append(e)
        return False

    def prefill_and_tpot_ms(self):
        torch.cuda.synchronize()
        ds = [self.events[i - 1].elapsed_time(self.events[i]) for i in range(1, len(self.events))]
        if not ds:
            return 0.0, 0.0
        return ds[0], (sum(ds[1:]) / len(ds[1:]) if len(ds) > 1 else 0.0)


class SuppressGistTokens(LogitsProcessor):
    """Gist ids must never be generated: the sparse decode cache extends by one
    non-gist column per step (see attn_candidate._decode_forward)."""

    def __init__(self, gist_token_ids):
        self.ids = [int(g) for g in gist_token_ids]

    def __call__(self, input_ids, scores):
        if self.ids:
            scores[:, self.ids] = float("-inf")
        return scores


def parse_args(cmd_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--ckpt_path", default=None, help="Override the checkpoint for --config.")
    ap.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS,
                    help="Raw context lengths in tokens (before gist insertion).")
    ap.add_argument("--gen_tokens", type=int, default=64, help="Decode steps per run (forced, no early EOS).")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--attn_impl", choices=["sparse", "ref"], default="sparse",
                    help="Gist models only: optimized sparse path or eager reference.")
    ap.add_argument("--mask", choices=["lazy", "dense"], default="lazy",
                    help="Gist models, sparse path: O(N) lazy masks or the original [N, N] tensors.")
    ap.add_argument("--dense_attn", default="flash_attention_2",
                    help="attn_implementation for the dense baseline (falls back to sdpa if flash-attn is missing).")
    ap.add_argument("--top_k", type=int, nargs="+", default=[3])
    ap.add_argument("--sink_size", type=int, default=3)
    ap.add_argument("--num_previous_chunks", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "latency_llama1b.csv"))
    return ap.parse_args(cmd_args)


# ---------------------------------------------------------------------------
# model setup
# ---------------------------------------------------------------------------

def load_model(config, ckpt_path, args):
    """Returns (model, tokenizer, attn_label). Configures Global_data for gist models
    exactly like longbench_pred.py / longbench_pred_qwen.py."""
    path = ckpt_path or CONFIGS[config][0]
    tokenizer = AutoTokenizer.from_pretrained(path)
    if config == "dense":
        attn = args.dense_attn
        if attn == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[latency_sweep] flash-attn not installed; dense baseline falls back to sdpa", flush=True)
                attn = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn,
        ).eval()
        return model, tokenizer, attn

    import gist_llama
    from gist_llama import GistLlamaForCausalLM
    from gist_utils import Global_data

    _, chunk_size, meta_gist = CONFIGS[config]
    model = GistLlamaForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    model.selective = True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    if len(tokenizer) != gist_llama.PRETRAINED_VOCAB_SIZE + 3:
        raise SystemExit(f"{path} does not look like a gist checkpoint (len(tokenizer)={len(tokenizer)})")

    num_gist_level = 2 if meta_gist else 1
    Global_data.gist_token_id = tokenizer.additional_special_tokens_ids[:num_gist_level]
    Global_data.gist_token = tokenizer.additional_special_tokens[:num_gist_level]
    top_k = list(args.top_k)
    Global_data.top_k = (top_k * 3)[:3] if len(top_k) == 1 else (top_k + top_k[-1:])[:3]
    Global_data.top_p = 0
    Global_data.sink_size = args.sink_size
    Global_data.pad_token_id = tokenizer.pad_token_id
    Global_data.use_nsa_gqa = True
    Global_data.use_grouped_sdpa_optimization = False
    Global_data.chunk_size = chunk_size
    Global_data.num_previous_chunks = args.num_previous_chunks
    Global_data.sep_num = 0
    Global_data.add_raw = False
    Global_data.kl_loss = False

    # gist_llama reads ATTN_IMPL on every forward; default there is the eager ref.
    os.environ["ATTN_IMPL"] = args.attn_impl
    label = args.attn_impl
    if args.attn_impl == "sparse":
        prefill = "lazy-flex" if args.mask == "lazy" else os.environ.get("PREFILL_KERNEL", "dense")
        label += f"/prefill={prefill}/select={os.environ.get('DECODE_SELECT', 'head')}"
    return model, tokenizer, label


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

_PASSAGE_IDS = None


def _passage_token_stream(tokenizer, needed):
    """Concatenated NQ-open passages, tokenized once and cached (grown on demand)."""
    global _PASSAGE_IDS
    if _PASSAGE_IDS is not None and len(_PASSAGE_IDS) >= needed:
        return _PASSAGE_IDS
    ids = []
    with open(DATA_PATH) as f:
        for line in f:
            for c in json.loads(line)["ctxs"]:
                ids.extend(tokenizer(f"(Title: {c['title']}) {c['text']}\n", add_special_tokens=False).input_ids)
            if len(ids) >= needed:
                break
    if len(ids) < needed:
        raise SystemExit(f"bundled passages only provide {len(ids)} tokens; asked for {needed}")
    _PASSAGE_IDS = ids
    return ids


def build_inputs(config, tokenizer, context_len, device="cuda", lazy_mask=None):
    """Returns generate() kwargs (input_ids, attention_mask[, attention_mask_gist]).

    lazy_mask (gist configs): pass only the last row of the gist mask, so the model
    evaluates masks on demand (gist_mask_lazy.py) instead of building [N, N]
    tensors. Defaults to on when ATTN_IMPL=sparse (the eager reference needs the
    dense mask)."""
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    doc = bos + _passage_token_stream(tokenizer, context_len)[:context_len]
    suffix = tokenizer("\n\nQuestion: Summarize the documents above.\nAnswer:", add_special_tokens=False).input_ids

    if config == "dense":
        ids = doc + suffix
        input_ids = torch.tensor([ids], device=device)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    from gist_utils import Global_data
    from src.data import gist

    if len(Global_data.gist_token_id) == 1:
        doc_with_gist = gist.insert_gist_tokens_word_aware(
            doc, Global_data.gist_token_id, Global_data.chunk_size[0], tokenizer, Global_data.sink_size,
        )
        gist_fn = gist.make_gist_mask
    else:
        doc_with_gist = gist.insert_hierarchical_gist_tokens(
            doc, Global_data.gist_token_id, Global_data.chunk_size[0], Global_data.chunk_size[1],
            tokenizer, Global_data.sink_size,
        )
        gist_fn = gist.make_hierarchical_gist_mask_greedy
    input_ids = torch.tensor([doc_with_gist + suffix], device=device)
    if lazy_mask is None:
        lazy_mask = os.environ.get("ATTN_IMPL", "ref") == "sparse"
    if lazy_mask:
        from gist_mask_lazy import GistMaskSpec
        spec = GistMaskSpec(input_ids, Global_data.gist_token_id, Global_data.sink_size,
                            Global_data.num_previous_chunks, tokenizer.pad_token_id)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "attention_mask_gist": spec.gist_last_row(),
        }
    attention_mask_gist = gist_fn(
        inputs=input_ids,
        gist_token=Global_data.gist_token_id,
        chunk_size=Global_data.chunk_size[0],
        attention_sink_size=Global_data.sink_size,
        num_previous_chunks=Global_data.num_previous_chunks,
        pad_token=tokenizer.pad_token_id,
        add_raw=Global_data.add_raw,
    ).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "attention_mask_gist": attention_mask_gist,
    }


def generate_timed(model, tokenizer, inputs, gen_tokens, gist_ids=()):
    """One greedy generation of exactly `gen_tokens` tokens. Returns (out_ids, ttft_ms, tpot_ms)."""
    timer = CudaTimingCriteria()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=gen_tokens,
            min_new_tokens=gen_tokens,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([SuppressGistTokens(gist_ids)]),
            stopping_criteria=StoppingCriteriaList([timer]),
        )[0]
    ttft, tpot = timer.prefill_and_tpot_ms()
    return out, ttft, tpot


def gist_ids_for(config):
    if config == "dense":
        return []
    from gist_utils import Global_data
    return list(Global_data.gist_token_id)


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

FIELDS = ["config", "ckpt", "attn", "gpu", "context_len", "input_len", "gen_tokens",
          "run", "ttft_ms", "tpot_ms", "peak_mem_mb", "status", "timestamp"]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(0)
    ckpt = args.ckpt_path or CONFIGS[args.config][0]
    model, tokenizer, attn_label = load_model(args.config, args.ckpt_path, args)
    gist_ids = gist_ids_for(args.config)
    gpu = torch.cuda.get_device_name(0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_file = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    def emit(**row):
        row.update(config=args.config, ckpt=ckpt, attn=attn_label, gpu=gpu,
                   gen_tokens=args.gen_tokens, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        writer.writerow(row)
        f.flush()

    print(f"[latency_sweep] {args.config} ({ckpt}) attn={attn_label} gpu={gpu}", flush=True)
    for L in sorted(args.lengths):
        try:
            inputs = build_inputs(args.config, tokenizer, L,
                                  lazy_mask=(args.mask == "lazy") if args.attn_impl == "sparse" else False)
            input_len = inputs["input_ids"].shape[1]
            for _ in range(args.warmup):
                generate_timed(model, tokenizer, inputs, args.gen_tokens, gist_ids)
            ttfts, tpots = [], []
            for r in range(args.repeats):
                torch.cuda.reset_peak_memory_stats()
                _, ttft, tpot = generate_timed(model, tokenizer, inputs, args.gen_tokens, gist_ids)
                peak = torch.cuda.max_memory_allocated() / 2**20
                ttfts.append(ttft); tpots.append(tpot)
                emit(context_len=L, input_len=input_len, run=r, ttft_ms=f"{ttft:.3f}",
                     tpot_ms=f"{tpot:.3f}", peak_mem_mb=f"{peak:.0f}", status="ok")
            print(f"  ctx={L:>6} input={input_len:>6}  TTFT median {statistics.median(ttfts):9.2f} ms"
                  f"  TPOT median {statistics.median(tpots):7.2f} ms", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  ctx={L:>6}  OOM -- skipping this and longer lengths", flush=True)
            emit(context_len=L, input_len="", run="", ttft_ms="", tpot_ms="", peak_mem_mb="", status="oom")
            break
        finally:
            inputs = None
            torch.cuda.empty_cache()
    f.close()


if __name__ == "__main__":
    main()
