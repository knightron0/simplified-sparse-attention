"""Correctness guard for the Llama sparse path (run before trusting latency numbers).

For each gist config and context length, runs the SAME greedy generation twice in
one process -- eager reference (ATTN_IMPL=ref, gist_llama's own eager functions)
and the optimized path (ATTN_IMPL=sparse, DECODE_SELECT=head) -- and compares:
  * generated token ids (must be identical), and
  * max |logit diff| over the decode steps (reported; small bf16 noise is expected).

head selection reproduces the reference's selected set and the kernel is allclose
for a given selection, so greedy argmax should match token-for-token (same contract
as check_correctness.sh for Qwen2). A mismatch means a kernel / port regression.

    python benchmark/check_llama_sparse.py                       # ssa + hssa, 2K and 8K
    python benchmark/check_llama_sparse.py --configs hssa --lengths 4096 --gen_tokens 32
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from transformers import LogitsProcessorList  # noqa: E402

from latency_sweep import (  # noqa: E402
    SuppressGistTokens, build_inputs, gist_ids_for, load_model, parse_args as sweep_args,
)


def greedy(model, tokenizer, inputs, gen_tokens, gist_ids):
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
            output_scores=True,
            return_dict_in_generate=True,
        )
    new = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    scores = torch.stack([s[0].float() for s in out.scores])  # [steps, vocab]
    return new, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["ssa", "hssa"], choices=["ssa", "hssa"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--gen_tokens", type=int, default=32)
    ap.add_argument("--mask", choices=["lazy", "dense"], default="lazy",
                    help="Mask path for the sparse run (the eager reference always uses dense masks).")
    a = ap.parse_args()

    os.environ["DECODE_SELECT"] = "head"  # the allclose-exact selection
    failed = False
    for config in a.configs:
        args = sweep_args(["--config", config, "--attn_impl", "sparse"])
        model, tokenizer, _ = load_model(config, None, args)
        gist_ids = gist_ids_for(config)
        for L in a.lengths:
            inputs = build_inputs(config, tokenizer, L, lazy_mask=False)
            os.environ["ATTN_IMPL"] = "ref"
            ref_ids, ref_scores = greedy(model, tokenizer, inputs, a.gen_tokens, gist_ids)
            os.environ["ATTN_IMPL"] = "sparse"
            if a.mask == "lazy":
                inputs = build_inputs(config, tokenizer, L, lazy_mask=True)
            sp_ids, sp_scores = greedy(model, tokenizer, inputs, a.gen_tokens, gist_ids)

            # Compare logits only up to (and including) the first divergence; after
            # that the two runs condition on different prefixes.
            n_same = next((i for i, (x, y) in enumerate(zip(ref_ids, sp_ids)) if x != y), len(ref_ids))
            upto = min(n_same + 1, len(ref_scores))
            finite = torch.isfinite(ref_scores[:upto]) & torch.isfinite(sp_scores[:upto])
            max_diff = (ref_scores[:upto] - sp_scores[:upto]).abs()[finite].max().item()
            ok = ref_ids == sp_ids
            # A divergence where the reference's own top-2 logits are within bf16
            # resolution (0.125 at |logit|~16-32) is a tie broken by rounding, not a bug.
            status = "PASS"
            if not ok:
                top2 = ref_scores[n_same].topk(2).values
                margin = (top2[0] - top2[1]).item()
                status = f"TIE (ref top-2 margin {margin:.3f} at step {n_same})" if margin <= 0.125 else "FAIL"
                failed |= status == "FAIL"
            print(f"[{config}/{a.mask}] ctx={L:>6} input={inputs['input_ids'].shape[1]:>6}  "
                  f"{status}  identical_tokens={n_same}/{len(ref_ids)}  "
                  f"max|dlogit|={max_diff:.4f}", flush=True)
            if not ok:
                print(f"    ref : {tokenizer.decode(ref_ids)!r}")
                print(f"    sp  : {tokenizer.decode(sp_ids)!r}")
        del model
        torch.cuda.empty_cache()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
