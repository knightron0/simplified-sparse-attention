"""GistMaskSpec must reproduce the reference dense gist masks exactly.

    python tests/test_gist_mask_lazy.py            # CPU, a few seconds
    python tests/test_gist_mask_lazy.py --real     # also real gist-inserted inputs (downloads tokenizers)
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from gist_mask_lazy import GistMaskSpec, _FLEX_BLOCK  # noqa: E402
from src.data import gist  # noqa: E402

PAD = 0


def reference_final(ids, gist_ids, sink, P, attn2d):
    """Reference prefill mask exactly as GistLlamaModel.forward builds it:
    causal(+2D padding) AND gist mask."""
    fn = gist.make_gist_mask if len(gist_ids) == 1 else gist.make_hierarchical_gist_mask_greedy
    g = fn(inputs=ids, gist_token=gist_ids, chunk_size=4, attention_sink_size=sink,
           num_previous_chunks=P, pad_token=PAD, add_raw=False).bool()
    N = ids.shape[1]
    causal = torch.ones(N, N).tril().bool() & attn2d[0].bool()[None, :]
    return g, causal[None, None] & g


def block_mask_dense(bm, N):
    """Reconstruct the element mask a flex BlockMask encodes (partial blocks via mask_mod)."""
    nq, nk = bm.kv_num_blocks.shape[-1], bm.kv_indices.shape[-1]
    out = torch.zeros(nq * _FLEX_BLOCK, nk * _FLEX_BLOCK, dtype=torch.bool)
    b = h = torch.zeros((), dtype=torch.long)
    for qb in range(nq):
        q = torch.arange(qb * _FLEX_BLOCK, (qb + 1) * _FLEX_BLOCK)
        for kb in bm.kv_indices[0, 0, qb, : bm.kv_num_blocks[0, 0, qb]].tolist():
            k = torch.arange(kb * _FLEX_BLOCK, (kb + 1) * _FLEX_BLOCK)
            out[q[:, None], k[None, :]] = bm.mask_mod(b, h, q[:, None], k[None, :])
        if bm.full_kv_num_blocks is not None:
            for kb in bm.full_kv_indices[0, 0, qb, : bm.full_kv_num_blocks[0, 0, qb]].tolist():
                k = torch.arange(kb * _FLEX_BLOCK, (kb + 1) * _FLEX_BLOCK)
                out[q[:, None], k[None, :]] = True
    return out[:N, :N]


def check(ids, gist_ids, sink, P, attn2d=None, tag=""):
    N = ids.shape[1]
    if attn2d is None:
        attn2d = torch.ones_like(ids)
    g_ref, final_ref = reference_final(ids, gist_ids, sink, P, attn2d)
    spec = GistMaskSpec(ids, gist_ids, sink, P, PAD, attn2d)
    assert torch.equal(spec.dense(), final_ref), f"{tag}: final mask mismatch"
    assert torch.equal(spec.gist_last_row(), g_ref[:, :, -1:]), f"{tag}: last gist row mismatch"
    q = torch.tensor([[0, N // 3, N - 2, N - 1]])
    assert torch.equal(spec.rows(q), final_ref[:, :, q[0]]), f"{tag}: rows mismatch"
    causal_ref = (torch.ones(N, N).tril().bool() & attn2d[0].bool()[None])[None, None]
    assert torch.equal(spec.causal_rows(q), causal_ref[:, :, q[0]]), f"{tag}: causal rows mismatch"
    # block mask encodes exactly the permuted final mask
    perm = spec.perm()
    assert sorted(perm.tolist()) == list(range(N)), f"{tag}: perm is not a permutation"
    got = block_mask_dense(spec.flex_block_mask(), N)
    assert torch.equal(got, final_ref[0, 0][:, perm]), f"{tag}: block mask mismatch"


def synthetic(N, gist_ids, chunk, meta, pad_at=None, rng=None):
    """Raw tokens with a level-0 gist after ~`chunk` raws (jittered) and a level-1
    gist after every `meta` level-0 gists, like the inserters produce."""
    toks, n0 = [], 0
    while len(toks) < N:
        toks += [rng.randint(5, 999) for _ in range(max(1, chunk + rng.randint(-1, 1)))]
        toks.append(gist_ids[0]); n0 += 1
        if len(gist_ids) > 1 and n0 % meta == 0:
            toks.append(gist_ids[1])
    toks = toks[:N] + [rng.randint(5, 999) for _ in range(7)]   # question suffix
    if pad_at is not None:
        toks[pad_at] = PAD
    return torch.tensor([toks])


def main():
    rng = random.Random(0)
    n = 0
    for gist_ids in ([1000], [1000, 1001]):
        for sink in (0, 3):
            for P in (0, 1, 2):
                for N in (37, 300, 700):
                    for pad_at in (None, 5):
                        ids = synthetic(N, gist_ids, 4, 4, pad_at, rng)
                        check(ids, gist_ids, sink, P, tag=f"syn levels={len(gist_ids)} sink={sink} P={P} N={N} pad={pad_at}")
                        n += 1
        # 2D attention-mask padding and a sequence with no gists at all
        ids = synthetic(200, gist_ids, 4, 4, None, rng)
        a2 = torch.ones_like(ids); a2[0, :3] = 0
        check(ids, gist_ids, 3, 1, attn2d=a2, tag="2d-pad"); n += 1
        check(torch.randint(5, 999, (1, 90)), gist_ids, 3, 1, tag="no-gist"); n += 1

    if "--real" in sys.argv:
        from transformers import AutoTokenizer
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark"))
        import latency_sweep as ls
        from gist_utils import Global_data
        for cfg in ("ssa", "hssa"):
            path, chunk, meta = ls.CONFIGS[cfg]
            tok = AutoTokenizer.from_pretrained(path)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            nl = 2 if meta else 1
            Global_data.gist_token_id = tok.additional_special_tokens_ids[:nl]
            Global_data.chunk_size = chunk
            Global_data.sink_size = 3; Global_data.num_previous_chunks = 1; Global_data.add_raw = False
            inp = ls.build_inputs(cfg, tok, 1500, device="cpu", lazy_mask=False)
            ids, g_ref = inp["input_ids"], inp["attention_mask_gist"].bool()
            spec = GistMaskSpec(ids, Global_data.gist_token_id, 3, 1, tok.pad_token_id)
            N = ids.shape[1]
            causal = torch.ones(N, N).tril().bool()[None, None]
            assert torch.equal(spec.dense(), causal & g_ref), f"real {cfg}: mismatch"
            assert torch.equal(spec.gist_last_row(), g_ref[:, :, -1:]), f"real {cfg}: last row mismatch"
            got = block_mask_dense(spec.flex_block_mask(), N)
            assert torch.equal(got, (causal & g_ref)[0, 0][:, spec.perm()]), f"real {cfg}: block mask mismatch"
            print(f"real {cfg}: N={N} OK")
            n += 1
    print(f"PASS: {n} cases identical to the reference masks")


if __name__ == "__main__":
    main()
