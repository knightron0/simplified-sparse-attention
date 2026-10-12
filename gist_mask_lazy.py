"""O(N)-memory gist attention masks for long-context prefill.

The reference pipeline materializes the gist mask as dense [N, N] tensors
(src/data/gist.py make_gist_mask / make_hierarchical_gist_mask_greedy, plus the
causal and float copies built in GistLlamaModel.forward), ~12 bytes per
(query, key) pair at peak -- ~14 GB at 32K, ~220 GB at 128K.

Every entry of those masks is an elementwise function of a few per-token vectors
(gist flags per level, chunk ids, section ids, padding), so `GistMaskSpec` stores
only those vectors and evaluates the rule on demand:

  * `allowed(i, j)`       -> exact mask value for broadcastable index tensors
  * `rows(q_pos)`         -> [B, 1, qb, N] mask rows for a few queries (prefill suffix)
  * `gist_last_row()`     -> [B, 1, 1, N] last row of the *gist* mask, which is all
                             the generation loop keeps for decoding
  * `flex_block_mask()`   -> flex BlockMask built in row chunks (never [N, N])

The rule reproduces the reference builders exactly for the inference settings
(B=1, attention_sink_size >= 0, no link tokens, add_raw=False);
tests/test_gist_mask_lazy.py checks it entry-for-entry against them.
"""

from __future__ import annotations

from typing import List, Optional

import torch

_FLEX_BLOCK = 128
_ROW_CHUNK = 2048  # query rows evaluated at once when building the block mask


def _chunk_ids(is_gist: torch.Tensor) -> torch.Tensor:
    """Gist tokens belong to the chunk before them (same as gist.py)."""
    c = is_gist.long().cumsum(-1)
    return torch.where(is_gist, c - 1, c).clamp_min(0)


class GistMaskSpec:
    def __init__(
        self,
        input_ids: torch.Tensor,
        gist_token_ids: List[int],
        attention_sink_size: int,
        num_previous_chunks: int,
        pad_token_id: Optional[int],
        attention_mask_2d: Optional[torch.Tensor] = None,
    ):
        assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "lazy gist mask is batch-1"
        assert attention_sink_size >= 0, "dynamic (negative) attention sinks are not supported"
        ids = input_ids[0]
        self.N = ids.shape[0]
        self.device = ids.device
        self.num_levels = len(gist_token_ids)
        self.sink = attention_sink_size
        self.P = num_previous_chunks
        self.pos = torch.arange(self.N, device=self.device)

        self.is_g = [ids == g for g in gist_token_ids]             # per level, [N] bool
        self.c = [_chunk_ids(g) for g in self.is_g]               # per level chunk ids
        # is_above[l]: token is a gist of any level > l; sec[l]: section id between
        # higher-level gists (used by the lower-level gist visibility rule).
        self.is_above, self.sec = [], []
        for lvl in range(self.num_levels):
            above = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            for h in range(lvl + 1, self.num_levels):
                above |= self.is_g[h]
            self.is_above.append(above)
            self.sec.append(above.long().cumsum(-1))
        self.has_gist = bool(torch.stack(self.is_g).any())
        # Padding enters twice in the reference: the gist mask drops pad-token keys,
        # the causal mask drops keys with attention_mask_2d == 0.
        ones = torch.ones(self.N, dtype=torch.bool, device=self.device)
        self.key_ok_gist = (ids != pad_token_id) if pad_token_id is not None else ones
        self.key_ok_causal = attention_mask_2d[0].bool() if attention_mask_2d is not None else ones
        self._block_mask = None
        self._perm = None

    # ------------------------------------------------------------------ rule
    def gist_allowed(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """Value of the reference *gist* mask (before AND-ing with causal)."""
        P = self.P
        causal = j <= i
        if self.num_levels == 1:
            # make_gist_mask(meta_gist=0) with a single gist level
            c = self.c[0]
            d = c[i] - c[j]
            m = (d == 0) | ((d > 0) & (d <= P)) | (self.is_g[0][j] & causal)
        else:
            # make_hierarchical_gist_mask_greedy
            L = self.num_levels
            m = self.is_g[L - 1][j] & causal
            for lvl in range(L - 2, -1, -1):
                m = m | (self.is_g[lvl][j] & causal & (self.sec[lvl][i] == self.sec[lvl][j]))
            d0 = self.c[0][i] - self.c[0][j]
            m = m | (((d0 == 0) | ((d0 > 0) & (d0 <= P))) & ~self.is_above[0][i])
            for lvl in range(1, L):
                cl = self.c[lvl]
                dl = cl[i] - cl[j]
                prev_g = self.is_g[lvl - 1][j]
                m = m | (self.is_g[lvl][i] & prev_g & (dl == 0))
                if P > 0:
                    m = m | ((dl > 0) & (dl <= P) & prev_g & causal & ~self.is_above[lvl][i])
        if self.sink > 0:
            m = m | (j < self.sink)
        if not self.has_gist:
            m = causal if self.num_levels > 1 else torch.ones_like(causal)
        return m & self.key_ok_gist[j]

    def allowed(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """Final prefill mask: causal AND padding AND gist mask."""
        return (j <= i) & self.key_ok_causal[j] & self.gist_allowed(i, j)

    # ------------------------------------------------------------ consumers
    def rows(self, q_pos: torch.Tensor) -> torch.Tensor:
        """[B, qb] query positions -> [B, 1, qb, N] bool rows of the final mask."""
        return self.allowed(q_pos[..., None], self.pos)[:, None]

    def causal_rows(self, q_pos: torch.Tensor) -> torch.Tensor:
        """Rows of the causal(+padding) mask alone (the reference `causal_mask`)."""
        return ((self.pos <= q_pos[..., None]) & self.key_ok_causal)[:, None]

    def gist_last_row(self) -> torch.Tensor:
        i = torch.tensor([self.N - 1], device=self.device)
        return self.gist_allowed(i[:, None], self.pos[None, :])[None, None]

    def dense(self) -> torch.Tensor:
        """Full [1, 1, N, N] final mask -- tests only."""
        return self.allowed(self.pos[:, None], self.pos[None, :])[None, None]

    # ---------------------------------------------------------------- flex
    def perm(self) -> torch.Tensor:
        """Key permutation: globally visible columns (top-level gists + sinks) first,
        then everything else, each in position order. Clusters the scattered global
        columns into a few dense key blocks so the rest is block-sparse. Purely a
        performance choice: the mask is evaluated on original positions."""
        if self._perm is None:
            glob = self.is_g[-1] | (self.pos < self.sink)
            self._perm = torch.cat([self.pos[glob], self.pos[~glob]])
        return self._perm

    def flex_block_mask(self):
        """BlockMask over (query, permuted key) built in row chunks: peak extra memory
        is _ROW_CHUNK x N bools, never N x N."""
        if self._block_mask is not None:
            return self._block_mask
        from torch.nn.attention.flex_attention import (
            _convert_mask_to_block_mask,
            _create_sparse_block_from_block_mask,
        )

        N, perm = self.N, self.perm()
        partial, full = [], []
        for r0 in range(0, N, _ROW_CHUNK):
            r1 = min(r0 + _ROW_CHUNK, N)
            m = self.allowed(self.pos[r0:r1, None], perm[None, :])   # [rows, N]
            p, f = _convert_mask_to_block_mask(
                m, Q_BLOCK_SIZE=_FLEX_BLOCK, KV_BLOCK_SIZE=_FLEX_BLOCK, separate_full_blocks=True,
            )
            partial.append(p); full.append(f)
        partial = torch.cat(partial, dim=2)
        full = torch.cat(full, dim=2)

        # A 0-d tensor, not a Python int: torch 2.6 inductor fails to lower length-
        # dependent ints captured by mask_mod once flex recompiles with dynamic shapes.
        last = torch.tensor(N - 1, device=self.device)

        def mask_mod(b, h, q_idx, kv_idx):
            # clamp keeps gathers in bounds on padded tail blocks; flex masks
            # out-of-range positions itself.
            i = torch.minimum(q_idx, last)
            j = perm[torch.minimum(kv_idx, last)]
            return self.allowed(i, j)

        self._block_mask = _create_sparse_block_from_block_mask(
            (partial, full), mask_mod, (N, N), _FLEX_BLOCK, _FLEX_BLOCK,
        )
        return self._block_mask
