"""V2 finetune sample builder (Phase D).

Reuses finetune/data.py's variation parser verbatim, bridges to the tested
V2 segment splice (dataloader/sequence_v2.py), and maps V1's flat-position
supervision onto the spliced (board->k latents) stream:

  - thinking_policy_head supervised at each `predict_from` position
    (start_think / end_var / board-stm for PV continuation)
  - policy_head supervised at end_think (final move)
  - wl_head at the move position, d_head at the wl position (V1 mechanism)
  - transition_head between every adjacent (board, move, next_board) triple
    in the trace, with optional scheduled sampling of the conditioning board

No real-data validation here (variation parquets are MCTS-generated /
gitignored); a synthetic-row smoke test exercises the full loss path.
"""
from __future__ import annotations

from chessdecoder.finetune.data import variation_to_token_ids
from chessdecoder.dataloader.sequence_v2 import variation_plan_from_token_ids
from chessdecoder.models.vocab import token_to_idx, full_idx_to_move_idx


def variation_to_v2_sample(row, max_variations=3, max_depth=5,
                           tau_base=0.3, tau_alpha=1.0,
                           use_backed_up_wdl=False):
    """row -> (plan, supervision). `plan` is the Seg list for
    build_mixed_sequence; `supervision` carries flat-position targets that the
    training loop translates to decoder indices via the returned plan's
    per-segment `meta["flat"]` / `meta["flat_start"]`."""
    (ids, thinking_move_data, final_move_data, value_data,
     block_boundaries, ranking, first_is_not_best,
     _maxd, _maxv) = variation_to_token_ids(
        row, max_variations=max_variations, max_depth=max_depth,
        tau_base=tau_base, tau_alpha=tau_alpha,
        use_backed_up_wdl=use_backed_up_wdl)

    vpos = {}
    for wl_p, d_p, wl, d, valid in value_data:
        vpos[wl_p] = (wl, valid, "wl")
        vpos[d_p] = (d, valid, "d")
    value_floats = {p: v[0] for p, v in vpos.items()}

    plan = variation_plan_from_token_ids(ids, block_boundaries, value_floats)

    # thinking-policy targets: flat predict_from -> move sub-vocab id
    think = [(p, full_idx_to_move_idx[token_to_idx[mv]])
             for p, mv in thinking_move_data
             if 0 <= p < len(ids)]
    final = None
    if final_move_data is not None:
        ep, ftok = final_move_data
        final = (ep, full_idx_to_move_idx[token_to_idx[ftok]])

    # transition triples: adjacent board segments + the move seg between them.
    board_idx = [i for i, s in enumerate(plan) if s.kind == "board"]
    triples = []
    for a, b in zip(board_idx, board_idx[1:]):
        mv = next((j for j in range(a + 1, b)
                   if plan[j].kind == "token"
                   and plan[j].token_id not in (
                       token_to_idx["wl_value"], token_to_idx["d_value"],
                       token_to_idx["end_var"], token_to_idx["end_think"],
                       token_to_idx["start_think"])), None)
        if mv is not None:
            triples.append((a, mv, b))         # (prev_board_seg, move_seg, next_board_seg)

    return plan, {
        "thinking": think,                     # [(flat_pos, move_sub_id)]
        "final": final,                        # (flat_pos, move_sub_id) | None
        "value": vpos,                         # flat_pos -> (val, valid, "wl"|"d")
        "transition_triples": triples,         # [(seg_i, seg_move, seg_next)]
        "first_is_not_best": first_is_not_best,
    }


def flat_to_decoder_index(plan, pos):
    """Map an original flat-token index -> its decoder index in the spliced
    sequence. Token/wl/d segs carry meta["flat"]; a board's stm token
    (flat_start+67, the only board-internal supervised position) maps to that
    board's last latent (pos["last"][seg])."""
    f2d = {}
    for i, s in enumerate(plan):
        last = pos["last"][i]
        if s.kind == "board":
            a = s.meta["flat_start"]
            for off in range(68):
                f2d[a + off] = last            # stm (a+67) -> last latent
        else:
            f2d[s.meta["flat"]] = last
    return f2d
