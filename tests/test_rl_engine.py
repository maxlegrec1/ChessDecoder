"""Rollout engine correctness. The logprob-consistency test is the gate for
GRPO: a silent KV/position/mask bug here poisons every importance ratio."""
import glob

import chess
import pytest
import torch

from chessdecoder.agent import patch_vocab as pv

pytestmark = [pytest.mark.gpu]

CKPT_GLOB = ("checkpoints/agent_pretrain_v2/pretrain-v2_20260611_144726/"
             "agent_35000.pt")
B = 16          # small batch: tests must be fast
K = 4


def _load_model():
    from chessdecoder.agent.model import AgentDecoder
    ck = torch.load(glob.glob(CKPT_GLOB)[0], map_location="cpu",
                    weights_only=False)
    sd = ck["model_state_dict"]
    vocab = sd["tok_embedding.weight"].shape[0]
    m = AgentDecoder(vocab_size=vocab)
    m.load_state_dict(sd)
    return m


@pytest.fixture(scope="module")
def setup():
    from chessdecoder.agent.oracle import Oracle
    from chessdecoder.agent.rl.engine import RolloutEngine
    model = _load_model()
    oracle = Oracle()
    # fp32: the precision-exact configuration (catches position/mask bugs
    # that bf16 noise would blur)
    eng = RolloutEngine(model, oracle, batch_size=B, k_budget=K,
                        temperature=1.0, dtype=torch.float32)
    return model, oracle, eng


@pytest.fixture(scope="module")
def roots():
    import pandas as pd
    f = sorted(glob.glob("/mnt/2tb_2/decoder/parquet_files_decoder/*.parquet"))[-1]
    fens = pd.read_parquet(f, columns=["fen"]).fen.drop_duplicates()
    return [chess.Board(x) for x in fens.head(B).tolist()]


@pytest.fixture(scope="module")
def episodes(setup, roots):
    _, _, eng = setup
    torch.manual_seed(0)
    return eng.rollout(roots)


def test_episodes_finish(episodes):
    for e in episodes:
        assert e.done, "episode did not reach <answer>"
        assert e.final_move is not None
        assert e.probes_valid + e.probes_invalid <= e.k_budget


def test_grammar_replay(episodes):
    """Every agent position parses back; sampled tokens satisfy the replayed
    masks; non-agent positions are exactly the injected segments."""
    from chessdecoder.agent.rl.episodes import (ANSWER_MV, VERB, mask_for,
                                                replay, verb_mask)
    for e in episodes:
        agent_pos = {p for p, _, _, _ in replay(e.ids, e.root_fen, e.k_budget)}
        flagged = {i for i, a in enumerate(e.agent) if a}
        assert agent_pos == flagged, "replay disagrees with engine flags"
        root = chess.Board(e.root_fen)
        for p, kind, budget, ans_ok in replay(e.ids, e.root_fen, e.k_budget):
            if kind == VERB:
                m = verb_mask(budget, ans_ok)
            elif kind == ANSWER_MV:
                m = mask_for(kind, root)
            else:
                m = mask_for(kind, None)
            assert m[e.ids[p]], f"token {e.ids[p]} at {p} violates mask {kind}"


def test_logprob_consistency(setup, episodes):
    """Teacher-forced no-cache forward reproduces the stored behavior
    logprobs (fp32: atol 1e-3). THE gate test."""
    from chessdecoder.agent.rl.episodes import (ANSWER_MV, VERB, mask_for,
                                                replay, verb_mask)
    model, _, eng = setup
    worst = 0.0
    for e in episodes:
        ids = torch.tensor(e.ids, device="cuda").unsqueeze(0)
        with torch.no_grad(), model.caches_disabled():
            h = model(ids)                      # no cache, plain causal
        root = chess.Board(e.root_fen)
        for p, kind, budget, ans_ok in replay(e.ids, e.root_fen, e.k_budget):
            logits = model.logits_at(h[0, p - 1].float().unsqueeze(0))[0]
            if kind == VERB:
                m = verb_mask(budget, ans_ok)
            elif kind == ANSWER_MV:
                m = mask_for(kind, root)
            else:
                m = mask_for(kind, None)
            lg = logits.masked_fill(~m.cuda(), float("-inf"))
            lp = torch.log_softmax(lg / eng.T, -1)[e.ids[p]].item()
            worst = max(worst, abs(lp - e.logprobs[p]))
    assert worst < 1e-3, f"worst logprob mismatch {worst}"


def test_oracle_injection_parity(setup, episodes):
    """Injected replies match an independent eager-Oracle query of the
    decoded probe board."""
    from chessdecoder.agent.rl.episodes import PREFIX_LEN, PROBE_TOKENS
    _, oracle, _ = setup
    checked = 0
    for e in episodes:
        i = PREFIX_LEN
        while i < len(e.ids):
            t = e.ids[i]
            if t == pv.PROBE:
                slots = e.ids[i + 1:i + 1 + pv.BOARD_LEN]
                i += PROBE_TOKENS
                reply = e.ids[i:i + 8]
                i += 8
                try:
                    b = pv.decode_board(slots)
                except Exception:
                    b = None
                if b is None or not b.is_valid() or b.is_game_over():
                    assert reply[0] == pv.INVALID
                else:
                    r = oracle.query(b)
                    assert reply[:7] == r.tokens(), "reply != oracle"
                checked += 1
            elif t == pv.ANSWER:
                i += 2
            elif t == pv.PAD:
                i += 1
            else:
                raise AssertionError(f"unexpected token {t} at {i}")
    assert checked > 0


def test_probe_stats_consistency(episodes):
    for e in episodes:
        n_probe = sum(1 for t in e.ids if t == pv.PROBE)
        assert n_probe == e.probes_valid + e.probes_invalid


@pytest.mark.slow
def test_throughput(setup, roots):
    """Aggregate tok/s floor at the test batch size. Conservative: the
    pretrain run may share the GPU."""
    import time
    _, _, eng = setup
    torch.manual_seed(1)
    t0 = time.perf_counter()
    eps = eng.rollout(roots)
    dt = time.perf_counter() - t0
    toks = sum(len(e.ids) for e in eps)
    rate = toks / dt
    print(f"\nrollout: {toks} tokens in {dt:.1f}s = {rate:,.0f} tok/s (B={eng.B})")
    # B=16 fp32 test engine, possibly sharing the GPU with live RL units;
    # the production configuration (B=128 bf16) measures 6.7k tok/s
    assert rate > 400, f"rollout too slow: {rate:.0f} tok/s"



def test_min_probes_forced(setup, roots):
    """min_probes masks <answer> until the quota is met."""
    _, _, eng = setup
    torch.manual_seed(3)
    eps = eng.rollout(roots, min_probes=[2] * B)
    for e in eps:
        assert e.done
        assert e.probes_valid + e.probes_invalid >= 2, "quota violated"
