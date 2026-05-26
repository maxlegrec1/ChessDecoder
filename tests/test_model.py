"""ChessEncoder forward + heads (CPU only, tiny config)."""
import torch

from chessdecoder.dataloader.data import fen_to_position_tokens
from chessdecoder.models.model import ChessEncoder
from chessdecoder.models.value_buckets import N_CELLS
from chessdecoder.models.vocab import move_vocab_size, token_to_idx, vocab_size

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _tiny(seq_len=68):
    return ChessEncoder(vocab_size=vocab_size, embed_dim=32, num_heads=4,
                        num_layers=2, seq_len=seq_len, d_ff=64)


def _bid(fen, batch=1):
    ids = torch.tensor([token_to_idx[t] for t in fen_to_position_tokens(fen)],
                       dtype=torch.long)
    return ids.unsqueeze(0).expand(batch, -1)


def test_forward_shapes():
    m = _tiny()
    out = m(_bid(START, batch=3))
    assert out["policy"].shape == (3, move_vocab_size)
    assert out["wdl"].shape == (3, N_CELLS)


def test_mean_wdl_is_a_distribution():
    m = _tiny()
    out = m(_bid(START, batch=2))
    p = m.mean_wdl(out["wdl"])                      # [2,3]
    assert p.shape == (2, 3)
    assert torch.allclose(p.sum(-1), torch.ones(2), atol=1e-5)
    assert (p >= 0).all() and (p <= 1).all()


def test_policy_logits_finite():
    m = _tiny()
    out = m(_bid(START))
    assert torch.isfinite(out["policy"]).all()
    assert torch.isfinite(out["wdl"]).all()


def test_grad_flows_through_both_heads():
    m = _tiny()
    out = m(_bid(START, batch=2))
    loss = out["policy"].sum() + out["wdl"].sum()
    loss.backward()
    # gradient should reach the token embedding (every position uses tok_emb)
    assert m.tok_embedding.weight.grad is not None
    assert m.tok_embedding.weight.grad.abs().sum() > 0
