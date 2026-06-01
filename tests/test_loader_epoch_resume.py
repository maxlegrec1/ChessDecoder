"""Regression test for the resume epoch-seed bug.

With persistent_workers=True a worker freezes its own copy of ``self.epoch`` at
spawn time. The per-epoch RNG seed must therefore be driven by a per-worker
counter seeded from the spawn-time epoch. The old ``max(self.epoch, counter)``
formula REPLAYED the resume epoch's data for ``start_epoch`` extra epochs (the
frozen self.epoch dominated the reset counter), re-memorizing already-seen plies
and producing a spurious loss jump when fresh data finally arrived.

These tests assert every epoch — fresh or resumed — gets a strictly increasing,
unique epoch index (hence a unique seed).
"""
from chessdecoder.dataloader.loader import ChessIterableDataset


def _seq(spawn_epoch, n):
    """Epoch indices a single persistent worker emits over n epochs, given its
    self.epoch was frozen at spawn_epoch (no propagation after spawn)."""
    ds = ChessIterableDataset.__new__(ChessIterableDataset)  # no file discovery
    ds.epoch = spawn_epoch                                   # frozen at spawn
    return [ds._advance_epoch() for _ in range(n)]


def test_fresh_run_unique_epochs():
    # fresh run: worker spawns at epoch 0
    assert _seq(0, 5) == [0, 1, 2, 3, 4]


def test_resume_at_epoch_1_no_replay():
    # ckpt@20k resumed at epoch index 1 — must NOT replay epoch 1
    assert _seq(1, 5) == [1, 2, 3, 4, 5]


def test_resume_at_deep_epoch_no_replay():
    # deeper resume must also advance cleanly
    assert _seq(7, 4) == [7, 8, 9, 10]


def test_indices_strictly_increasing_so_seeds_differ():
    for start in (0, 1, 3, 10):
        s = _seq(start, 6)
        assert s == sorted(set(s)), f"replay/duplicate epoch index for start={start}: {s}"
        assert s[0] == start
