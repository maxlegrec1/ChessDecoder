"""Rollout generation using the C++ inference engine in subprocess workers.

Each worker owns one ThinkingInferenceEngine instance in a separate process
for GPU memory isolation. Workers stay alive across rollout batches to
avoid the ~2s engine initialization cost.
"""

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import torch

from src.rl.config import GRPOConfig


@dataclass
class RolloutResult:
    fen: str
    final_move: str
    token_ids: list[int]
    wl_entries: list[tuple[int, float]]
    d_entries: list[tuple[int, float]]
    num_tokens: int


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

_RELOAD = "__RELOAD__"  # Signals worker to reload engine


def _worker_loop(
    worker_id: int,
    export_dir: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    think_temperature: float,
    policy_temperature: float,
    board_temperature: float,
):
    """Worker process entry point. Owns one C++ engine instance."""
    import _decoder_inference_cpp as cpp

    def _create_engine(edir: str):
        return cpp.ThinkingInferenceEngine(
            str(Path(edir) / "backbone.pt"),
            str(Path(edir) / "weights"),
            str(Path(edir) / "vocab.json"),
            str(Path(edir) / "config.json"),
        )

    engine = _create_engine(export_dir)
    engine.think_temperature = think_temperature
    engine.policy_temperature = policy_temperature
    engine.board_temperature = board_temperature

    while True:
        item = task_queue.get()

        if item is None:
            break

        if isinstance(item, str) and item.startswith(_RELOAD):
            new_dir = item.split("|", 1)[1]
            del engine
            engine = _create_engine(new_dir)
            engine.think_temperature = think_temperature
            engine.policy_temperature = policy_temperature
            engine.board_temperature = board_temperature
            result_queue.put(("__RELOADED__", worker_id))
            continue

        task_id, fen = item
        try:
            move = engine.predict_move(fen, think_temperature)
            token_ids = list(engine.last_token_ids())
            wl_entries = list(engine.last_wl_entries())
            d_entries = list(engine.last_d_entries())
            result = RolloutResult(
                fen=fen,
                final_move=move or "",
                token_ids=token_ids,
                wl_entries=wl_entries,
                d_entries=d_entries,
                num_tokens=len(token_ids),
            )
            result_queue.put((task_id, result))
        except Exception as e:
            # Return empty result on failure
            result_queue.put((task_id, RolloutResult(
                fen=fen, final_move="", token_ids=[], wl_entries=[],
                d_entries=[], num_tokens=0,
            )))


# ---------------------------------------------------------------------------
# Rollout engine (orchestrator)
# ---------------------------------------------------------------------------

class RolloutEngine:
    """Multi-process rollout generation using C++ inference engines."""

    def __init__(self, export_dir: str, config: GRPOConfig):
        self.config = config
        self.export_dir = export_dir
        self.num_workers = config.num_workers

        self._task_queue: mp.Queue = mp.Queue()
        self._result_queue: mp.Queue = mp.Queue()
        self._workers: list[mp.Process] = []

        for i in range(self.num_workers):
            p = mp.Process(
                target=_worker_loop,
                args=(
                    i, export_dir, self._task_queue, self._result_queue,
                    config.think_temperature, config.policy_temperature,
                    config.board_temperature,
                ),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

    def generate(self, fens: list[str]) -> list[list[RolloutResult]]:
        """Generate G rollouts per FEN.

        Args:
            fens: list of B FEN strings.

        Returns:
            Nested list [B][G] of RolloutResults.
        """
        G = self.config.group_size
        B = len(fens)
        total = B * G

        # Submit all tasks: (task_id, fen)
        # task_id encodes (fen_idx, sample_idx)
        for fen_idx, fen in enumerate(fens):
            for sample_idx in range(G):
                task_id = fen_idx * G + sample_idx
                self._task_queue.put((task_id, fen))

        # Collect results
        results_flat: dict[int, RolloutResult] = {}
        collected = 0
        while collected < total:
            task_id, result = self._result_queue.get()
            results_flat[task_id] = result
            collected += 1

        # Reshape into [B][G]
        grouped: list[list[RolloutResult]] = []
        for fen_idx in range(B):
            group = []
            for sample_idx in range(G):
                task_id = fen_idx * G + sample_idx
                group.append(results_flat[task_id])
            grouped.append(group)

        return grouped

    def reload(self, export_dir: str):
        """Signal all workers to reload from a new export directory."""
        self.export_dir = export_dir
        for _ in self._workers:
            self._task_queue.put(f"{_RELOAD}|{export_dir}")
        # Wait for all workers to confirm reload
        reloaded = 0
        while reloaded < self.num_workers:
            item = self._result_queue.get()
            if isinstance(item, tuple) and item[0] == "__RELOADED__":
                reloaded += 1

    def shutdown(self):
        """Cleanly terminate all workers."""
        for _ in self._workers:
            self._task_queue.put(None)
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self._workers.clear()


# ---------------------------------------------------------------------------
# Model export utility
# ---------------------------------------------------------------------------

def export_model(model: torch.nn.Module, config: dict, export_dir: str | Path):
    """Export current model weights for C++ engine consumption.

    Follows the pattern from src/finetune/cpp_eval.py:167-187.

    Args:
        model: ChessDecoder (possibly wrapped in DDP).
        config: full config dict (must have "model" key).
        export_dir: directory to write backbone.pt, weights/, vocab.json, config.json.
    """
    from src.export.common import export_head_weights, export_vocab, export_config
    from src.export.backbone_causal import from_chess_decoder
    from src.export.export_torchscript import export_torchscript

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model.module if hasattr(model, "module") else model
    was_training = raw_model.training
    raw_model.eval()

    export_vocab(export_dir)
    export_config(config, raw_model, export_dir)
    export_head_weights(raw_model, export_dir / "weights")

    causal_backbone = from_chess_decoder(raw_model)
    export_torchscript(causal_backbone, export_dir, config)
    del causal_backbone
    torch.cuda.empty_cache()

    if was_training:
        raw_model.train()
