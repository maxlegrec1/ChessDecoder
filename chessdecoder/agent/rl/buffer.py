"""Cross-process plumbing: episode-group buffer + weight publishing.

Rollout process writes scored EpisodeGroups as files into a /dev/shm dir;
the trainer consumes oldest-first, drops groups staler than max_staleness
policy versions, deletes after use. Trainer publishes weights atomically;
rollout hot-reloads between waves. All writes are tmp-file + os.replace
(atomic on the same filesystem) so readers never see partial files.
"""
from __future__ import annotations

import glob
import os
import time

import torch

GROUP_PREFIX = "group"
WEIGHTS_NAME = "agent_weights.pt"


class GroupBuffer:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._n_written = 0

    # -- rollout side --------------------------------------------------------
    def write(self, group: dict, version: int) -> str:
        name = (f"{GROUP_PREFIX}_v{version:07d}_{os.getpid()}_"
                f"{self._n_written:08d}.pt")
        tmp = os.path.join(self.root, "." + name)
        torch.save(group, tmp)
        path = os.path.join(self.root, name)
        os.replace(tmp, path)
        self._n_written += 1
        return path

    # -- trainer side --------------------------------------------------------
    def depth(self) -> int:
        return len(self._files())

    def _files(self) -> list[str]:
        return sorted(f for f in glob.glob(
            os.path.join(self.root, f"{GROUP_PREFIX}_*.pt")))

    @staticmethod
    def _version(path: str) -> int:
        return int(os.path.basename(path).split("_")[1][1:])

    def consume(self, n: int, current_version: int,
                max_staleness: int) -> tuple[list[dict], int]:
        """Oldest-first; returns (groups, n_dropped_stale). Blocks the caller
        only as far as what exists — returns fewer than n if buffer shallow."""
        out, dropped = [], 0
        for path in self._files():
            try:
                if current_version - self._version(path) > max_staleness:
                    os.unlink(path)
                    dropped += 1
                    continue
                g = torch.load(path, weights_only=False)
                os.unlink(path)
            except FileNotFoundError:
                continue
            out.append(g)
            if len(out) == n:
                break
        return out, dropped

    def wait_depth(self, n: int, timeout_s: float = 600,
                   poll_s: float = 0.5) -> bool:
        t0 = time.monotonic()
        while self.depth() < n:
            if time.monotonic() - t0 > timeout_s:
                return False
            time.sleep(poll_s)
        return True


def publish_weights(dir_: str, state_dict: dict, version: int) -> None:
    os.makedirs(dir_, exist_ok=True)
    tmp = os.path.join(dir_, "." + WEIGHTS_NAME)
    torch.save({"version": version,
                "state_dict": {k: v.to(torch.bfloat16).cpu()
                               for k, v in state_dict.items()}}, tmp)
    os.replace(tmp, os.path.join(dir_, WEIGHTS_NAME))


def load_weights_if_newer(dir_: str, have_version: int):
    """-> (state_dict, version) or None."""
    path = os.path.join(dir_, WEIGHTS_NAME)
    if not os.path.exists(path):
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None                      # mid-replace race: retry next poll
    if ck["version"] <= have_version:
        return None
    return ck["state_dict"], ck["version"]
