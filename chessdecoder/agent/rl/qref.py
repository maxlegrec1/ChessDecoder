"""Q_ref producer: batched PUCT search over the frozen oracle.

The reward judge for GRPO: for each root, an N-sim PUCT search over
OracleEngine yields Q(root, a) for every root move. Search-over-oracle is
strictly stronger than oracle-greedy — that gap is the headroom the agent is
trained to capture (claim L1), and the same machinery is the L2 baseline.

Many independent trees advance in lockstep: each wave selects one leaf per
tree, evaluates all leaves in one oracle batch, expands and backs up.

Sign convention (tested on mate-in-1/2 in tests/test_rl_qref.py):
  - oracle q is from the side-to-move's POV of the evaluated position
  - edge value Q(s, a) is stored from the POV of the player to move at s,
    so child evaluation v(s') backs up as -v through each ply
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from chessdecoder.agent.rl.oracle_engine import OracleEngine, _legal_indices

C_PUCT = 1.5
FPU_REDUCTION = 0.2          # unvisited child Q = node value - this


@dataclass
class _Node:
    fen: str
    moves: list[str] = field(default_factory=list)   # uci, legal order
    child: list[int] = field(default_factory=list)   # node ids (-1 = unexpanded)
    P: np.ndarray | None = None
    N: np.ndarray | None = None
    W: np.ndarray | None = None
    v: float = 0.0               # oracle value at this node (stm POV)
    terminal_v: float | None = None


class Tree:
    def __init__(self, root: chess.Board):
        self.nodes: list[_Node] = [_Node(fen=root.fen())]
        self.root_board = root

    def select(self) -> tuple[list[tuple[int, int]], chess.Board, _Node]:
        """Walk PUCT from the root; returns (path of (node_id, child_idx),
        board at the leaf, leaf node placeholder info). The leaf is either
        unexpanded (needs eval) or terminal."""
        nid, path = 0, []
        board = self.root_board.copy(stack=False)
        board.clear_stack()          # repetition window = the search path
        while True:
            node = self.nodes[nid]
            if node.terminal_v is not None or node.P is None:
                return path, board, node
            q = np.where(node.N > 0, node.W / np.maximum(node.N, 1),
                         node.v - FPU_REDUCTION)
            u = C_PUCT * node.P * (np.sqrt(max(1, node.N.sum()))
                                   / (1 + node.N))
            ci = int(np.argmax(q + u))
            path.append((nid, ci))
            board.push(chess.Move.from_uci(node.moves[ci]))
            if node.child[ci] == -1:
                node.child[ci] = len(self.nodes)
                self.nodes.append(_Node(fen=board.fen()))
            nid = node.child[ci]

    def backup(self, path: list[tuple[int, int]], v_leaf: float) -> None:
        """v_leaf: value of the leaf from the LEAF side-to-move's POV."""
        v = v_leaf
        for nid, ci in reversed(path):
            v = -v                              # one ply up flips POV
            n = self.nodes[nid]
            n.N[ci] += 1
            n.W[ci] += v


def _expand(node: _Node, board: chess.Board, q: float,
            pol: torch.Tensor) -> None:
    """Fill priors from legal-masked softmax of oracle policy logits."""
    idxs, _, gids = _legal_indices(board)
    moves = list(board.legal_moves)
    lg = pol[idxs].float()
    # collapse castling double-spellings: keep the better logit per move
    best = {}
    for j, g in enumerate(gids):
        if g not in best or lg[j] > lg[best[g]]:
            best[g] = j
    # moves with no vocab key (rare) are unplayable for the agent too:
    # exclude them from the search rather than crash
    keep = [g for g in range(len(moves)) if g in best]
    sel = [best[g] for g in keep]
    moves = [moves[g] for g in keep]
    pri = torch.softmax(lg[sel], 0).numpy()
    node.moves = [m.uci() for m in moves]
    node.child = [-1] * len(moves)
    node.P = pri
    node.N = np.zeros(len(moves), dtype=np.int64)
    node.W = np.zeros(len(moves))
    node.v = q


def _terminal_value(board: chess.Board) -> float | None:
    if board.is_checkmate():
        return -1.0                  # side to move is mated
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.halfmove_clock >= 100        # match chess.hpp exactly
            or board.is_repetition(3)):
        return 0.0
    return None


@dataclass
class QRefResult:
    fen: str
    moves: list[str]
    q: list[float]
    visits: list[int]
    oracle_greedy: str
    search_best: str


def search_batch(engine: OracleEngine, roots: list[chess.Board],
                 sims: int = 800) -> list[QRefResult]:
    trees = [Tree(b) for b in roots]
    # root expansion (one batch)
    q0, _, pol0 = engine.eval_batch([b.fen() for b in roots])
    pol0 = pol0.cpu()
    for t, b, qq, pp in zip(trees, roots, q0.cpu().tolist(), pol0):
        _expand(t.nodes[0], b, qq, pp)
    for _ in range(sims):
        pend = []                     # (tree, path, node) awaiting oracle
        for t in trees:
            path, board, node = t.select()
            if node.terminal_v is None and node.P is None:
                tv = _terminal_value(board)
                if tv is not None:
                    node.terminal_v = tv
            if node.terminal_v is not None:
                t.backup(path, node.terminal_v)
            else:
                pend.append((t, path, board, node))
        if pend:
            q, _, pol = engine.eval_batch([b.fen() for _, _, b, _ in pend])
            pol = pol.cpu()
            ql = q.cpu().tolist()
            for (t, path, board, node), qq, pp in zip(pend, ql, pol):
                _expand(node, board, qq, pp)
                t.backup(path, qq)
    out = []
    for t in trees:
        r = t.nodes[0]
        # unvisited root moves: "worse than anything explored", not a -1.0
        # cliff (the rewarder reads these q's; a cliff distorts regret scale)
        vis = r.N > 0
        floor = float((r.W[vis] / r.N[vis]).min()) - 0.1 if vis.any() else -1.0
        q = np.where(vis, r.W / np.maximum(r.N, 1), max(-1.0, floor))
        bi = int(np.argmax(r.N + q))           # visits, Q breaks ties
        out.append(QRefResult(
            fen=t.root_board.fen(), moves=r.moves,
            q=[float(x) for x in q], visits=[int(x) for x in r.N],
            oracle_greedy=r.moves[int(np.argmax(r.P))],
            search_best=r.moves[bi]))
    return out


# -- C++ engine (chessdecoder/cpp/puct) --------------------------------------
_cpp_ready = False


def _init_cpp() -> None:
    global _cpp_ready
    if _cpp_ready:
        return
    import _puct_cpp
    from chessdecoder.models.vocab import move_token_to_idx
    from chessdecoder.dataloader import loader as L
    consts = {"start": L._START_IDX, "end": L._END_IDX,
              "empty": L._EMPTY_IDX, "wtm": L._WTM_IDX, "btm": L._BTM_IDX,
              "nocastle": L._NO_CASTLE_IDX}
    for ch, tok in L._PIECE_IDX.items():
        consts[ch] = int(tok)
    _puct_cpp.init_maps(dict(move_token_to_idx),
                        {k: int(v) for k, v in consts.items()},
                        {k: int(v) for k, v in L._CASTLE_IDX.items()})
    _cpp_ready = True


def search_batch_cpp(engine: OracleEngine, roots: list[chess.Board],
                     sims: int = 800) -> list[QRefResult]:
    """C++ trees, python GPU eval. Same semantics as search_batch (the
    first wave is the root expansion, then `sims` PUCT waves)."""
    import _puct_cpp
    _init_cpp()
    t = _puct_cpp.BatchedPuct([b.fen() for b in roots])
    for _ in range(sims + 1):
        idx, ids = t.select()
        if idx:
            q, _, pol = engine.eval_ids(ids)
            t.expand_backup(idx,
                            q.float().cpu().numpy(),
                            pol.float().cpu().numpy())
    out = []
    for b, (moves, q, vis, greedy, best) in zip(roots, t.results()):
        moves = list(moves)
        out.append(QRefResult(
            fen=b.fen(), moves=moves, q=[float(x) for x in q],
            visits=[int(x) for x in vis], oracle_greedy=moves[greedy],
            search_best=moves[best]))
    return out
