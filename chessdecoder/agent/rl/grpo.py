"""GRPO trainer (C in the architecture): consumes scored groups, updates the
policy, publishes weights, sole wandb writer.

  uv run python -m chessdecoder.agent.rl.grpo chessdecoder/agent/rl/config_grpo.yaml

Loss: token-level clipped surrogate on AGENT tokens only, advantage =
group-normalized episode reward, masks re-derived by episodes.replay (never
trusted from the rollout process). Entropy bonus on the masked distribution.
"""
from __future__ import annotations

import os
import sys
import time

import chess
import numpy as np
import torch
import wandb
import yaml

from chessdecoder.agent import patch_vocab as pv
from chessdecoder.agent.grammar import board_slot_mask
from chessdecoder.agent.rl.buffer import GroupBuffer, publish_weights
from chessdecoder.agent.rl.episodes import ANSWER_MV, VERB, replay, verb_mask
from chessdecoder.agent.rl.reward import move_id_to_uci
from chessdecoder.agent.rl.rollout_proc import load_model
from chessdecoder.utils.muon import build_optimizer

DEV = "cuda"


def _legal_mask(root: chess.Board) -> torch.Tensor:
    m = torch.zeros(pv.VOCAB_SIZE, dtype=torch.bool)
    for mv in root.legal_moves:
        for k in pv.move_keys(root, mv):
            if k in pv.MOVE_TO_ID:
                m[pv.MOVE_TO_ID[k]] = True
    return m


# static mask table rows: 0..18 board slots, 19 = verb(budget>0), 20 = verb(0)
_STATIC = torch.stack([board_slot_mask(s) for s in range(pv.BOARD_LEN)]
                      + [verb_mask(1), verb_mask(0)])


def _batch_positions(groups: list[dict]):
    """Flatten all agent positions of all episodes into one indexable batch.

    Returns (ids [N,Lmax], pos_b, pos_t, mask_rows [P,V], adv [P],
    behavior_lp [P], n_eps, stats). Padding uses PAD (never an agent pos).
    """
    eps_ids, advs, rows = [], [], []
    pos_b, pos_t, beh = [], [], []
    masks_extra = []
    zero_var = 0
    for g in groups:
        r = g["rewards"]
        std = r.std()
        if std < 1e-6:
            zero_var += 1
            continue
        a = ((r - r.mean()) / (std + 1e-4)).tolist()
        legal = _legal_mask(chess.Board(g["root_fen"]))
        for e in range(g["ids"].shape[0]):
            ids = g["ids"][e].tolist()
            b = len(eps_ids)
            eps_ids.append(g["ids"][e])
            for p, kind, budget in replay(ids, g["root_fen"], g["k_budget"]):
                pos_b.append(b)
                pos_t.append(p)
                beh.append(float(g["logprobs"][e][p]))
                advs.append(a[e])
                if kind == ANSWER_MV:
                    rows.append(len(masks_extra) + _STATIC.shape[0])
                elif kind == VERB:
                    rows.append(19 if budget > 0 else 20)
                else:
                    rows.append(kind)
        masks_extra.append(legal)
    if not eps_ids:
        return None
    Lmax = max(x.shape[0] for x in eps_ids)
    ids = torch.full((len(eps_ids), Lmax), pv.PAD, dtype=torch.int64)
    for i, x in enumerate(eps_ids):
        ids[i, :x.shape[0]] = x
    table = torch.cat([_STATIC, torch.stack(masks_extra)]) \
        if masks_extra else _STATIC
    return (ids, torch.tensor(pos_b), torch.tensor(pos_t),
            table[torch.tensor(rows)], torch.tensor(advs, dtype=torch.float32),
            torch.tensor(beh, dtype=torch.float32), len(eps_ids), zero_var)


def main(cfg_path: str):
    full_cfg = yaml.safe_load(open(cfg_path))
    cfg = full_cfg["trainer"]
    torch.manual_seed(cfg.get("seed", 0))
    model = load_model(cfg["base_ckpt"]).to(DEV).train()
    opt = build_optimizer(model, cfg.get("optimizer", "muon"),
                          cfg["learning_rate"], cfg["weight_decay"])
    buf = GroupBuffer(cfg["buffer_dir"])
    wandb.init(project=full_cfg["project_name"], name=full_cfg["run_name"],
               config=full_cfg)
    version = 1
    publish_weights(cfg["weights_dir"], model.state_dict(), version)
    print("published v1, waiting for groups...", flush=True)

    step = 0
    while step < cfg["max_steps"]:
        buf.wait_depth(cfg["min_buffer_groups"], timeout_s=3600)
        groups, dropped = buf.consume(cfg["groups_per_step"], version,
                                      cfg["max_staleness"])
        batch = _batch_positions(groups)
        if batch is None:
            time.sleep(1.0)
            continue
        ids, pos_b, pos_t, masks, adv, beh, n_eps, zero_var = batch
        ids = ids.to(DEV)
        pos_b, pos_t = pos_b.to(DEV), pos_t.to(DEV)
        masks, adv, beh = masks.to(DEV), adv.to(DEV), beh.to(DEV)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(ids)                                # plain causal
            sel = h[pos_b, pos_t - 1]                     # predict position t
            logits = model.logits_at(sel.float())
        lg = logits.float().masked_fill(~masks, float("-inf"))
        lsm = torch.log_softmax(lg, -1)
        tok = ids[pos_b, pos_t]
        new_lp = lsm.gather(1, tok.unsqueeze(1)).squeeze(1)
        p = lsm.exp()                       # differentiable; 0 at masked
        entropy = -(p * lsm.masked_fill(masks.logical_not(), 0.0)).sum(-1)

        ratio = (new_lp - beh).exp()
        eps_clip = cfg["clip_eps"]
        s1 = ratio * adv
        s2 = ratio.clamp(1 - eps_clip, 1 + eps_clip) * adv
        pg_loss = -torch.min(s1, s2).mean()
        loss = pg_loss - cfg["ent_coef"] * entropy.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg["grad_clip"])
        opt.step()
        step += 1
        if step % cfg["sync_every"] == 0:
            version += 1
            publish_weights(cfg["weights_dir"], model.state_dict(), version)

        # -- metrics (trainer = sole wandb writer) ---------------------------
        ms = [m for g in groups for m in g["metrics"]]
        rew = torch.cat([g["rewards"] for g in groups])
        stale = [version - g["version"] for g in groups]
        log = {
            "train/pg_loss": pg_loss.item(),
            "train/entropy": entropy.mean().item(),
            "train/clip_frac": ((ratio - 1).abs() > eps_clip).float()
                               .mean().item(),
            "train/ratio_dev": (ratio - 1).abs().mean().item(),
            "train/grad_norm": float(gnorm),
            "train/zero_var_groups": zero_var,
            "train/staleness_mean": float(np.mean(stale)),
            "train/buffer_depth": buf.depth(),
            "train/dropped_stale": dropped,
            "train/episodes": n_eps,
            "reward/mean": rew.mean().item(),
            "reward/p10": rew.quantile(0.1).item(),
            "reward/p90": rew.quantile(0.9).item(),
            "reward/beat_greedy": np.mean([m["beat_greedy"] for m in ms]),
            "reward/match_greedy": np.mean([m["match_greedy"] for m in ms]),
            "reward/match_search_best": np.mean([m["match_search_best"]
                                                 for m in ms]),
            "reward/match_corpus_best": np.mean([m["match_corpus_best"]
                                                 for m in ms]),
            "probes/valid_per_ep": np.mean([m["probes_valid"] for m in ms]),
            "probes/invalid_per_ep": np.mean([m["probes_invalid"]
                                              for m in ms]),
            "probes/validity": (lambda v, i: v / max(v + i, 1))(
                sum(m["probes_valid"] for m in ms),
                sum(m["probes_invalid"] for m in ms)),
        }
        wandb.log(log, step=step)
        if step % 20 == 0:
            print(f"step {step}: R {log['reward/mean']:+.4f} "
                  f"beat_greedy {log['reward/beat_greedy']:.3f} "
                  f"validity {log['probes/validity']:.2f} "
                  f"clip {log['train/clip_frac']:.3f}", flush=True)
        if step % cfg["ckpt_every"] == 0:
            os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(),
                        "step": step},
                       f"{cfg['checkpoint_dir']}/grpo_{step}.pt")
        if cfg["trainer_wait_s"] > 0:
            time.sleep(cfg["trainer_wait_s"])


if __name__ == "__main__":
    main(sys.argv[1])
