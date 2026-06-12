"""GRPO trainer (C in the architecture): consumes scored groups, updates the
policy, publishes weights, sole wandb writer.

  uv run python -m chessdecoder.agent.rl.grpo chessdecoder/agent/rl/config_grpo.yaml

Loss: token-level clipped surrogate on AGENT tokens only, advantage =
group-normalized episode reward, masks re-derived by episodes.replay (never
trusted from the rollout process). Entropy bonus on the masked distribution.
"""
from __future__ import annotations

import gc
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
from chessdecoder.agent.rl.buffer import (GroupBuffer,
                                           load_weights_if_newer,
                                           publish_weights)
from chessdecoder.agent.rl.episodes import ANSWER_MV, VERB, replay, verb_mask
from chessdecoder.agent.rl.reward import move_id_to_uci
from chessdecoder.agent.rl.rollout_proc import load_model
from chessdecoder.utils.muon import build_optimizer

DEV = "cuda"


def _aux_stream(cfg):
    """Infinite Stage-A stream for the auxiliary world-model-maintenance
    loss (catastrophic-forgetting counter; measured: k=1 apply 97.3% ->
    82.8% after 16.8k pure-RL steps)."""
    from chessdecoder.agent.tasks.stream import AgentTaskDataset
    ds = AgentTaskDataset(parquet_dir=cfg["aux_parquet_dir"],
                          label_glob=cfg["aux_label_glob"],
                          paired_glob=cfg.get("aux_paired_glob"),
                          task_mix=cfg["aux_mix"],
                          seed=cfg.get("seed", 0)
                               + cfg.get("_aux_seed_salt", 0))
    return iter(ds)


def _legal_mask(root: chess.Board) -> torch.Tensor:
    m = torch.zeros(pv.VOCAB_SIZE, dtype=torch.bool)
    for mv in root.legal_moves:
        for k in pv.move_keys(root, mv):
            if k in pv.MOVE_TO_ID:
                m[pv.MOVE_TO_ID[k]] = True
    return m


# static mask table rows: 0..18 board slots, then verb variants
# 19 = probe+answer, 20 = answer only, 21 = probe only (min quota unmet)
_STATIC = torch.stack([board_slot_mask(s) for s in range(pv.BOARD_LEN)]
                      + [verb_mask(1, True), verb_mask(0, True),
                         verb_mask(1, False)])


def _batch_positions(groups: list[dict]):
    """Flatten all agent positions of all episodes into one indexable batch.

    Returns (ids [N,Lmax], pos_b, pos_t, mask_rows [P,V], adv [P],
    behavior_lp [P], n_eps, stats). Padding uses PAD (never an agent pos).
    """
    eps_ids, advs, rows = [], [], []
    pos_b, pos_t, beh = [], [], []
    masks_extra = []
    ep_npos = []                     # positions per episode (length-bias fix)
    zero_var = 0
    for g in groups:
        r = g["rewards"]
        std = r.std()
        if std < 1e-6:
            # zero-variance: no PG signal (adv=0) but episodes stay in the
            # batch so the entropy bonus still applies exactly where the
            # policy has collapsed (audit risk #7)
            zero_var += 1
            a = [0.0] * int(r.shape[0])
        else:
            a = ((r - r.mean()) / (std + 1e-4)).tolist()
        legal = _legal_mask(chess.Board(g["root_fen"]))
        for e in range(g["ids"].shape[0]):
            ids = g["ids"][e].tolist()
            b = len(eps_ids)
            eps_ids.append(g["ids"][e])
            n0 = len(pos_t)
            for p, kind, budget, ans_ok in replay(
                    ids, g["root_fen"], g["k_budget"],
                    g.get("min_probes", 0)):
                pos_b.append(b)
                pos_t.append(p)
                beh.append(float(g["logprobs"][e][p]))
                advs.append(a[e])
                if kind == ANSWER_MV:
                    rows.append(len(masks_extra) + _STATIC.shape[0])
                elif kind == VERB:
                    rows.append((19 if ans_ok else 21) if budget > 0 else 20)
                else:
                    rows.append(kind)
            ep_npos.append(len(pos_t) - n0)
        masks_extra.append(legal)
    if not eps_ids:
        return None
    Lmax = max(x.shape[0] for x in eps_ids)
    ids = torch.full((len(eps_ids), Lmax), pv.PAD, dtype=torch.int64)
    for i, x in enumerate(eps_ids):
        ids[i, :x.shape[0]] = x
    table = torch.cat([_STATIC, torch.stack(masks_extra)]) \
        if masks_extra else _STATIC
    rows_t = torch.tensor(rows)
    # token family per position: 0=board slot, 1=verb, 2=answer move
    fam = torch.where(rows_t < 19, 0,
                      torch.where(rows_t < _STATIC.shape[0], 1, 2))
    # per-position weight 1/len(episode): every episode contributes equally
    # to the loss regardless of probe count (audit: token-mean gave a K=16
    # episode ~100x the gradient weight of a zero-probe sibling, drowning
    # the answer token ~340:1)
    w = torch.cat([torch.full((n,), 1.0 / max(n, 1)) for n in ep_npos])
    return (ids, torch.tensor(pos_b), torch.tensor(pos_t),
            table[rows_t], torch.tensor(advs, dtype=torch.float32),
            torch.tensor(beh, dtype=torch.float32), len(eps_ids), zero_var,
            fam, w)


def main(cfg_path: str):
    full_cfg = yaml.safe_load(open(cfg_path))
    cfg = full_cfg["trainer"]
    torch.manual_seed(cfg.get("seed", 0))
    model = load_model(cfg["base_ckpt"]).to(DEV).train()
    opt = build_optimizer(model, cfg.get("optimizer", "muon"),
                          cfg["learning_rate"], cfg["weight_decay"])
    base_lrs = [pg_["lr"] for pg_ in opt.param_groups]
    warmup = cfg.get("warmup_steps", 100)
    aux_coef = cfg.get("aux_coef", 0.0)
    cfg["_aux_seed_salt"] = (os.getpid() * 17 + int(time.time())) % 1_000_000
    aux_it = _aux_stream(cfg) if aux_coef > 0 else None
    buf = GroupBuffer(cfg["buffer_dir"])
    # groups in the buffer were generated by a previous policy: behavior
    # logprobs are stale relative to this fresh trainer -> wipe at startup
    stale = buf._files()
    for f in stale:
        os.unlink(f)
    if stale:
        print(f"wiped {len(stale)} stale buffer groups", flush=True)
    wandb.init(project=full_cfg["project_name"], name=full_cfg["run_name"],
               config=full_cfg)
    # resume the version counter: a reset counter desyncs a surviving
    # rollout forever and defeats the staleness check (audit bug #1)
    prev = load_weights_if_newer(cfg["weights_dir"], -1)
    version = (prev[1] if prev is not None else 0) + 1
    publish_weights(cfg["weights_dir"], model.state_dict(), version)
    print("published v1, waiting for groups...", flush=True)

    step = 0
    while step < cfg["max_steps"]:
        buf.wait_depth(cfg["min_buffer_groups"], timeout_s=3600)
        groups, dropped = buf.consume(cfg["groups_per_step"], version,
                                      cfg["max_staleness"])
        try:
            batch = _batch_positions(groups)
        except AssertionError as e:
            print(f"skipping corrupt group batch: {e}", flush=True)
            continue
        if batch is None:
            time.sleep(1.0)
            continue
        ids, pos_b, pos_t, masks, adv, beh, n_eps, zero_var, fam, w = batch
        ids = ids.to(DEV)
        pos_b, pos_t = pos_b.to(DEV), pos_t.to(DEV)
        masks, adv, beh = masks.to(DEV), adv.to(DEV), beh.to(DEV)
        w = w.to(DEV)
        temp = float(groups[0].get("temperature", 1.0))

        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(ids)                                # plain causal
            sel = h[pos_b, pos_t - 1]                     # predict position t
        logits = model.logits_at(sel.float())             # fp32 unembed, like
        lg = logits.masked_fill(~masks, float("-inf"))    # the sampling path
        lsm = torch.log_softmax(lg / temp, -1)            # match behavior T
        tok = ids[pos_b, pos_t]
        new_lp = lsm.gather(1, tok.unsqueeze(1)).squeeze(1)
        p = lsm.exp()                       # differentiable; 0 at masked
        entropy = -(p * lsm.masked_fill(masks.logical_not(), 0.0)).sum(-1)

        aux_loss = torch.zeros((), device=DEV)
        if aux_it is not None:
            # same convention as pretrain: hidden[p-1] predicts ids[p]
            a_ids, a_loss, _, _, a_pos = next(aux_it)
            a_ids = a_ids.to(DEV).unsqueeze(0)
            a_pos = a_pos.to(DEV).unsqueeze(0)
            a_loss = a_loss.to(DEV).unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ah = model(a_ids, a_pos)
            a_loss[:, 0] = False
            sel = a_loss.reshape(-1).nonzero(as_tuple=True)[0]
            hf = ah.reshape(-1, ah.shape[-1])[sel - 1]
            a_logits = model.logits_at(hf.float())
            tgt = a_ids.reshape(-1)[sel]
            aux_loss = torch.nn.functional.cross_entropy(a_logits, tgt)

        ratio = (new_lp - beh).exp()
        eps_clip = cfg["clip_eps"]
        s1 = ratio * adv
        s2 = ratio.clamp(1 - eps_clip, 1 + eps_clip) * adv
        # per-episode mean (w = 1/episode_len), then mean over episodes
        pg_loss = -(w * torch.min(s1, s2)).sum() / n_eps
        ent_term = (w * entropy).sum() / n_eps
        loss = (pg_loss - cfg["ent_coef"] * ent_term
                + aux_coef * aux_loss)

        for pg_, base in zip(opt.param_groups, base_lrs):
            pg_["lr"] = base * min(1.0, (step + 1) / max(warmup, 1))
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
        # forced-vs-free probing: reward and beat_greedy split by quota.
        # The thesis signal is quota>0 groups overtaking quota=0 groups.
        by_q = {}
        for g in groups:
            by_q.setdefault(g.get("min_probes", 0), []).append(g)
        stale = [version - g["version"] for g in groups]
        log = {
            "train/pg_loss": pg_loss.item(),
            "train/entropy": entropy.mean().item(),
            "train/entropy_board": entropy[fam == 0].mean().item()
                                   if (fam == 0).any() else 0.0,
            "train/entropy_verb": entropy[fam == 1].mean().item()
                                  if (fam == 1).any() else 0.0,
            "train/entropy_answer": entropy[fam == 2].mean().item()
                                    if (fam == 2).any() else 0.0,
            "train/clip_frac": ((ratio - 1).abs() > eps_clip).float()
                               .mean().item(),
            "train/ratio_dev": (ratio - 1).abs().mean().item(),
            "train/grad_norm": float(gnorm),
            "train/aux_loss": aux_loss.item(),
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
            "probes/repeat_per_ep": np.mean([m.get("probes_repeat", 0)
                                             for m in ms]),
            "probes/validity": (lambda v, i: v / max(v + i, 1))(
                sum(m["probes_valid"] for m in ms),
                sum(m["probes_invalid"] for m in ms)),
        }
        for mp, gs in by_q.items():
            r_q = torch.cat([g["rewards"] for g in gs])
            ms_q = [m for g in gs for m in g["metrics"]]
            log[f"quota/reward_mp{mp}"] = r_q.mean().item()
            log[f"quota/regret_mp{mp}"] = float(np.mean(
                [m["regret"] for m in ms_q]))
            log[f"quota/beat_greedy_mp{mp}"] = float(np.mean(
                [m["beat_greedy"] for m in ms_q]))
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
        if step % 500 == 0:
            gc.collect()                  # host-RAM creep mitigation
        if cfg["trainer_wait_s"] > 0:
            time.sleep(cfg["trainer_wait_s"])


if __name__ == "__main__":
    main(sys.argv[1])
