"""Metric tracking and WandB logging for GRPO training."""

from dataclasses import dataclass, field

import wandb

from src.finetune.cpp_eval import _normalize_castling


@dataclass
class RolloutMetrics:
    """Accumulated metrics from a single rollout batch."""
    reward_total: list[float] = field(default_factory=list)
    reward_move_quality: list[float] = field(default_factory=list)
    reward_format: list[float] = field(default_factory=list)
    reward_coherence: list[float] = field(default_factory=list)
    rollout_lengths: list[int] = field(default_factory=list)
    acc_at_1: list[bool] = field(default_factory=list)
    acc_at_k: list[bool] = field(default_factory=list)
    coherence_hits: list[bool] = field(default_factory=list)
    rollout_time: float = 0.0


@dataclass
class TrainingMetrics:
    """Accumulated metrics from PPO inner loop steps."""
    policy_loss: list[float] = field(default_factory=list)
    kl_loss: list[float] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    kl: list[float] = field(default_factory=list)
    clip_fraction: list[float] = field(default_factory=list)
    thinking_entropy: list[float] = field(default_factory=list)
    final_entropy: list[float] = field(default_factory=list)


class GRPOMetrics:
    """Tracks and logs all GRPO metrics to WandB."""

    def __init__(self):
        self._rollout = RolloutMetrics()
        self._training = TrainingMetrics()

    def reset(self):
        self._rollout = RolloutMetrics()
        self._training = TrainingMetrics()

    def log_rollout_batch(
        self,
        grouped_rollouts: list[list],
        grouped_rewards: list[list[tuple[float, dict[str, float]]]],
        ground_truths: list[dict],
        rollout_time: float,
    ):
        """Record metrics from a rollout batch.

        Args:
            grouped_rollouts: [B][G] RolloutResults.
            grouped_rewards: [B][G] (total_reward, component_dict).
            ground_truths: [B] dicts with best_move, mcts_action.
            rollout_time: seconds for the rollout generation.
        """
        self._rollout.rollout_time = rollout_time

        for fen_idx, (group, rewards_group, gt) in enumerate(
            zip(grouped_rollouts, grouped_rewards, ground_truths)
        ):
            best_move = _normalize_castling(gt["best_move"])

            # acc@k: any of G completions matches best_move
            any_correct = False
            for rollout in group:
                if _normalize_castling(rollout.final_move) == best_move:
                    any_correct = True
                    break
            self._rollout.acc_at_k.append(any_correct)

            # acc@1: first completion matches (proxy for greedy)
            first_correct = _normalize_castling(group[0].final_move) == best_move
            self._rollout.acc_at_1.append(first_correct)

            for sample_idx, (rollout, (total_r, components)) in enumerate(
                zip(group, rewards_group)
            ):
                self._rollout.reward_total.append(total_r)
                self._rollout.reward_move_quality.append(components.get("move_quality", 0.0))
                self._rollout.reward_format.append(components.get("format", 0.0))
                self._rollout.reward_coherence.append(components.get("coherence", 0.0))
                self._rollout.rollout_lengths.append(rollout.num_tokens)
                self._rollout.coherence_hits.append(components.get("coherence", 0.0) > 0.0)

    def log_training_step(self, loss_info: dict):
        """Record metrics from one PPO inner step."""
        self._training.policy_loss.append(loss_info["policy_loss"])
        self._training.kl_loss.append(loss_info["kl_loss"])
        self._training.total_loss.append(loss_info["policy_loss"] + loss_info["kl_loss"])
        self._training.kl.append(loss_info["kl"])
        self._training.clip_fraction.append(loss_info["clip_fraction"])

    def log_entropy(self, entropy_info: tuple[float, float]):
        """Record policy entropy metrics (computed once per outer step)."""
        self._training.thinking_entropy.append(entropy_info[0])
        self._training.final_entropy.append(entropy_info[1])

    def to_wandb(self, step: int, lr: float | None = None, eval_results: dict | None = None):
        """Flush accumulated metrics to WandB."""
        d: dict = {}

        def _mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        # Rollout metrics
        r = self._rollout
        if r.reward_total:
            d["rl/reward_mean"] = _mean(r.reward_total)
            d["rl/reward_std"] = (
                (sum((x - _mean(r.reward_total)) ** 2 for x in r.reward_total) / len(r.reward_total)) ** 0.5
                if len(r.reward_total) > 1 else 0.0
            )
            d["rl/reward_move_quality"] = _mean(r.reward_move_quality)
            d["rl/reward_format"] = _mean(r.reward_format)
            d["rl/reward_coherence"] = _mean(r.reward_coherence)
            d["rl/mean_seq_len"] = _mean(r.rollout_lengths)
            d["rl/min_seq_len"] = min(r.rollout_lengths)
            d["rl/max_seq_len"] = max(r.rollout_lengths)
            if len(r.rollout_lengths) > 1:
                mean = _mean(r.rollout_lengths)
                d["rl/std_seq_len"] = (sum((x - mean) ** 2 for x in r.rollout_lengths)
                                       / len(r.rollout_lengths)) ** 0.5
            total_tokens = sum(r.rollout_lengths)
            d["rl/rollout_tok_per_sec"] = total_tokens / r.rollout_time if r.rollout_time > 0 else 0.0

        if r.acc_at_1:
            d["rl/acc_at_1"] = _mean([float(x) for x in r.acc_at_1])
        if r.acc_at_k:
            d["rl/acc_at_k"] = _mean([float(x) for x in r.acc_at_k])
        if r.coherence_hits:
            d["rl/coherence_rate"] = _mean([float(x) for x in r.coherence_hits])

        # Training metrics
        t = self._training
        if t.policy_loss:
            d["rl/policy_loss"] = _mean(t.policy_loss)
            d["rl/kl_loss"] = _mean(t.kl_loss)
            d["rl/total_loss"] = _mean(t.total_loss)
            d["rl/approx_kl"] = _mean(t.kl)
            d["rl/clip_fraction"] = _mean(t.clip_fraction)
        if t.thinking_entropy:
            d["rl/thinking_entropy"] = _mean(t.thinking_entropy)
        if t.final_entropy:
            d["rl/final_entropy"] = _mean(t.final_entropy)

        if lr is not None:
            d["rl/lr"] = lr

        d["rl/outer_step"] = step

        # Eval metrics
        if eval_results is not None:
            d["eval/cpp_var_mcts_acc"] = eval_results.get("var_mcts_acc", 0.0)
            d["eval/cpp_var_best_acc"] = eval_results.get("var_best_acc", 0.0)
            d["eval/cpp_pt_best_acc"] = eval_results.get("pt_best_acc", 0.0)

        wandb.log(d, step=step)
