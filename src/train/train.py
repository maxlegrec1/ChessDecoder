import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from datetime import datetime
from tqdm import tqdm
from src.models.model import ChessDecoder
from src.models.vocab import (vocab_size, token_to_idx, board_vocab_size, move_vocab_size,
                              board_idx_to_full_idx, move_idx_to_full_idx)
from src.dataloader.loader import get_dataloader
from src.utils.distributed import (
    setup_distributed, cleanup_distributed, is_main_process, get_device,
    average_gradients, barrier, print_rank0,
)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def soft_bucket_loss(logits, target_values, bucket_centers, valid_mask):
    """Soft CE loss: distribute target probability across two nearest buckets via linear interpolation."""
    N = target_values.shape[0]
    n_buckets = bucket_centers.shape[0]

    if N == 0 or valid_mask.sum() == 0:
        # Return zero without requires_grad to avoid memory leak from retaining computation graph
        return torch.tensor(0.0, device=logits.device)

    # Find lower bucket: last bucket where center <= target
    diffs = target_values.unsqueeze(-1) - bucket_centers  # (N, B)
    lower_idx = (diffs >= 0).long().sum(dim=-1) - 1        # (N,)
    lower_idx = lower_idx.clamp(0, n_buckets - 2)
    upper_idx = lower_idx + 1

    lower_centers = bucket_centers[lower_idx]
    upper_centers = bucket_centers[upper_idx]
    span = (upper_centers - lower_centers).clamp(min=1e-8)
    upper_weight = (target_values - lower_centers) / span
    upper_weight = upper_weight.clamp(0.0, 1.0)

    soft_labels = torch.zeros(N, n_buckets, device=logits.device)
    soft_labels.scatter_(1, lower_idx.unsqueeze(1), (1 - upper_weight).unsqueeze(1))
    soft_labels.scatter_(1, upper_idx.unsqueeze(1), upper_weight.unsqueeze(1))

    loss = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)  # (N,)
    return (loss * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)


def train():
    config = load_config("src/train/config.yaml")

    rank, local_rank, world_size = setup_distributed()
    device = get_device(local_rank)
    print_rank0(f"Using device: {device}, world_size: {world_size}")

    # Check if resuming from checkpoint
    resume_from = config["training"].get("resume_from")
    seed = config["training"].get("seed", 42)
    start_epoch = 0
    step = 0
    resume_epoch_step = 0  # batches to skip in the first epoch on resume

    # Model
    model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        max_seq_len=config["model"]["max_seq_len"],
        d_ff=config["model"].get("d_ff"),
        n_buckets=config["model"].get("n_buckets", 100),
        value_hidden_size=config["model"].get("value_hidden_size", 256),
        num_fourier_freq=config["model"].get("num_fourier_freq", 128),
        wl_sigma=config["model"].get("wl_sigma", 0.4),
    ).to(device)

    # Dataloader
    dataloader = get_dataloader(
        config["data"]["parquet_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
        max_seq_len=config["data"]["max_seq_len"],
        skip_board_prob=config["data"].get("skip_board_prob", 0.0),
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    # Loss functions for board generation and move prediction
    IGNORE_INDEX = -100
    board_ce_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='none')
    move_ce_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='none')

    # Mixed precision training
    use_amp = config["training"].get("use_amp", False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print_rank0(f"Mixed precision training: {'enabled' if use_amp else 'disabled'}")

    # Resume from checkpoint if specified
    if resume_from:
        checkpoint_files = [f for f in os.listdir(resume_from) if f.startswith("checkpoint_") and f.endswith(".pt")]
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {resume_from}")

        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
        latest_checkpoint = os.path.join(resume_from, checkpoint_files[-1])

        print_rank0(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)

        state_dict = checkpoint["model_state_dict"]
        def migrate_state_dict(state_dict):
            """Migrate old checkpoint: sub-vocab heads, expand embedding, clone thinking_policy_head."""
            # Expand tok_embedding if needed
            if "tok_embedding.weight" in state_dict:
                t = state_dict["tok_embedding.weight"]
                if t.shape[0] < vocab_size:
                    pad = torch.zeros(vocab_size - t.shape[0], *t.shape[1:], dtype=t.dtype, device=t.device)
                    state_dict["tok_embedding.weight"] = torch.cat([t, pad], dim=0)

            # board_head: extract rows for board sub-vocab from old full-vocab weights
            if "board_head.weight" in state_dict:
                old_w = state_dict["board_head.weight"]
                old_b = state_dict["board_head.bias"]
                if old_w.shape[0] > board_vocab_size:
                    old_vocab_sz = old_w.shape[0]
                    new_w = torch.zeros(board_vocab_size, old_w.shape[1], dtype=old_w.dtype, device=old_w.device)
                    new_b = torch.zeros(board_vocab_size, dtype=old_b.dtype, device=old_b.device)
                    for i, full_idx in enumerate(board_idx_to_full_idx):
                        if full_idx < old_vocab_sz:
                            new_w[i] = old_w[full_idx]
                            new_b[i] = old_b[full_idx]
                    state_dict["board_head.weight"] = new_w
                    state_dict["board_head.bias"] = new_b

            # policy_head / thinking_policy_head: extract rows for move sub-vocab
            for head in ["policy_head", "thinking_policy_head"]:
                if f"{head}.weight" not in state_dict:
                    continue
                old_w = state_dict[f"{head}.weight"]
                old_b = state_dict[f"{head}.bias"]
                if old_w.shape[0] > move_vocab_size:
                    old_vocab_sz = old_w.shape[0]
                    new_w = torch.zeros(move_vocab_size, old_w.shape[1], dtype=old_w.dtype, device=old_w.device)
                    new_b = torch.zeros(move_vocab_size, dtype=old_b.dtype, device=old_b.device)
                    for i, full_idx in enumerate(move_idx_to_full_idx):
                        if full_idx < old_vocab_sz:
                            new_w[i] = old_w[full_idx]
                            new_b[i] = old_b[full_idx]
                    state_dict[f"{head}.weight"] = new_w
                    state_dict[f"{head}.bias"] = new_b

            # Clone policy_head -> thinking_policy_head (if not already present)
            if "thinking_policy_head.weight" not in state_dict and "policy_head.weight" in state_dict:
                state_dict["thinking_policy_head.weight"] = state_dict["policy_head.weight"].clone()
                state_dict["thinking_policy_head.bias"] = state_dict["policy_head.bias"].clone()

            return state_dict
        state_dict = migrate_state_dict(state_dict)
        # # Checkpoint compatibility: rename old policy_head -> board_head
        # if "policy_head.weight" in state_dict and "board_head.weight" not in state_dict:
        #     state_dict["board_head.weight"] = state_dict.pop("policy_head.weight")
        #     state_dict["board_head.bias"] = state_dict.pop("policy_head.bias")
        # # Remove old 3-class value head (incompatible shape)
        # state_dict.pop("value_head.weight", None)
        # state_dict.pop("value_head.bias", None)

        model.load_state_dict(state_dict, strict=False)

        # Migrate optimizer state dict: handle shape changes from vocab expansion / new params
        opt_state = checkpoint["optimizer_state_dict"]
        old_n_params = len(opt_state["param_groups"][0]["params"])
        new_n_params = len(list(model.parameters()))
        print_rank0(f"Migrating optimizer state: {old_n_params} -> {new_n_params} params")
        # Update param_groups to match current optimizer
        opt_state["param_groups"] = optimizer.state_dict()["param_groups"]
        existing_step = next(iter(opt_state["state"].values()))["step"] if opt_state["state"] else torch.tensor(0)
        step_val = existing_step.clone() if torch.is_tensor(existing_step) else torch.tensor(existing_step)
        for i, param in enumerate(model.parameters()):
            if i not in opt_state["state"]:
                # New parameter index: initialize zero state
                opt_state["state"][i] = {
                    "step": step_val.clone(),
                    "exp_avg": torch.zeros_like(param),
                    "exp_avg_sq": torch.zeros_like(param),
                }
            else:
                old_state = opt_state["state"][i]
                old_exp_avg = old_state["exp_avg"]
                # Check if shapes match or can be expanded along vocab dim only
                if old_exp_avg.shape == param.shape:
                    continue  # No migration needed
                elif len(old_exp_avg.shape) == len(param.shape) and old_exp_avg.shape[1:] == param.shape[1:] and old_exp_avg.shape[0] < param.shape[0]:
                    # Vocab expansion: pad dim 0 with zeros
                    pad_size = param.shape[0] - old_exp_avg.shape[0]
                    for key in ["exp_avg", "exp_avg_sq"]:
                        old_t = old_state[key]
                        pad = torch.zeros(pad_size, *old_t.shape[1:], dtype=old_t.dtype, device=old_t.device)
                        old_state[key] = torch.cat([old_t, pad], dim=0)
                else:
                    # Incompatible shape (param indices shifted): reinitialize
                    old_state["exp_avg"] = torch.zeros_like(param)
                    old_state["exp_avg_sq"] = torch.zeros_like(param)
        optimizer.load_state_dict(opt_state)
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        step = checkpoint["step"]
        resume_epoch_step = checkpoint.get("epoch_step", 0)

        print_rank0(f"Resumed from epoch {start_epoch}, step {step}, epoch_step {resume_epoch_step}")

        run_checkpoint_dir = resume_from
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_checkpoint_dir = os.path.join(
            config["training"]["checkpoint_dir"],
            f"{config['run_name']}_{timestamp}"
        )
        if is_main_process():
            os.makedirs(run_checkpoint_dir, exist_ok=True)
        barrier()

    print_rank0(f"Checkpoints will be saved to: {run_checkpoint_dir}")

    # Initialize wandb (resume existing run if ID file exists in checkpoint dir)
    if is_main_process():
        wandb_id_path = os.path.join(run_checkpoint_dir, "wandb_run_id.txt")
        wandb_run_id = None
        if os.path.exists(wandb_id_path):
            wandb_run_id = open(wandb_id_path).read().strip()
            print(f"Resuming wandb run: {wandb_run_id}")

        wandb.init(
            project=config["project_name"],
            name=config["run_name"],
            config=config,
            id=wandb_run_id,
            resume="must" if wandb_run_id else None,
        )

        if not wandb_run_id:
            with open(wandb_id_path, "w") as f:
                f.write(wandb.run.id)
            print(f"Saved wandb run ID to {wandb_id_path}")

    model.train()

    # Loss weights
    move_weight = config["loss"]["move_weight"]
    board_weight = config["loss"]["board_weight"]
    wl_weight = config["loss"].get("wl_weight", 1.0)
    d_weight = config["loss"].get("d_weight", 1.0)

    grad_accum = config["training"].get("gradient_accumulation_steps", 1)

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        print_rank0(f"Epoch {epoch+1}/{config['training']['num_epochs']}")

        # Set epoch on dataset for deterministic seeding (workers pick this up)
        dataloader.dataset.epoch = epoch

        # On resume, fast-forward the dataloader to where we left off
        skip_batches = resume_epoch_step
        resume_epoch_step = 0  # only skip on the first epoch after resume
        if skip_batches > 0:
            print_rank0(f"Fast-forwarding dataloader by {skip_batches} batches...")

        epoch_step = 0

        for batch in tqdm(dataloader, disable=not is_main_process()):
            epoch_step += 1
            if epoch_step <= skip_batches:
                if epoch_step == skip_batches:
                    print_rank0(f"Fast-forwarded {skip_batches} batches, resuming training...")
                continue

            input_ids = batch["input_ids"].to(device)
            board_target_ids = batch["board_target_ids"].to(device)
            move_target_ids = batch["move_target_ids"].to(device)
            move_mask = batch["move_mask"].to(device)
            wl_positions = batch["wl_positions"].to(device)
            d_positions = batch["d_positions"].to(device)
            wl_targets = batch["wl_targets"].to(device)
            d_targets = batch["d_targets"].to(device)
            wdl_valid = batch["wdl_valid"].to(device)
            block_id = batch["block_id"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                # === Prepare Fourier inputs (shared by both passes) ===
                wl_fourier_input = torch.zeros_like(wl_targets)
                d_fourier_input = torch.zeros_like(d_targets)

                if wl_positions.any():
                    wl_vals_at_pos = wl_targets[wl_positions]
                    wl_disc = model.discretize_to_bucket(wl_vals_at_pos, model.wl_bucket_centers)
                    wl_fourier_input[wl_positions] = wl_disc

                if d_positions.any():
                    d_vals_at_pos = d_targets[d_positions]
                    d_disc = model.discretize_to_bucket(d_vals_at_pos, model.d_bucket_centers)
                    d_fourier_input[d_positions] = d_disc

                # === Pass 1: Causal masking for board generation ===
                h_causal = model(
                    input_ids, mask_type="causal",
                    wl_values=wl_fourier_input, d_values=d_fourier_input,
                    wl_positions=wl_positions, d_positions=d_positions
                )
                board_logits = model.board_head(h_causal)

                # Board mask: non-ignored positions, excluding pre-first-move
                board_mask = board_target_ids != IGNORE_INDEX

                first_move_idx = move_mask.int().argmax(dim=1)  # (B,)
                has_moves = move_mask.any(dim=1)
                first_move_idx[~has_moves] = move_mask.size(1)
                indices = torch.arange(move_mask.size(1), device=device).unsqueeze(0)
                pre_first_move_mask = indices < first_move_idx.unsqueeze(1)
                board_mask = board_mask & (~pre_first_move_mask)

                ce_board = board_ce_fn(board_logits.view(-1, board_vocab_size), board_target_ids.view(-1))
                ce_board = ce_board.view(board_target_ids.shape)
                board_loss = (ce_board * board_mask.float()).sum() / (board_mask.sum() + 1e-8)

                # === Pass 2: Prefix masking for move + value prediction ===
                h_prefix = model(
                    input_ids, mask_type="prefix", block_id=block_id,
                    wl_values=wl_fourier_input, d_values=d_fourier_input,
                    wl_positions=wl_positions, d_positions=d_positions
                )

                # --- 1. Move prediction at stm positions ---
                move_logits = model.policy_head(h_prefix)
                ce_move = move_ce_fn(move_logits.view(-1, move_vocab_size), move_target_ids.view(-1))
                ce_move = ce_move.view(move_target_ids.shape)
                move_loss = (ce_move * move_mask.float()).sum() / (move_mask.sum() + 1e-8)

                # --- 2. WL prediction at move token positions (stm + 1) ---
                stm_nonzero = move_mask.nonzero(as_tuple=False)  # (N, 2)
                if stm_nonzero.shape[0] > 0:
                    wl_pred_batch = stm_nonzero[:, 0]
                    wl_pred_seq = stm_nonzero[:, 1] + 1  # move token = stm + 1
                    wl_pred_seq = wl_pred_seq.clamp(max=h_prefix.shape[1] - 1)
                    h_at_move = h_prefix[wl_pred_batch, wl_pred_seq]  # (N, E)
                    wl_logits = model.wl_head(h_at_move)  # (N, 100)
                    wl_valid_flat = wdl_valid[move_mask]  # (N,)
                    wl_gt_flat = wl_targets[move_mask]    # (N,)
                    wl_loss = soft_bucket_loss(wl_logits, wl_gt_flat, model.wl_bucket_centers, wl_valid_flat)
                else:
                    wl_loss = torch.tensor(0.0, device=device)
                    wl_logits = None

                # --- 3. D prediction at WL placeholder positions (stm + 2) ---
                if wl_positions.any():
                    h_at_wl = h_prefix[wl_positions]  # (M, E)
                    d_logits = model.d_head(h_at_wl)   # (M, 100)
                    d_valid_flat = wdl_valid[d_positions]
                    d_gt_flat = d_targets[d_positions]
                    d_loss = soft_bucket_loss(d_logits, d_gt_flat, model.d_bucket_centers, d_valid_flat)
                else:
                    d_loss = torch.tensor(0.0, device=device)
                    d_logits = None

                # === Total loss ===
                total_loss = (
                    move_weight * move_loss +
                    board_weight * board_loss +
                    wl_weight * wl_loss +
                    d_weight * d_loss
                )

            # Scale loss for gradient accumulation
            loss = total_loss / grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                average_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Detach tensors used for metrics to prevent memory leak from retained computation graph
            board_logits = board_logits.detach()
            move_logits = move_logits.detach()
            if wl_logits is not None:
                wl_logits = wl_logits.detach()
            if d_logits is not None:
                d_logits = d_logits.detach()

            # Metrics
            with torch.no_grad():
                # Move accuracy (compare in move sub-vocab space)
                preds_prefix = torch.argmax(move_logits, dim=-1)
                move_correct = (preds_prefix == move_target_ids) & move_mask
                move_acc = move_correct.sum() / (move_mask.sum() + 1e-8)

                # Board accuracy (per-token, compare in board sub-vocab space)
                preds_causal = torch.argmax(board_logits, dim=-1)
                board_correct = (preds_causal == board_target_ids) & board_mask
                board_acc = board_correct.sum() / (board_mask.sum() + 1e-8)

                # Board accuracy (per-block, grouped by block_id)
                B, S = block_id.shape
                max_bid = block_id.max() + 1
                uid = block_id + (torch.arange(B, device=device) * max_bid).unsqueeze(1)
                flat_uid = uid.view(-1)
                flat_bc = board_correct.view(-1).float()
                flat_bm = board_mask.view(-1)
                bp = flat_bm.nonzero(as_tuple=True)[0]

                if bp.numel() > 0:
                    bp_uid = flat_uid[bp]
                    bp_bc = flat_bc[bp]
                    unique_uids, inv = bp_uid.unique(return_inverse=True)
                    n_unique = unique_uids.shape[0]
                    sum_correct = torch.zeros(n_unique, device=device).scatter_add_(0, inv, bp_bc)
                    sum_total = torch.zeros(n_unique, device=device).scatter_add_(0, inv, torch.ones_like(bp_bc))
                    board_total_acc = (sum_correct == sum_total).float().mean().item()

                    # Intra-block offset for sub-metrics
                    bp_pos = bp.float()
                    block_starts = torch.full((n_unique,), float('inf'), device=device)
                    block_starts.scatter_reduce_(0, inv, bp_pos, reduce="amin")
                    intra_idx = (bp_pos - block_starts[inv]).long()

                    # Squares (intra 1-64): per-board all-correct
                    sq = (intra_idx >= 1) & (intra_idx <= 64)
                    if sq.any():
                        sq_correct = torch.zeros(n_unique, device=device).scatter_add_(0, inv[sq], bp_bc[sq])
                        sq_total = torch.zeros(n_unique, device=device).scatter_add_(0, inv[sq], torch.ones_like(bp_bc[sq]))
                        has_sq = sq_total > 0
                        board_square_acc = ((sq_correct == sq_total) & has_sq).float().sum() / (has_sq.sum() + 1e-8)
                        board_square_acc = board_square_acc.item()
                    else:
                        board_square_acc = 0.0

                    # Castling (intra 65: end_pos predicts castling token)
                    castle = intra_idx == 65
                    board_castling_acc = bp_bc[castle].mean().item() if castle.any() else 0.0

                    # STM (intra 66: castling predicts STM token)
                    stm_metric = intra_idx == 66
                    board_stm_acc = bp_bc[stm_metric].mean().item() if stm_metric.any() else 0.0
                else:
                    board_total_acc = 0.0
                    board_square_acc = 0.0
                    board_castling_acc = 0.0
                    board_stm_acc = 0.0

                # WL MAE (expected value) and MSE (argmax bucket center)
                if wl_logits is not None and stm_nonzero.shape[0] > 0:
                    wl_probs = F.softmax(wl_logits.float(), dim=-1)
                    expected_wl = (wl_probs * model.wl_bucket_centers.unsqueeze(0)).sum(dim=-1)
                    wl_valid_f = wl_valid_flat.float()
                    wl_mae = ((expected_wl - wl_gt_flat).abs() * wl_valid_f).sum() / (wl_valid_f.sum() + 1e-8)
                    argmax_wl = model.wl_bucket_centers[wl_logits.argmax(dim=-1)]
                    wl_mse = (((argmax_wl - wl_gt_flat) ** 2) * wl_valid_f).sum() / (wl_valid_f.sum() + 1e-8)
                else:
                    wl_mae = torch.tensor(0.0, device=device)
                    wl_mse = torch.tensor(0.0, device=device)

                # D MAE (expected value) and MSE (argmax bucket center)
                if d_logits is not None and wl_positions.any():
                    d_probs = F.softmax(d_logits.float(), dim=-1)
                    expected_d = (d_probs * model.d_bucket_centers.unsqueeze(0)).sum(dim=-1)
                    d_valid_f = d_valid_flat.float()
                    d_mae = ((expected_d - d_gt_flat).abs() * d_valid_f).sum() / (d_valid_f.sum() + 1e-8)
                    argmax_d = model.d_bucket_centers[d_logits.argmax(dim=-1)]
                    d_mse = (((argmax_d - d_gt_flat) ** 2) * d_valid_f).sum() / (d_valid_f.sum() + 1e-8)
                else:
                    d_mae = torch.tensor(0.0, device=device)
                    d_mse = torch.tensor(0.0, device=device)

                # nth-move metrics
                max_track_moves = config["training"].get("max_track_nth_moves", 20)
                move_cumsum = move_mask.cumsum(dim=1)
                move_indices = (move_cumsum - 1).long()
                valid_nth_mask = move_mask & (move_indices < max_track_moves)

                if valid_nth_mask.any():
                    flat_move_indices = move_indices[valid_nth_mask]
                    flat_move_correct = move_correct[valid_nth_mask].float()

                    move_correct_by_nth = torch.zeros(max_track_moves, device=device)
                    count_by_nth = torch.zeros(max_track_moves, device=device)

                    move_correct_by_nth.scatter_add_(0, flat_move_indices, flat_move_correct)
                    count_by_nth.scatter_add_(0, flat_move_indices, torch.ones_like(flat_move_correct))

                    move_acc_by_nth = move_correct_by_nth / (count_by_nth + 1e-8)
                else:
                    move_acc_by_nth = torch.zeros(max_track_moves, device=device)
                    count_by_nth = torch.zeros(max_track_moves, device=device)

            if step % config["training"]["log_every_n_steps"] == 0 and is_main_process():
                print(f"Step {step}: Loss {total_loss.item():.4f} "
                      f"(Move: {move_loss.item():.4f}, Board: {board_loss.item():.4f}, "
                      f"WL: {wl_loss.item():.4f}, D: {d_loss.item():.4f})")

                log_dict = {
                    "train/total_loss": total_loss.item(),
                    "train/move_loss": move_loss.item(),
                    "train/board_loss": board_loss.item(),
                    "train/wl_loss": wl_loss.item(),
                    "train/d_loss": d_loss.item(),
                    "train/move_acc": move_acc.item(),
                    "train/board_acc": board_acc.item(),
                    "train/board_total_acc": board_total_acc,
                    "train/board_square_acc": board_square_acc,
                    "train/board_castling_acc": board_castling_acc,
                    "train/board_stm_acc": board_stm_acc,
                    "train/wl_mae": wl_mae.item(),
                    "train/wl_mse": wl_mse.item(),
                    "train/d_mae": d_mae.item(),
                    "train/d_mse": d_mse.item(),
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/grad_step": step / grad_accum,
                }

                for i in range(max_track_moves):
                    if count_by_nth[i] > 0:
                        log_dict[f"train/move_acc_nth/{i}"] = move_acc_by_nth[i].item()

                wandb.log(log_dict)

            # Explicitly delete heavy tensors to prevent memory leak
            del h_causal, h_prefix, board_logits, move_logits
            del wl_logits, d_logits
            del total_loss, loss

            step += 1

            # Save checkpoint every N steps
            save_every = config["training"].get("save_every_n_steps")
            if save_every and step % save_every == 0 and is_main_process():
                checkpoint = {
                    "epoch": epoch,
                    "step": step,
                    "epoch_step": epoch_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "config": config,
                }
                checkpoint_path = os.path.join(run_checkpoint_dir, f"checkpoint_{step}.pt")
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                barrier()

        # Save checkpoint at end of epoch (epoch_step=0 so next epoch starts fresh)
        if is_main_process():
            checkpoint = {
                "epoch": epoch + 1,
                "step": step,
                "epoch_step": 0,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
            }
            checkpoint_path = os.path.join(run_checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        barrier()

    cleanup_distributed()

if __name__ == "__main__":
    train()
