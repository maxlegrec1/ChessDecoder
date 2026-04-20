import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import glob
import os
import math
import random
from chessdecoder.dataloader.data import fen_to_position_tokens
from chessdecoder.models.vocab import token_to_idx, policy_to_idx


# Avg tokens the decoder emits per ply (68 board + stm + move + wl + d).
# Measured at 71.0 across sampled games in the pretrain parquets.
TOKENS_PER_PLY = 71


class ChessEncoderDataset(IterableDataset):
    """
    Dataset for encoder model: extracts (FEN, best_move) pairs.

    Two sampling modes controlled by ``match_decoder_sampling``:

    * ``False`` (default): yield every row, shuffled within each parquet file.
      Distribution is uniform over positions in the dataset.

    * ``True``: mirror the decoder's ``loader.py`` distribution — group rows
      by ``game_id``, pick one random start ply per game, then yield the
      next ``decoder_max_seq_len // TOKENS_PER_PLY`` consecutive plies as
      individual samples. Each game contributes one contiguous window per
      file pass, exactly as the decoder does (see loader.py:76).
      ``decoder_max_seq_len`` should match the decoder's training
      ``max_seq_len`` (e.g. 4096).
    """

    def __init__(
        self,
        parquet_dir: str,
        max_seq_len: int = 128,
        shuffle_files: bool = True,
        shuffle_positions: bool = True,
        match_decoder_sampling: bool = False,
        decoder_max_seq_len: int = 4096,
    ):
        self.parquet_dir = parquet_dir
        self.max_seq_len = max_seq_len
        self.shuffle_files = shuffle_files
        self.shuffle_positions = shuffle_positions
        self.match_decoder_sampling = match_decoder_sampling
        self.decoder_max_seq_len = decoder_max_seq_len
        self.window_size = decoder_max_seq_len // TOKENS_PER_PLY
        self.pad_id = token_to_idx["pad"]

        self.files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if not self.files:
            print(f"Warning: No parquet files found in {parquet_dir}")

    def _row_to_sample(self, fen: str, best_move: str):
        if best_move not in policy_to_idx:
            return None

        tokens = fen_to_position_tokens(fen)
        token_ids = [token_to_idx[t] for t in tokens]
        seq_len = len(token_ids)
        if seq_len > self.max_seq_len:
            return None

        input_ids = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
        input_ids[:seq_len] = torch.tensor(token_ids, dtype=torch.long)

        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        attention_mask[:seq_len] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": torch.tensor(policy_to_idx[best_move], dtype=torch.long),
            "seq_len": seq_len,
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            files_to_read = list(self.files)
        else:
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.files))
            files_to_read = self.files[iter_start:iter_end]

        if self.shuffle_files:
            random.shuffle(files_to_read)

        for file_path in files_to_read:
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

            if self.match_decoder_sampling:
                yield from self._iter_game_windows(df)
            else:
                yield from self._iter_rows(df)

    def _iter_rows(self, df: pd.DataFrame):
        indices = df.index.tolist()
        if self.shuffle_positions:
            random.shuffle(indices)

        for idx in indices:
            row = df.loc[idx]
            sample = self._row_to_sample(row["fen"], row["best_move"])
            if sample is not None:
                yield sample

    def _iter_game_windows(self, df: pd.DataFrame):
        """Decoder-matched: one random ``window_size``-ply contiguous window per game."""
        grouped = df.groupby("game_id", sort=False)
        game_ids = list(grouped.groups.keys())
        if self.shuffle_positions:
            random.shuffle(game_ids)

        for gid in game_ids:
            game_df = grouped.get_group(gid).sort_values("ply")
            n = len(game_df)
            if n == 0:
                continue

            start = random.randint(0, n - 1)  # valid_starts = [0, 1, ..., n-1]
            end = min(start + self.window_size, n)
            window = game_df.iloc[start:end]

            # Emit the window as individual samples — mirrors the decoder's
            # per-stm-position loss across a contiguous game slice.
            for _, row in window.iterrows():
                sample = self._row_to_sample(row["fen"], row["best_move"])
                if sample is not None:
                    yield sample


def get_encoder_dataloader(
    parquet_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    max_seq_len: int = 128,
    match_decoder_sampling: bool = False,
    decoder_max_seq_len: int = 4096,
) -> DataLoader:
    """Create dataloader for encoder training."""
    dataset = ChessEncoderDataset(
        parquet_dir,
        max_seq_len=max_seq_len,
        shuffle_files=True,
        shuffle_positions=True,
        match_decoder_sampling=match_decoder_sampling,
        decoder_max_seq_len=decoder_max_seq_len,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
