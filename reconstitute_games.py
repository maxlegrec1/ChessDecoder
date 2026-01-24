import gzip
import struct
import numpy as np
import chess
import sys
import tarfile
import io
import json
import time
import os
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from policy_index import policy_index


class Timer:
    """Simple context manager for timing code blocks."""
    def __init__(self, name, stats):
        self.name = name
        self.stats = stats
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.stats[self.name] = elapsed


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def format_rate(count, seconds):
    """Format processing rate."""
    if seconds == 0:
        return "N/A"
    rate = count / seconds
    if rate > 1_000_000:
        return f"{rate/1_000_000:.2f}M/s"
    elif rate > 1_000:
        return f"{rate/1_000:.2f}K/s"
    else:
        return f"{rate:.1f}/s"

# Lc0 Training Data V6 Format Constants
V6_RECORD_SIZE = 8356
V6_STRUCT_STRING = '<4si7432s832sBBBBBBBbfffffffffffffffIHH4H'
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

# Pre-compute policy index length for faster bounds check
POLICY_INDEX_LEN = len(policy_index)


def reconstruct_fen(planes_raw, us_ooo, us_oo, them_ooo, them_oo, rule50_count):
    all_bits = np.unpackbits(np.frombuffer(planes_raw, dtype=np.uint8), bitorder='little')
    planes = all_bits.reshape(104, 64)
    board = chess.Board(None)
    
    def get_real_square(sq_idx):
        rank = sq_idx // 8
        file = sq_idx % 8
        real_file = 7 - file
        return rank * 8 + real_file

    for i in range(6):
        for sq_idx in np.where(planes[i] == 1)[0]:
            board.set_piece_at(get_real_square(sq_idx), chess.Piece(PIECE_TYPES[i], chess.WHITE))
    for i in range(6):
        for sq_idx in np.where(planes[i+6] == 1)[0]:
            board.set_piece_at(get_real_square(sq_idx), chess.Piece(PIECE_TYPES[i], chess.BLACK))
            
    board.turn = chess.WHITE
    fen_castling = ""
    if us_oo: fen_castling += "K"
    if us_ooo: fen_castling += "Q"
    if them_oo: fen_castling += "k"
    if them_ooo: fen_castling += "q"
    if not fen_castling: fen_castling = "-"
    board.set_castling_fen(fen_castling)
    board.halfmove_clock = rule50_count
    return board.fen()


def mirror_move_uci(move_uci):
    if move_uci.startswith("unknown"): return move_uci
    try:
        move = chess.Move.from_uci(move_uci)
        mirrored_move = chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion
        )
        return mirrored_move.uci()
    except:
        return move_uci


def process_record(record):
    data = struct.unpack(V6_STRUCT_STRING, record)
    probs_raw = data[2]
    planes_raw = data[3]
    us_ooo, us_oo, them_ooo, them_oo = data[4:8]
    stm_field = data[8]
    rule50_count = data[9]
    
    q_fields = [
        "root_q", "best_q", "root_d", "best_d", "root_m", "best_m",
        "plies_left", "result_q", "result_d", "played_q", "played_d",
        "played_m", "orig_q", "orig_d", "orig_m"
    ]
    q_values = {f: data[12+i] for i, f in enumerate(q_fields)}
    
    visits = data[27]
    played_idx = data[28]
    best_idx = data[29]
    
    mirrored_fen = reconstruct_fen(planes_raw, us_ooo, us_oo, them_ooo, them_oo, rule50_count)
    mirrored_played_move = policy_index[played_idx] if played_idx < POLICY_INDEX_LEN else f"unknown({played_idx})"
    mirrored_best_move = policy_index[best_idx] if best_idx < POLICY_INDEX_LEN else f"unknown({best_idx})"
    
    is_black = (stm_field == 1)
    
    if is_black:
        board = chess.Board(mirrored_fen)
        real_board = board.mirror()
        real_fen = real_board.fen()
        real_played_move = mirror_move_uci(mirrored_played_move)
        real_best_move = mirror_move_uci(mirrored_best_move)
        q_multiplier = -1
    else:
        real_fen = mirrored_fen
        real_played_move = mirrored_played_move
        real_best_move = mirrored_best_move
        q_multiplier = 1
        
    row = {
        "fen": real_fen,
        "played_move": real_played_move,
        "best_move": real_best_move,
        "is_black": is_black,
        "visits": visits,
        "input_format": data[1],
        "rule50": rule50_count,
        "invariance": data[10],
        "dep_result": data[11]
    }
    
    for k, v in q_values.items():
        row[k] = v
        
    q = row["best_q"]
    d = row["best_d"]
    row["win"] = (1.0 + q - d) / 2.0
    row["draw"] = d
    row["loss"] = (1.0 - q - d) / 2.0
        
    return row


def process_batch(records):
    """Process a batch of records - used for multiprocessing."""
    return [process_record(r) for r in records]


def assign_game_ids(rows, base_id):
    """
    Assign game IDs and ply numbers to consecutive positions.
    
    Validates move continuity by:
    1. Expanding FEN to 64-square array for accurate comparison
    2. Simulating the move and checking if it produces the next position
    3. Handling standard chess and Chess960 castling, en passant, promotions
    """
    if not rows: return []
    
    def fen_to_board64(fen):
        """Expand FEN board string to 64-character array (a1=0, h8=63)."""
        board_str = fen.split()[0]
        board = []
        for char in board_str:
            if char == '/':
                continue
            elif char.isdigit():
                board.extend(['.'] * int(char))
            else:
                board.append(char)
        # FEN is rank 8 to rank 1, so reverse to get a1=0
        # Actually FEN gives us a8-h8, a7-h7, ..., a1-h1
        # We want index 0 = a1, so we need to reverse the ranks
        ranks = [board[i:i+8] for i in range(0, 64, 8)]
        ranks.reverse()
        return [sq for rank in ranks for sq in rank]
    
    def square_to_index(sq):
        """Convert algebraic square (e.g., 'e4') to index 0-63."""
        file = ord(sq[0]) - ord('a')  # 0-7
        rank = int(sq[1]) - 1          # 0-7
        return rank * 8 + file
    
    def get_turn(fen):
        """Get side to move from FEN."""
        parts = fen.split()
        return parts[1] if len(parts) > 1 else 'w'
    
    def simulate_move(board, move_uci):
        """
        Apply a move to a board array and return the new board.
        Handles regular moves, captures, castling, en passant, and promotion.
        
        NOTE: Lc0 uses "king captures rook" notation for castling:
          - e1h1 = white kingside (king to g1, rook h1->f1)
          - e1a1 = white queenside (king to c1, rook a1->d1)
          - e8h8 = black kingside (king to g8, rook h8->f8)
          - e8a8 = black queenside (king to c8, rook a8->d8)
        """
        if move_uci.startswith("unknown"):
            return None
            
        new_board = board.copy()
        
        from_sq = move_uci[:2]
        to_sq = move_uci[2:4]
        promotion = move_uci[4] if len(move_uci) > 4 else None
        
        from_idx = square_to_index(from_sq)
        to_idx = square_to_index(to_sq)
        
        piece = new_board[from_idx]
        if piece == '.':
            return None  # No piece at source
        
        # Handle castling - Lc0/Chess960 notation: king moves to rook's square
        # Detect by: king moving to a square occupied by a friendly rook
        is_castling = False
        if piece.lower() == 'k':
            target_piece = new_board[to_idx]
            from_file = from_idx % 8
            to_file = to_idx % 8
            rank = from_idx // 8  # 0 for white (rank 1), 7 for black (rank 8)
            
            # Check if target has a friendly rook (same color as king)
            is_friendly_rook = (
                (piece == 'K' and target_piece == 'R') or  # White king to white rook
                (piece == 'k' and target_piece == 'r')     # Black king to black rook
            )
            
            # Standard chess: king on e-file moving to a/h file corner
            is_standard_castling = (from_file == 4 and to_file in (0, 7))
            
            if is_friendly_rook or is_standard_castling:
                is_castling = True
                
                # Determine kingside vs queenside by which side the rook is
                # In Chess960, rook can be anywhere, so check relative to king
                if is_friendly_rook:
                    rook_from = to_idx  # King is moving TO the rook's square
                    is_kingside = (to_file > from_file)
                else:
                    # Standard chess: rook at corner
                    rook_from = rank * 8 + (7 if to_file == 7 else 0)
                    is_kingside = (to_file == 7)
                
                # Standard castling destinations (same for standard and Chess960)
                if is_kingside:
                    king_to = rank * 8 + 6      # g1 or g8
                    rook_to = rank * 8 + 5      # f1 or f8
                else:
                    king_to = rank * 8 + 2      # c1 or c8
                    rook_to = rank * 8 + 3      # d1 or d8
                
                rook = new_board[rook_from]
                new_board[from_idx] = '.'       # King leaves original square
                new_board[rook_from] = '.'      # Rook leaves original square
                new_board[king_to] = piece      # King to final square
                new_board[rook_to] = rook       # Rook to final square
        
        if not is_castling:
            # Handle en passant (pawn captures diagonally to empty square)
            if piece.lower() == 'p' and from_idx % 8 != to_idx % 8 and board[to_idx] == '.':
                # Diagonal pawn move to empty square = en passant
                captured_idx = to_idx - 8 if piece.isupper() else to_idx + 8
                new_board[captured_idx] = '.'
            
            # Move the piece
            new_board[from_idx] = '.'
            if promotion:
                # Promotion: use uppercase if white pawn, lowercase if black
                new_board[to_idx] = promotion.upper() if piece.isupper() else promotion.lower()
            else:
                new_board[to_idx] = piece
        
        return new_board
    
    def move_makes_sense(prev_fen, prev_move, curr_fen):
        """
        Check if applying prev_move to prev_fen results in curr_fen's board position.
        """
        if prev_move.startswith("unknown"):
            return False
        
        # Check turn alternation first (fast check)
        prev_turn = get_turn(prev_fen)
        curr_turn = get_turn(curr_fen)
        expected_turn = 'b' if prev_turn == 'w' else 'w'
        if curr_turn != expected_turn:
            return False
        
        try:
            prev_board = fen_to_board64(prev_fen)
            curr_board = fen_to_board64(curr_fen)
            
            # Simulate the move
            simulated = simulate_move(prev_board, prev_move)
            if simulated is None:
                return False
            
            # Compare board positions (ignore metadata like castling rights, en passant)
            return simulated == curr_board
            
        except Exception:
            return False
    
    game_id_counter = 0
    current_ply = 1
    
    # First row
    rows[0]["game_id"] = f"{base_id}_{game_id_counter}"
    rows[0]["ply"] = current_ply
    
    # Process remaining rows
    for i in range(1, len(rows)):
        prev_row = rows[i - 1]
        curr_row = rows[i]
        
        if move_makes_sense(prev_row['fen'], prev_row['played_move'], curr_row['fen']):
            current_ply += 1
        else:
            game_id_counter += 1
            current_ply = 1
            
        rows[i]["game_id"] = f"{base_id}_{game_id_counter}"
        rows[i]["ply"] = current_ply
        
    return rows


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run --with pandas --with pyarrow reconstitute_games.py <path_to_tar>")
        sys.exit(1)
    
    tar_path = sys.argv[1]
    output_path = tar_path.replace(".tar", ".parquet")
    base_id = tar_path.split('/')[-1].replace('.tar', '')
    
    total_start = time.perf_counter()
    stats = {}
    
    print(f"{'='*60}")
    print(f"Processing: {tar_path}")
    print(f"Output:     {output_path}")
    print(f"Workers:    {cpu_count()}")
    print(f"{'='*60}")
    
    # 1. Read all raw records into memory first (bulk read with parallel decompression)
    print(f"\n[1/4] Reading tar archive...")
    with Timer("tar_read", stats):
        # First, extract all gzip file contents from tar (sequential - tar reading)
        gz_contents = []
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.gz'):
                    f = tar.extractfile(member)
                    if f:
                        gz_contents.append(f.read())
        
        gz_file_count = len(gz_contents)
        
        # Decompress and parse records in parallel (CPU-bound, releases GIL)
        def decompress_and_parse(compressed_data):
            records = []
            data = gzip.decompress(compressed_data)
            for i in range(0, len(data), V6_RECORD_SIZE):
                chunk = data[i:i+V6_RECORD_SIZE]
                if len(chunk) == V6_RECORD_SIZE:
                    records.append(chunk)
            return records, len(data)
        
        raw_records = []
        total_bytes = 0
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            results = executor.map(decompress_and_parse, gz_contents)
            for records, nbytes in results:
                raw_records.extend(records)
                total_bytes += nbytes
    
    print(f"      └─ {gz_file_count} gzip files, {total_bytes/1024/1024:.1f} MB decompressed")
    print(f"      └─ {len(raw_records):,} records extracted")
    print(f"      └─ Time: {format_time(stats['tar_read'])} ({format_rate(len(raw_records), stats['tar_read'])})")
    
    # 2. Process records in parallel
    print(f"\n[2/4] Processing records (parallel)...")
    with Timer("process_records", stats):
        batch_size = max(1000, len(raw_records) // (cpu_count() * 4))
        batches = [raw_records[i:i+batch_size] for i in range(0, len(raw_records), batch_size)]
        
        with Pool(cpu_count()) as pool:
            results = pool.map(process_batch, batches)
        
        all_rows = [row for batch in results for row in batch]
    
    print(f"      └─ {len(batches)} batches, ~{batch_size:,} records/batch")
    print(f"      └─ {len(all_rows):,} positions extracted")
    print(f"      └─ Time: {format_time(stats['process_records'])} ({format_rate(len(all_rows), stats['process_records'])})")
    
    # 3. Assign game IDs (sequential - requires previous row context)
    print(f"\n[3/4] Assigning game IDs and plies...")
    with Timer("assign_game_ids", stats):
        processed_rows = assign_game_ids(all_rows, base_id)
    
    if not processed_rows:
        print(f"      └─ Warning: No valid training data found!")
        return
    
    num_games = len(set(r['game_id'] for r in processed_rows))
    print(f"      └─ {num_games:,} games identified")
    print(f"      └─ Time: {format_time(stats['assign_game_ids'])} ({format_rate(len(processed_rows), stats['assign_game_ids'])})")

    # 4. Write to Parquet
    print(f"\n[4/4] Writing Parquet file...")
    with Timer("parquet_write", stats):
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        table = pa.Table.from_pylist(processed_rows)
        pq.write_table(table, output_path)
    
    output_size = os.path.getsize(output_path)
    print(f"      └─ Output size: {output_size/1024/1024:.1f} MB")
    print(f"      └─ Time: {format_time(stats['parquet_write'])}")
    
    # Summary
    total_time = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"PROFILING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Stage':<30} {'Time':>12} {'%':>8}")
    print(f"{'-'*50}")
    for name, elapsed in stats.items():
        pct = (elapsed / total_time) * 100
        print(f"{name:<30} {format_time(elapsed):>12} {pct:>7.1f}%")
    print(f"{'-'*50}")
    print(f"{'TOTAL':<30} {format_time(total_time):>12} {'100.0%':>8}")
    print(f"{'='*60}")
    print(f"Done! {len(processed_rows):,} positions across {num_games:,} games.")


if __name__ == "__main__":
    main()