import gzip
import struct
import numpy as np
import chess
import sys
import tarfile

import json
import pandas as pd
from policy_index import policy_index

# Lc0 Training Data V6 Format Constants
V6_RECORD_SIZE = 8356
V6_STRUCT_STRING = '<4si7432s832sBBBBBBBbfffffffffffffffIHH4H'
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

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
    
    # Q values (Original = STM perspective)
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
    mirrored_played_move = policy_index[played_idx] if played_idx < len(policy_index) else f"unknown({played_idx})"
    mirrored_best_move = policy_index[best_idx] if best_idx < len(policy_index) else f"unknown({best_idx})"
    
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
        
    # Flattened row for Parquet
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
        
    # Calculate WDL (Side-to-Move perspective)
    # win = (1 + q - d) / 2
    # draw = d
    # loss = (1 - q - d) / 2
    q = row["best_q"]
    d = row["best_d"]
    row["win"] = (1.0 + q - d) / 2.0
    row["draw"] = d
    row["loss"] = (1.0 - q - d) / 2.0
        
    return row

def assign_game_ids(rows, base_id):
    if not rows: return []
    
    game_id_counter = 0
    current_ply = 1
    
    processed_rows = []
    
    for i, row in enumerate(rows):
        if i == 0:
            row["game_id"] = f"{base_id}_{game_id_counter}"
            row["ply"] = current_ply
            processed_rows.append(row)
            continue
            
        prev_row = processed_rows[-1]
        board = chess.Board(prev_row['fen'])
        try:
            move = chess.Move.from_uci(prev_row['played_move'])
            board.push(move)
            if board.fen().split()[:4] == row['fen'].split()[:4]:
                current_ply += 1
            else:
                game_id_counter += 1
                current_ply = 1
        except:
            game_id_counter += 1
            current_ply = 1
            
        row["game_id"] = f"{base_id}_{game_id_counter}"
        row["ply"] = current_ply
        processed_rows.append(row)
        
    return processed_rows

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run --with pandas --with pyarrow reconstitute_games.py <path_to_tar>")
        sys.exit(1)
        
    tar_path = sys.argv[1]
    output_path = tar_path.replace(".tar", ".parquet")
    base_id = tar_path.split('/')[-1].replace('.tar', '')
    
    print(f"Processing {tar_path}...")
    
    all_rows = []
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.gz'):
                f = tar.extractfile(member)
                if f:
                    with gzip.open(f, 'rb') as gz:
                        while True:
                            record = gz.read(V6_RECORD_SIZE)
                            if len(record) < V6_RECORD_SIZE: break
                            all_rows.append(process_record(record))
    
    print(f"Extracted {len(all_rows)} positions. Assigning game IDs and plies...")
    processed_rows = assign_game_ids(all_rows, base_id)
    
    if not processed_rows:
        print(f"Warning: No valid training data found in {tar_path}. No Parquet file created.")
        return

    print(f"Saving to Parquet: {output_path}")
    df = pd.DataFrame(processed_rows)
    df.to_parquet(output_path, index=False)
    
    print(f"Done. Processed {len(df)} positions across {df['game_id'].nunique()} games.")

if __name__ == "__main__":
    main()
