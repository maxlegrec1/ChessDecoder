import pandas as pd
from src.dataloader.data import game_to_sequence

def main():
    df = pd.read_parquet('parquets/training-run2-test91-20251106-0917.parquet')
    
    # Pick a game_id
    game_id = df['game_id'].iloc[0]
    game_df = df[df['game_id'] == game_id].sort_values('ply')
    
    print(f"Extracting sequence for Game ID: {game_id}")
    print(f"Number of plies: {len(game_df)}")
    
    sequence = game_to_sequence(game_df)
    
    print("\nFull Game Sequence (first 100 tokens):")
    print(", ".join(sequence[:100]))
    
    print(f"\nTotal tokens in sequence: {len(sequence)}")
    
    # Show a transition between positions
    # Find the first move token
    first_move_idx = -1
    for i, token in enumerate(sequence):
        if len(token) == 4 and token[0] in "abcdefgh" and token[1] in "12345678":
            first_move_idx = i
            break
            
    if first_move_idx != -1:
        print("\nTransition example (Position -> Move -> Next Position):")
        start = max(0, first_move_idx - 5)
        end = min(len(sequence), first_move_idx + 10)
        print(", ".join(sequence[start:end]))

if __name__ == "__main__":
    main()
