"""
Evaluate accuracy on excluded games, split by standard vs non-standard (960).
"""
import os, random, time, sys, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import chess
import _decoder_inference_cpp as cpp

_PS = {'e1h1':'e1g1','e1a1':'e1c1','e8h8':'e8g8','e8a8':'e8c8'}
def norm(m): return _PS.get(m, m) if isinstance(m, str) else m

def _is_standard_game(origin_fen, moves):
    board = chess.Board(origin_fen)
    for move_str in moves:
        uci = _PS.get(move_str, move_str)
        try:
            move = board.parse_uci(uci)
        except (chess.InvalidMoveError, chess.IllegalMoveError):
            return False
        if move not in board.legal_moves:
            return False
        board.push(move)
    return True

def sample_one_per_game(df, seed):
    def _pick(g):
        gs = hash((seed, g.name)) % (2**31)
        return g.sample(1, random_state=gs)
    return df.groupby('game_id', group_keys=False).apply(_pick).reset_index(drop=True)

seed = 42
pt_dir = '/home/maxime/parquet_files_decoder/'
var_dir = 'parquets_variations/'
test_file = 'training-run2-test91-20251106-0917.parquet'

# Get variation game_ids
var_df = pd.read_parquet(os.path.join(var_dir, test_file), columns=['game_id'])
var_games = set(var_df['game_id'].unique())

# Load full pretrain file
print('Loading pretrain file...', flush=True)
pt_df = pd.read_parquet(os.path.join(pt_dir, test_file),
                        columns=['fen', 'best_move', 'game_id', 'ply', 'played_move'])

# Classify excluded games as standard vs non-standard
excluded_games = [g for g in pt_df['game_id'].unique() if g not in var_games]
print(f'Excluded games: {len(excluded_games)}', flush=True)
print('Classifying excluded games (standard vs 960)...', flush=True)

standard_games = []
nonstandard_games = []
for i, game_id in enumerate(excluded_games):
    game = pt_df[pt_df['game_id'] == game_id].sort_values('ply')
    origin_fen = game.iloc[0]['fen']
    moves = list(game['played_move'])
    if _is_standard_game(origin_fen, moves):
        standard_games.append(game_id)
    else:
        nonstandard_games.append(game_id)
    if (i+1) % 2000 == 0:
        print(f'  {i+1}/{len(excluded_games)}...', flush=True)

print(f'\nExcluded standard games:     {len(standard_games)}', flush=True)
print(f'Excluded non-standard games: {len(nonstandard_games)}', flush=True)
print(flush=True)

# Sample positions from each group
def sample_positions(game_ids, df, seed, n):
    sub = df[df['game_id'].isin(set(game_ids))]
    sampled = sample_one_per_game(sub[['fen','best_move','game_id']], seed)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    seen = set()
    pairs = []
    for _, r in sampled.iterrows():
        if r['fen'] not in seen:
            seen.add(r['fen'])
            pairs.append({'fen': r['fen'], 'best_move': norm(r['best_move'])})
        if len(pairs) >= n:
            break
    return pairs

n = 1000
pairs_std = sample_positions(standard_games, pt_df, seed, n)
pairs_960 = sample_positions(nonstandard_games, pt_df, seed, n)
pairs_var = sample_positions(list(var_games), pt_df, seed, n)
print(f'Sampled: {len(pairs_std)} standard-excluded, {len(pairs_960)} non-standard, {len(pairs_var)} variation-games', flush=True)

# Load engine
print('\nLoading engine...', flush=True)
export_dir = Path('export_eval_end')
engine = cpp.ThinkingInferenceEngine(
    str(export_dir/'backbone.pt'), str(export_dir/'weights'),
    str(export_dir/'vocab.json'), str(export_dir/'config.json'))

def evaluate(engine, pairs, label):
    correct = 0
    t0 = time.time()
    for i, p in enumerate(pairs):
        m = engine.predict_move(p['fen'], 0.0)
        if m and norm(m) == p['best_move']: correct += 1
        if (i+1) % 200 == 0:
            print(f'  [{label}] {i+1}/{len(pairs)} acc={correct/(i+1):.3f}', flush=True)
    elapsed = time.time() - t0
    acc = correct/len(pairs) if pairs else 0
    print(f'  {label}: {correct}/{len(pairs)} = {acc:.3f}  [{elapsed:.0f}s]\n', flush=True)
    return acc

acc_var = evaluate(engine, pairs_var, 'variation games (in-set)')
acc_std = evaluate(engine, pairs_std, 'excluded standard')
acc_960 = evaluate(engine, pairs_960, 'excluded non-standard')

print('='*60, flush=True)
print(f'Variation games (in-set):    {acc_var:.3f}  (n={len(pairs_var)})', flush=True)
print(f'Excluded standard games:     {acc_std:.3f}  (n={len(pairs_std)})', flush=True)
print(f'Excluded non-standard games: {acc_960:.3f}  (n={len(pairs_960)})', flush=True)
print(f'\nvar - excluded_std gap:       {acc_var - acc_std:+.3f}', flush=True)
print(f'var - excluded_960 gap:       {acc_var - acc_960:+.3f}', flush=True)
print(f'excluded_std - excluded_960:  {acc_std - acc_960:+.3f}', flush=True)
