"""
Debug the var_best vs pt_best gap by testing on the SAME file
with different loading strategies.
"""
import os, random, time, sys, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import _decoder_inference_cpp as cpp

_PS = {'e1h1':'e1g1','e1a1':'e1c1','e8h8':'e8g8','e8a8':'e8c8'}
def norm(m): return _PS.get(m, m) if isinstance(m, str) else m

def sample_one_per_game(df, seed):
    def _pick(g):
        gs = hash((seed, g.name)) % (2**31)
        return g.sample(1, random_state=gs)
    return df.groupby('game_id', group_keys=False).apply(_pick).reset_index(drop=True)

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

def load_and_dedup(df, seed, n):
    sampled = sample_one_per_game(df, seed)
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

seed = 42
n = 1000
pt_dir = '/home/maxime/parquet_files_decoder/'
var_dir = 'parquets_variations/'

# Pick a file that exists in BOTH directories
var_basenames = set(os.path.basename(f) for f in glob.glob(os.path.join(var_dir, '*.parquet')))
common = sorted(f for f in os.listdir(pt_dir) if f.endswith('.parquet') and f in var_basenames)
test_file = common[0]
print(f'Test file: {test_file}\n', flush=True)

# Strategy A: Load from PRETRAIN parquet (all rows, ~118/game)
print('Loading strategy A: pretrain parquet (all rows)...', flush=True)
pt_df = pd.read_parquet(os.path.join(pt_dir, test_file), columns=['fen','best_move','game_id','ply'])
print(f'  Rows: {len(pt_df)}, Games: {pt_df.game_id.nunique()}, Rows/game: {len(pt_df)/pt_df.game_id.nunique():.1f}', flush=True)
pairs_a = load_and_dedup(pt_df[['fen','best_move','game_id']], seed, n)
print(f'  Sampled: {len(pairs_a)} positions', flush=True)

# Ply distribution for strategy A
pt_sampled = sample_one_per_game(pt_df, seed)
ply_a = pt_sampled['ply'].describe()
print(f'  Ply stats: mean={ply_a["mean"]:.1f}, median={ply_a["50%"]:.1f}, std={ply_a["std"]:.1f}', flush=True)
print(flush=True)

# Strategy B: Load from VARIATION parquet (8 rows/game)
print('Loading strategy B: variation parquet (8 rows/game)...', flush=True)
var_df = pd.read_parquet(os.path.join(var_dir, test_file), columns=['fen','best_move','game_id','ply','mcts_action'])
var_df = var_df[var_df['mcts_action'].notna() & (var_df['mcts_action'] != '')]
print(f'  Rows: {len(var_df)}, Games: {var_df.game_id.nunique()}, Rows/game: {len(var_df)/var_df.game_id.nunique():.1f}', flush=True)
pairs_b = load_and_dedup(var_df[['fen','best_move','game_id']], seed, n)
print(f'  Sampled: {len(pairs_b)} positions', flush=True)

# Ply distribution for strategy B
var_sampled = sample_one_per_game(var_df[['fen','best_move','game_id','ply']], seed)
ply_b = var_sampled['ply'].describe()
print(f'  Ply stats: mean={ply_b["mean"]:.1f}, median={ply_b["50%"]:.1f}, std={ply_b["std"]:.1f}', flush=True)
print(flush=True)

# Strategy C: Load from PRETRAIN parquet but subsample to 8 rows/game FIRST
print('Loading strategy C: pretrain parquet, subsampled to 8/game...', flush=True)
rng = random.Random(seed + 999)  # different seed for subsampling
def subsample_8(g):
    if len(g) <= 8:
        return g
    return g.sample(8, random_state=rng.randint(0, 2**31))
pt_sub = pt_df.groupby('game_id', group_keys=False).apply(subsample_8).reset_index(drop=True)
print(f'  Rows: {len(pt_sub)}, Games: {pt_sub.game_id.nunique()}, Rows/game: {len(pt_sub)/pt_sub.game_id.nunique():.1f}', flush=True)
pairs_c = load_and_dedup(pt_sub[['fen','best_move','game_id']], seed, n)
print(f'  Sampled: {len(pairs_c)} positions', flush=True)

pt_sub_sampled = sample_one_per_game(pt_sub[['fen','best_move','game_id','ply']], seed)
ply_c = pt_sub_sampled['ply'].describe()
print(f'  Ply stats: mean={ply_c["mean"]:.1f}, median={ply_c["50%"]:.1f}, std={ply_c["std"]:.1f}', flush=True)
print(flush=True)

# Check FEN overlap between A and B
fens_a = set(p['fen'] for p in pairs_a)
fens_b = set(p['fen'] for p in pairs_b)
print(f'FEN overlap between A and B: {len(fens_a & fens_b)}', flush=True)

# Check best_move consistency for shared FENs
if fens_a & fens_b:
    bm_a = {p['fen']: p['best_move'] for p in pairs_a}
    bm_b = {p['fen']: p['best_move'] for p in pairs_b}
    mismatches = sum(1 for f in fens_a & fens_b if bm_a[f] != bm_b[f])
    print(f'best_move mismatches on shared FENs: {mismatches}/{len(fens_a & fens_b)}', flush=True)
print(flush=True)

# Check game_id overlap
games_a = set(pt_df['game_id'].unique())
games_b = set(var_df['game_id'].unique())
print(f'Games in pretrain: {len(games_a)}', flush=True)
print(f'Games in variation: {len(games_b)}', flush=True)
print(f'Game overlap: {len(games_a & games_b)}', flush=True)
print(f'Games only in pretrain (filtered by _is_standard_game): {len(games_a - games_b)}', flush=True)
print(flush=True)

# Load engine
print('Loading engine...', flush=True)
export_dir = Path('export_eval_end')
engine = cpp.ThinkingInferenceEngine(
    str(export_dir/'backbone.pt'), str(export_dir/'weights'),
    str(export_dir/'vocab.json'), str(export_dir/'config.json'))
print(flush=True)

# Evaluate all three strategies
acc_a = evaluate(engine, pairs_a, 'A: pretrain all-rows')
acc_b = evaluate(engine, pairs_b, 'B: variation 8/game')
acc_c = evaluate(engine, pairs_c, 'C: pretrain subsampled 8/game')

print('='*60, flush=True)
print(f'A (pretrain, all rows):      {acc_a:.3f}', flush=True)
print(f'B (variation, 8/game):       {acc_b:.3f}', flush=True)
print(f'C (pretrain, sub 8/game):    {acc_c:.3f}', flush=True)
print(f'B - A gap:                   {acc_b - acc_a:+.3f}', flush=True)
print(f'C - A gap:                   {acc_c - acc_a:+.3f}', flush=True)
print(f'B - C gap:                   {acc_b - acc_c:+.3f}', flush=True)
