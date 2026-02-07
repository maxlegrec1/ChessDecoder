from src.mcts import LeelaMCTS
leela = LeelaMCTS(engine_path="leela_minibatch.trt", temperature=600, cpuct=1.0)
from tqdm import tqdm

# fen = "6k1/6p1/2p1r1q1/p2p4/1p4p1/1BP2bQ1/PP1B3R/6K1 w - - 6 36"
# history = ["h2f2","e6f6","d2c1","g6e4","c1d2","e4b1","f2f1","b1b2","g3e1"]

fen = ""
history = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5","g8f6","e1g1"]

# res = leela.run(fen,history, temperature=1, simulations=600, cpuct=1.0)
# print(res)
for i in tqdm(range(10)):
    res = leela.run_with_variations(fen,history, temperature=1, simulations=600, cpuct=1.0, max_variations=5, max_variation_depth=5)
    for variation in res['variations']:
        print(variation)