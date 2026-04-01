"""Calculate Model win rate from a PGN file (draws = 0.5)."""
import re
import sys

def calc_winrate(pgn_path: str):
    text = open(pgn_path).read()
    games = re.split(r'\n(?=\[Event )', text.strip())

    score, total = 0.0, 0
    for game in games:
        white = re.search(r'\[White "(.+?)"\]', game)
        black = re.search(r'\[Black "(.+?)"\]', game)
        result = re.search(r'\[Result "(.+?)"\]', game)
        if not (white and black and result):
            continue
        r = result.group(1)
        if r == "*":
            continue
        total += 1
        model_is_white = white.group(1) == "Model"
        if r == "1-0" and model_is_white:
            score += 1
        elif r == "0-1" and not model_is_white:
            score += 1
        elif r == "1/2-1/2":
            score += 0.5

    print(f"{pgn_path}: {score}/{total}  ({score/total*100:.1f}%)" if total else "No games found")

if __name__ == "__main__":
    for path in sys.argv[1:]:
        calc_winrate(path)
