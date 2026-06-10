"""Minimal opening book for head-to-head matches.

A list of well-known opening lines, each as a list of UCI moves. We replay
each line from the starting position before letting the engines take over,
so two color-swapped games starting from each line give balanced color
exposure for both engines.
"""
from __future__ import annotations

# 4-8 ply openings spanning 1.e4 / 1.d4 / 1.c4 / 1.Nf3 main systems.
OPENING_LINES: list[list[str]] = [
    # ---- 1.e4 e5
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],            # Ruy Lopez
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],            # Italian
    ["e2e4", "e7e5", "g1f3", "g8f6"],                     # Petroff
    ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"],            # Scotch
    ["e2e4", "e7e5", "b1c3"],                             # Vienna
    ["e2e4", "e7e5", "f2f4"],                             # King's Gambit
    # ---- Sicilian (1.e4 c5)
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"],  # Open Sicilian -> Najdorf-shape
    ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4"],                  # Open Sicilian -> ...Nc6
    ["e2e4", "c7c5", "g1f3", "e7e6"],                     # Sicilian Kan-shape
    ["e2e4", "c7c5", "c2c3"],                             # Alapin
    ["e2e4", "c7c5", "g1f3", "d7d6", "f1b5"],            # Moscow Variation
    # ---- 1.e4 other
    ["e2e4", "e7e6"],                                      # French
    ["e2e4", "c7c6"],                                      # Caro-Kann
    ["e2e4", "d7d6"],                                      # Pirc/Modern
    ["e2e4", "g7g6"],                                      # Modern
    ["e2e4", "c7c5", "b1c3"],                             # Closed Sicilian
    # ---- 1.d4 d5
    ["d2d4", "d7d5", "c2c4", "e7e6"],                     # QGD
    ["d2d4", "d7d5", "c2c4", "c7c6"],                     # Slav
    ["d2d4", "d7d5", "c2c4", "d5c4"],                     # QGA
    # ---- 1.d4 Nf6
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],    # Nimzo-Indian
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"],    # QID
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"],    # Grünfeld
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"],    # KID setup
    ["d2d4", "g8f6", "c2c4", "c7c5"],                     # Benoni-shape
    ["d2d4", "f7f5"],                                      # Dutch
    # ---- Flank
    ["c2c4"],                                              # English
    ["g1f3", "g8f6"],                                      # Réti
    ["c2c4", "e7e5"],                                      # Reversed Sicilian
    ["g1f3", "d7d5", "g2g3"],                             # King's Indian Attack
]
