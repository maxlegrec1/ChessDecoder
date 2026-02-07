import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append('/mnt/2tb_2/decoder')

from src.models.model import ChessDecoder
from src.models.vocab import token_to_idx, idx_to_token

class TestPredictMove(unittest.TestCase):
    def setUp(self):
        self.vocab_size = len(token_to_idx)
        # Use small model for speed
        self.model = ChessDecoder(vocab_size=self.vocab_size, num_layers=1, num_heads=1, embed_dim=32)
        self.model.eval()

    def test_castling_correction(self):
        # Mock forward to return logits favoring 'e1h1'
        e1h1_idx = token_to_idx.get('e1h1')
        if e1h1_idx is None:
            print("e1h1 not in vocab, skipping castling test")
            return

        # Create logits where e1h1 is max
        def mock_forward(x):
            logits = torch.zeros(1, 1, self.vocab_size)
            logits[0, 0, e1h1_idx] = 100.0
            return logits, None
        
        self.model.forward = mock_forward
        
        # FEN doesn't matter much for this test if we don't force legal
        move = self.model.predict_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", temperature=0.0)
        self.assertEqual(move, 'e1g1')

    def test_legal_masking(self):
        # Setup a board where only a few moves are legal.
        # Start position.
        illegal_move = 'a1a8' # Rook can't move through pawns
        legal_move = 'e2e4'
        
        illegal_idx = token_to_idx.get(illegal_move)
        legal_idx = token_to_idx.get(legal_move)
        
        if illegal_idx is None or legal_idx is None:
            print("Moves not in vocab")
            return

        # Mock logits: Illegal > Legal > Others
        def mock_forward(x):
            logits = torch.zeros(1, 1, self.vocab_size)
            logits[0, 0, :] = -10.0
            logits[0, 0, illegal_idx] = 10.0
            logits[0, 0, legal_idx] = 5.0
            return logits, None
            
        self.model.forward = mock_forward
        
        # Without force_legal, should predict illegal_move
        move = self.model.predict_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", temperature=0.0, force_legal=False)
        self.assertEqual(move, illegal_move)
        
        # With force_legal, should predict legal_move
        move = self.model.predict_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", temperature=0.0, force_legal=True)
        self.assertEqual(move, legal_move)

if __name__ == '__main__':
    unittest.main()
