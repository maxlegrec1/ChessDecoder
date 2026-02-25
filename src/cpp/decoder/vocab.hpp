#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace decoder
{

// Vocabulary constants matching src/models/vocab.py
constexpr int kBoardVocabSize = 41;
constexpr int kMoveVocabSize = 1924;
constexpr int kPositionTokenLength = 68;
constexpr int kNumBuckets = 100;

class DecoderVocab
{
public:
    /// Load vocabulary from vocab.json exported by export_onnx.py.
    explicit DecoderVocab(const std::string& vocab_json_path);

    // Full vocabulary lookups
    int tokenToIdx(const std::string& token) const;
    const std::string& idxToToken(int idx) const;
    int vocabSize() const { return static_cast<int>(vocab_.size()); }

    // Sub-vocabulary mappings
    int boardIdxToFullIdx(int board_idx) const { return boardIdxToFullIdx_[board_idx]; }
    int moveIdxToFullIdx(int move_idx) const { return moveIdxToFullIdx_[move_idx]; }
    int boardTokenToIdx(const std::string& token) const;
    int moveTokenToIdx(const std::string& token) const;
    std::optional<int> fullIdxToMoveIdx(int full_idx) const;

    // Special token indices (full vocab)
    int startPosIdx() const { return startPosIdx_; }
    int endPosIdx() const { return endPosIdx_; }
    int whiteToMoveIdx() const { return whiteToMoveIdx_; }
    int blackToMoveIdx() const { return blackToMoveIdx_; }
    int emptyIdx() const { return emptyIdx_; }
    int wlValueIdx() const { return wlValueIdx_; }
    int dValueIdx() const { return dValueIdx_; }
    int startThinkIdx() const { return startThinkIdx_; }
    int endThinkIdx() const { return endThinkIdx_; }
    int endVarIdx() const { return endVarIdx_; }

    // Board sub-vocab indices for structural tokens
    int boardEndVarIdx() const { return boardEndVarIdx_; }
    int boardEndThinkIdx() const { return boardEndThinkIdx_; }

    /// Convert FEN string to token ID sequence (68 tokens).
    std::vector<int> fenToTokenIds(const std::string& fen) const;

    /// Convert standard UCI to pseudo-UCI for vocabulary lookup.
    static std::string standardToPseudoUci(const std::string& uci);

    /// Convert pseudo-UCI (from vocabulary) to standard UCI.
    static std::string pseudoToStandardUci(const std::string& uci);

    /// Get legal move indices (in move sub-vocab) for a FEN position.
    std::vector<int> legalMoveIndices(const std::string& fen) const;

    /// Check if a token string looks like a move (4-5 chars, square-square format).
    static bool isMoveToken(const std::string& token);

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> tokenToIdx_;
    std::vector<int> boardIdxToFullIdx_;
    std::vector<int> moveIdxToFullIdx_;
    std::unordered_map<std::string, int> boardTokenToIdx_;
    std::unordered_map<std::string, int> moveTokenToIdx_;
    std::unordered_map<int, int> fullIdxToMoveIdx_;

    // Piece tokens
    std::unordered_map<std::string, std::string> pieceSymbolToToken_;

    // Castling tokens
    std::unordered_map<std::string, std::string> castlingToToken_;

    // Special token indices (full vocab)
    int startPosIdx_{};
    int endPosIdx_{};
    int whiteToMoveIdx_{};
    int blackToMoveIdx_{};
    int emptyIdx_{};
    int wlValueIdx_{};
    int dValueIdx_{};
    int startThinkIdx_{};
    int endThinkIdx_{};
    int endVarIdx_{};

    // Board sub-vocab indices
    int boardEndVarIdx_{};
    int boardEndThinkIdx_{};
};

} // namespace decoder
