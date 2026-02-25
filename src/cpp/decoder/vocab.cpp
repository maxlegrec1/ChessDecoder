#include "vocab.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>

// nlohmann/json single-header — we include a minimal parser instead
// to avoid external dependency. We parse the vocab.json manually.
#include <algorithm>

#include "../chess-library/include/chess.hpp"

namespace decoder
{

namespace
{

// Minimal JSON array/object parser for vocab.json
// Supports: strings, ints, arrays of strings/ints, objects with string keys

std::string trim(const std::string& s)
{
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

// Parse a JSON string (assumes cursor is at opening quote)
std::string parseJsonString(const std::string& json, size_t& pos)
{
    if (json[pos] != '"')
        throw std::runtime_error("Expected '\"' at position " + std::to_string(pos));
    pos++; // skip opening quote
    std::string result;
    while (pos < json.size() && json[pos] != '"')
    {
        if (json[pos] == '\\')
        {
            pos++;
            if (pos < json.size()) result += json[pos];
        }
        else
        {
            result += json[pos];
        }
        pos++;
    }
    pos++; // skip closing quote
    return result;
}

// Parse a JSON integer
int parseJsonInt(const std::string& json, size_t& pos)
{
    size_t start = pos;
    if (json[pos] == '-') pos++;
    while (pos < json.size() && std::isdigit(json[pos])) pos++;
    return std::stoi(json.substr(start, pos - start));
}

void skipWhitespace(const std::string& json, size_t& pos)
{
    while (pos < json.size() && std::isspace(json[pos])) pos++;
}

// Parse a JSON array of strings
std::vector<std::string> parseStringArray(const std::string& json, size_t& pos)
{
    std::vector<std::string> result;
    skipWhitespace(json, pos);
    if (json[pos] != '[')
        throw std::runtime_error("Expected '[' at position " + std::to_string(pos));
    pos++;
    skipWhitespace(json, pos);
    while (pos < json.size() && json[pos] != ']')
    {
        skipWhitespace(json, pos);
        result.push_back(parseJsonString(json, pos));
        skipWhitespace(json, pos);
        if (json[pos] == ',') pos++;
    }
    pos++; // skip ']'
    return result;
}

// Parse a JSON array of ints
std::vector<int> parseIntArray(const std::string& json, size_t& pos)
{
    std::vector<int> result;
    skipWhitespace(json, pos);
    if (json[pos] != '[')
        throw std::runtime_error("Expected '[' at position " + std::to_string(pos));
    pos++;
    skipWhitespace(json, pos);
    while (pos < json.size() && json[pos] != ']')
    {
        skipWhitespace(json, pos);
        result.push_back(parseJsonInt(json, pos));
        skipWhitespace(json, pos);
        if (json[pos] == ',') pos++;
    }
    pos++; // skip ']'
    return result;
}

// Parse a JSON object with string keys and int values
std::unordered_map<std::string, int> parseStringIntMap(const std::string& json, size_t& pos)
{
    std::unordered_map<std::string, int> result;
    skipWhitespace(json, pos);
    if (json[pos] != '{')
        throw std::runtime_error("Expected '{' at position " + std::to_string(pos));
    pos++;
    skipWhitespace(json, pos);
    while (pos < json.size() && json[pos] != '}')
    {
        skipWhitespace(json, pos);
        std::string key = parseJsonString(json, pos);
        skipWhitespace(json, pos);
        if (json[pos] != ':')
            throw std::runtime_error("Expected ':' at position " + std::to_string(pos));
        pos++;
        skipWhitespace(json, pos);
        int val = parseJsonInt(json, pos);
        result[key] = val;
        skipWhitespace(json, pos);
        if (json[pos] == ',') pos++;
    }
    pos++; // skip '}'
    return result;
}

// Skip a JSON value (string, number, array, object)
void skipJsonValue(const std::string& json, size_t& pos)
{
    skipWhitespace(json, pos);
    if (json[pos] == '"')
    {
        parseJsonString(json, pos);
    }
    else if (json[pos] == '[')
    {
        int depth = 1;
        pos++;
        while (pos < json.size() && depth > 0)
        {
            if (json[pos] == '[') depth++;
            else if (json[pos] == ']') depth--;
            else if (json[pos] == '"') { parseJsonString(json, pos); continue; }
            pos++;
        }
    }
    else if (json[pos] == '{')
    {
        int depth = 1;
        pos++;
        while (pos < json.size() && depth > 0)
        {
            if (json[pos] == '{') depth++;
            else if (json[pos] == '}') depth--;
            else if (json[pos] == '"') { parseJsonString(json, pos); continue; }
            pos++;
        }
    }
    else
    {
        // number, bool, null
        while (pos < json.size() && json[pos] != ',' && json[pos] != '}' && json[pos] != ']')
            pos++;
    }
}

// Map piece char + color to our token name
std::string pieceToToken(char piece)
{
    static const std::unordered_map<char, std::string> map = {
        {'K', "white_king"}, {'Q', "white_queen"}, {'R', "white_rook"},
        {'B', "white_bishop"}, {'N', "white_knight"}, {'P', "white_pawn"},
        {'k', "black_king"}, {'q', "black_queen"}, {'r', "black_rook"},
        {'b', "black_bishop"}, {'n', "black_knight"}, {'p', "black_pawn"},
    };
    auto it = map.find(piece);
    if (it != map.end()) return it->second;
    return "empty";
}

} // anonymous namespace

DecoderVocab::DecoderVocab(const std::string& vocab_json_path)
{
    // Read entire file
    std::ifstream f(vocab_json_path);
    if (!f) throw std::runtime_error("Failed to open vocab file: " + vocab_json_path);
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Parse top-level object
    size_t pos = 0;
    skipWhitespace(json, pos);
    if (json[pos] != '{') throw std::runtime_error("Expected top-level JSON object");
    pos++;

    while (pos < json.size() && json[pos] != '}')
    {
        skipWhitespace(json, pos);
        std::string key = parseJsonString(json, pos);
        skipWhitespace(json, pos);
        if (json[pos] != ':') throw std::runtime_error("Expected ':'");
        pos++;
        skipWhitespace(json, pos);

        if (key == "vocab")
        {
            vocab_ = parseStringArray(json, pos);
        }
        else if (key == "token_to_idx")
        {
            tokenToIdx_ = parseStringIntMap(json, pos);
        }
        else if (key == "board_idx_to_full_idx")
        {
            boardIdxToFullIdx_ = parseIntArray(json, pos);
        }
        else if (key == "move_idx_to_full_idx")
        {
            moveIdxToFullIdx_ = parseIntArray(json, pos);
        }
        else if (key == "board_token_to_idx")
        {
            auto map = parseStringIntMap(json, pos);
            for (auto& [k, v] : map) boardTokenToIdx_[k] = v;
        }
        else if (key == "move_token_to_idx")
        {
            auto map = parseStringIntMap(json, pos);
            for (auto& [k, v] : map) moveTokenToIdx_[k] = v;
        }
        else
        {
            skipJsonValue(json, pos);
        }

        skipWhitespace(json, pos);
        if (json[pos] == ',') pos++;
    }

    // Build reverse mapping: full_idx -> move_idx
    for (int i = 0; i < static_cast<int>(moveIdxToFullIdx_.size()); i++)
    {
        fullIdxToMoveIdx_[moveIdxToFullIdx_[i]] = i;
    }

    // Cache special token indices
    startPosIdx_ = tokenToIdx("start_pos");
    endPosIdx_ = tokenToIdx("end_pos");
    whiteToMoveIdx_ = tokenToIdx("white_to_move");
    blackToMoveIdx_ = tokenToIdx("black_to_move");
    emptyIdx_ = tokenToIdx("empty");
    wlValueIdx_ = tokenToIdx("wl_value");
    dValueIdx_ = tokenToIdx("d_value");
    startThinkIdx_ = tokenToIdx("start_think");
    endThinkIdx_ = tokenToIdx("end_think");
    endVarIdx_ = tokenToIdx("end_var");

    boardEndVarIdx_ = boardTokenToIdx("end_var");
    boardEndThinkIdx_ = boardTokenToIdx("end_think");
}

int DecoderVocab::tokenToIdx(const std::string& token) const
{
    auto it = tokenToIdx_.find(token);
    if (it == tokenToIdx_.end())
        throw std::runtime_error("Token not found in vocabulary: " + token);
    return it->second;
}

const std::string& DecoderVocab::idxToToken(int idx) const
{
    if (idx < 0 || idx >= static_cast<int>(vocab_.size()))
        throw std::runtime_error("Index out of range: " + std::to_string(idx));
    return vocab_[idx];
}

int DecoderVocab::boardTokenToIdx(const std::string& token) const
{
    auto it = boardTokenToIdx_.find(token);
    if (it == boardTokenToIdx_.end())
        throw std::runtime_error("Board token not found: " + token);
    return it->second;
}

int DecoderVocab::moveTokenToIdx(const std::string& token) const
{
    auto it = moveTokenToIdx_.find(token);
    if (it == moveTokenToIdx_.end())
        throw std::runtime_error("Move token not found: " + token);
    return it->second;
}

std::optional<int> DecoderVocab::fullIdxToMoveIdx(int full_idx) const
{
    auto it = fullIdxToMoveIdx_.find(full_idx);
    if (it == fullIdxToMoveIdx_.end()) return std::nullopt;
    return it->second;
}

std::vector<int> DecoderVocab::fenToTokenIds(const std::string& fen) const
{
    // Parse FEN using chess-library
    chess::Board board;
    board.setFen(fen);

    std::vector<int> tokens;
    tokens.reserve(kPositionTokenLength);

    // start_pos
    tokens.push_back(startPosIdx_);

    // 64 board squares: a1, b1, c1, ..., h8
    // chess-library uses a1=0, b1=1, ..., h8=63 (same as python-chess)
    for (int sq = 0; sq < 64; sq++)
    {
        auto piece = board.at(chess::Square(sq));
        if (piece == chess::Piece::NONE)
        {
            tokens.push_back(emptyIdx_);
        }
        else
        {
            // Map piece to token
            static const std::array<const char*, 12> pieceTokens = {
                "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
                "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king",
            };
            // chess::Piece enum: WHITEPAWN=0, WHITEKNIGHT=1, ..., BLACKKING=11
            int idx = static_cast<int>(piece);
            tokens.push_back(tokenToIdx(pieceTokens[idx]));
        }
    }

    // end_pos
    tokens.push_back(endPosIdx_);

    // Castling rights
    auto cr = board.castlingRights();
    std::string castling;
    if (cr.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE)) castling += 'K';
    if (cr.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE)) castling += 'Q';
    if (cr.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE)) castling += 'k';
    if (cr.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE)) castling += 'q';

    if (castling.empty())
    {
        tokens.push_back(tokenToIdx("no_castling_rights"));
    }
    else
    {
        tokens.push_back(tokenToIdx(castling));
    }

    // Side to move
    if (board.sideToMove() == chess::Color::WHITE)
        tokens.push_back(whiteToMoveIdx_);
    else
        tokens.push_back(blackToMoveIdx_);

    return tokens;
}

std::string DecoderVocab::standardToPseudoUci(const std::string& uci)
{
    if (uci == "e1g1") return "e1h1";
    if (uci == "e1c1") return "e1a1";
    if (uci == "e8g8") return "e8h8";
    if (uci == "e8c8") return "e8a8";
    return uci;
}

std::string DecoderVocab::pseudoToStandardUci(const std::string& uci)
{
    if (uci == "e1h1") return "e1g1";
    if (uci == "e1a1") return "e1c1";
    if (uci == "e8h8") return "e8g8";
    if (uci == "e8a8") return "e8c8";
    return uci;
}

std::vector<int> DecoderVocab::legalMoveIndices(const std::string& fen) const
{
    chess::Board board;
    board.setFen(fen);

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    std::vector<int> indices;
    indices.reserve(moves.size());

    for (const auto& move : moves)
    {
        std::string uci = chess::uci::moveToUci(move, false);

        // moveToUci with chess960=false already converts castling to standard UCI (e1g1, etc.)
        // We need pseudo-UCI for our vocabulary (e1h1, etc.)
        if (move.typeOf() == chess::Move::CASTLING)
        {
            uci = standardToPseudoUci(uci);
        }

        auto it = moveTokenToIdx_.find(uci);
        if (it != moveTokenToIdx_.end())
        {
            indices.push_back(it->second);
        }
    }

    return indices;
}

bool DecoderVocab::isMoveToken(const std::string& token)
{
    if (token.size() < 4 || token.size() > 5) return false;
    if (token[0] < 'a' || token[0] > 'h') return false;
    if (token[1] < '1' || token[1] > '8') return false;
    if (token[2] < 'a' || token[2] > 'h') return false;
    if (token[3] < '1' || token[3] > '8') return false;
    if (token.size() == 5 && token[4] != 'q' && token[4] != 'r' && token[4] != 'b' && token[4] != 'n')
        return false;
    return true;
}

} // namespace decoder
