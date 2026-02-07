#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "../chess-library/tests/doctest/doctest.hpp"

#include <algorithm>

#include "inference.hpp"
#include "mcts/common.hpp"

using chessrl::mcts::MctsOptions;
using chessrl::mcts::SearchInput;

TEST_CASE("parse_history_argument splits and trims")
{
    const std::string raw = "e2e4, e7e5 ,g1f3";
    const auto result = chessrl::mcts::parse_history_argument(raw);
    REQUIRE(result.size() == 3);
    CHECK(result[0] == "e2e4");
    CHECK(result[1] == "e7e5");
    CHECK(result[2] == "g1f3");
}

TEST_CASE("apply_history reproduces expected board")
{
    SearchInput input;
    input.fen = std::string(chess::constants::STARTPOS);
    input.history = {"e2e4", "c7c5", "g1f3"};

    chess::Board board = chessrl::mcts::apply_history(input);
    CHECK(board.getFen() == "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 3");
}

TEST_CASE("parse_cli populates options")
{
    const char* argv[] = {
        "mcts",
        "--fen",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "--history",
        "e2e4,e7e5",
        "--simulations",
        "32",
        "--cpuct",
        "1.2",
        "--temperature",
        "0.5",
    };
    const int argc = static_cast<int>(std::size(argv));
    auto options = chessrl::mcts::parse_cli(argc, const_cast<char**>(argv));

    CHECK(options.request.fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    REQUIRE(options.request.history.size() == 2);
    CHECK(options.request.history[0] == "e2e4");
    CHECK(options.request.history[1] == "e7e5");
    CHECK(options.simulations == 32);
    CHECK(options.cpuct == doctest::Approx(1.2F));
    CHECK(options.temperature == doctest::Approx(0.5F));
    CHECK(!options.use_dirichlet);
}

TEST_CASE("policy vocabularies expose expected sizes")
{
    const auto& small_vocab = chessrl::small_policy_vocabulary();
    CHECK(small_vocab.size() == chessrl::small_policy_size());
    CHECK(std::find(small_vocab.begin(), small_vocab.end(), "e2e4") != small_vocab.end());

    const auto& leela_vocab = chessrl::leela_policy_vocabulary();
    CHECK(leela_vocab.size() == chessrl::leela_policy_size());
    CHECK(std::find(leela_vocab.begin(), leela_vocab.end(), "e1h1") != leela_vocab.end());
}


