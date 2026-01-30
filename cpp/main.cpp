#include <algorithm>
#include <iostream>
#include <string_view>
#include <vector>

#include "mcts/adversarial_runner.hpp"
#include "mcts/leela_runner.hpp"
#include "mcts/small_runner.hpp"
#include "mcts/summary.hpp"

namespace
{

void print_summary(std::string_view label, const chessrl::mcts::MctsSummary& summary)
{
    std::cout << label << " move: " << (summary.action.empty() ? "<none>" : summary.action) << '\n';

    if (!summary.policy.empty())
    {
        std::cout << label << " policy:" << '\n';
        for (const auto& [move, prob] : summary.policy)
        {
            std::cout << "  " << move << ": " << prob << '\n';
        }
    }

    if (!summary.rollouts.empty())
    {
        std::cout << label << " visits:" << '\n';
        for (const auto& detail : summary.rollouts)
        {
            std::cout << "  " << detail.root_move << ": " << detail.visit_count << '\n';
        }
    }
}

} // namespace

int main()
{
    try
    {
        chessrl::mcts::MctsOptions small_options;
        small_options.request.fen = "r1bqkb1r/p4ppp/2p2n2/nB2p1N1/8/5Q2/PPPP1PPP/RNB1K2R b KQkq - 1 8";
        small_options.simulations = 600;
        small_options.cpuct = 1.5F;
        auto small_summary = chessrl::mcts::run_small_mcts(small_options);
        print_summary("[small]", small_summary);

        chessrl::mcts::MctsOptions leela_options;
        leela_options.request.fen = "";
        leela_options.request.history = {"e2e4", "e7e5", "g1f3", "b8c6", "f1c4","g8f6","f3g5","d7d5","e4d5","c6a5","c4b5","c7c6","d5c6","b7c6","d1f3"};
        leela_options.simulations = 600;
        leela_options.cpuct = 1.5F;
        auto leela_summary = chessrl::mcts::run_leela_mcts(leela_options);
        print_summary("[leela]", leela_summary);

        chessrl::mcts::MctsOptions adv_options;
        adv_options.request.fen = "";
        adv_options.request.history = {"e2e4", "e7e5", "g1f3", "b8c6", "f1c4","g8f6","f3g5","d7d5","e4d5","c6a5","c4b5","c7c6","d5c6","b7c6","d1f3"};
        adv_options.simulations = 600;
        adv_options.cpuct = 1.5F;
        auto adv_summary = chessrl::mcts::run_adversarial_mcts(adv_options);
        print_summary("[adversarial]", adv_summary);

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
