#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace chessrl::mcts
{

struct RolloutDetail
{
    std::string root_move;
    int visit_count{0};
    float visit_fraction{0.0F};
    float prior{0.0F};
    float average_value{0.0F};
    std::vector<std::string> continuation;
};

struct MctsSummary
{
    std::string action;
    std::vector<std::pair<std::string, float>> policy;
    std::array<float, 3> value{{0.0F, 0.0F, 0.0F}};
    std::vector<RolloutDetail> rollouts;
};

} // namespace chessrl::mcts


