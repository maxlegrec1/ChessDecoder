#pragma once

#include "mcts/common.hpp"
#include "mcts/summary.hpp"

namespace chessrl::mcts
{

MctsSummary run_leela_mcts(MctsOptions options);

} // namespace chessrl::mcts


