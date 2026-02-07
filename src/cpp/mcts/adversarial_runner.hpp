#pragma once

#include "mcts/common.hpp"
#include "mcts/summary.hpp"

namespace chessrl::mcts
{

MctsSummary run_adversarial_mcts(MctsOptions options);

} // namespace chessrl::mcts


