#pragma once

#include "common.hpp"
#include "summary.hpp"

namespace chessrl::mcts
{

MctsSummary run_adversarial_mcts(MctsOptions options);

} // namespace chessrl::mcts


