#pragma once

#include <string>
#include <vector>

#include "common.hpp"
#include "summary.hpp"

namespace chessrl::mcts
{

MctsSummary run_leela_mcts(MctsOptions options);

std::vector<MctsSummary> run_parallel_leela_mcts(
    std::vector<MctsOptions> requests,
    const std::string& engine_path = "model_dynamic_leela.trt",
    int max_batch_size = 256);

} // namespace chessrl::mcts


