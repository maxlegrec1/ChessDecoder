#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "inference.hpp"
#include "mcts/adversarial_runner.hpp"
#include "mcts/leela_runner.hpp"
#include "mcts/small_runner.hpp"
#include "mcts/summary.hpp"

namespace py = pybind11;

namespace
{

py::dict summary_to_python(const chessrl::mcts::MctsSummary& summary)
{
    py::dict result;
    result["action"] = summary.action;
    result["policy"] = summary.policy;
    result["value"] = summary.value;

    py::dict q_values;
    for (const auto& detail : summary.rollouts)
    {
        q_values[py::str(detail.root_move)] = detail.average_value;
    }
    result["q_values"] = std::move(q_values);

    if (!summary.variations.empty())
    {
        py::list var_list;
        for (const auto& var : summary.variations)
        {
            py::dict var_dict;
            var_dict["root_move"] = var.root_move;
            var_dict["visit_count"] = var.visit_count;
            var_dict["visit_fraction"] = var.visit_fraction;
            var_dict["prior"] = var.prior;
            py::list nodes_list;
            for (const auto& node : var.nodes)
            {
                py::dict node_dict;
                node_dict["fen"] = node.fen;
                node_dict["move"] = node.move;
                node_dict["wdl"] = node.wdl;
                node_dict["visit_count"] = node.visit_count;
                nodes_list.append(std::move(node_dict));
            }
            var_dict["nodes"] = std::move(nodes_list);
            var_list.append(std::move(var_dict));
        }
        result["variations"] = std::move(var_list);
    }

    return result;
}

} // namespace

PYBIND11_MODULE(_inference_cpp, m)
{
    py::class_<chessrl::InferenceTimings>(m, "InferenceTimings")
        .def_readonly("encode_ms", &chessrl::InferenceTimings::encodeMs)
        .def_readonly("inference_ms", &chessrl::InferenceTimings::inferMs)
        .def_readonly("sampling_ms", &chessrl::InferenceTimings::sampleMs)
        .def_readonly("total_ms", &chessrl::InferenceTimings::totalMs);

    py::class_<chessrl::SmallInferenceResult>(m, "SmallInferenceResult")
        .def_property_readonly("moves", [](const chessrl::SmallInferenceResult& self) { return self.moves; })
        .def_property_readonly("wdl", [](const chessrl::SmallInferenceResult& self) { return self.wdl; })
        .def_property_readonly("timings", [](const chessrl::SmallInferenceResult& self) { return self.timings; });

    py::class_<chessrl::LeelaInferenceResult>(m, "LeelaInferenceResult")
        .def_property_readonly("moves", [](const chessrl::LeelaInferenceResult& self) { return self.moves; })
        .def_property_readonly(
            "value_winner", [](const chessrl::LeelaInferenceResult& self) { return self.value_winner; })
        .def_property_readonly("value_q", [](const chessrl::LeelaInferenceResult& self) { return self.value_q; })
        .def_property_readonly("timings", [](const chessrl::LeelaInferenceResult& self) { return self.timings; });

    m.def(
        "small_generate_moves",
        [](const std::vector<std::string>& fens, float temperature, int iterations, const std::string& engine_path) {
            return chessrl::small_generate_moves(fens, temperature, iterations, engine_path);
        },
        py::arg("fens"),
        py::arg("temperature"),
        py::arg("iterations") = 1,
        py::arg("engine_path") = std::string("model.trt"),
        "Run the TensorRT policy head and return sampled moves with timing stats.");

    m.def(
        "leela_generate_from_move_lists",
        [](const std::vector<std::string>& origin_fens,
           const std::vector<std::vector<std::string>>& move_lists,
           float temperature,
           int iterations,
           const std::string& engine_path) {
            return chessrl::leela_generate_from_move_lists(origin_fens, move_lists, temperature, iterations, engine_path);
        },
        py::arg("origin_fens"),
        py::arg("move_lists"),
        py::arg("temperature"),
        py::arg("iterations") = 1,
        py::arg("engine_path") = std::string("leela.trt"),
        "Run Leela TensorRT inference on batched move histories and return selected moves and value heads.");

    m.def(
        "small_mcts_search",
        [](const std::string& fen,
           const std::vector<std::string>& history,
           int simulations,
           float cpuct,
           float temperature,
           const std::string& engine_path) {
            chessrl::mcts::MctsOptions options;
            options.request.fen = fen;
            options.request.history = history;
            options.simulations = simulations;
            options.cpuct = cpuct;
            options.temperature = temperature;
            options.small_engine_path = engine_path;
            auto summary = chessrl::mcts::run_small_mcts(std::move(options));
            return summary_to_python(summary);
        },
        py::arg("fen"),
        py::arg("history") = std::vector<std::string>{},
        py::arg("simulations") = 800,
        py::arg("cpuct") = 1.5F,
        py::arg("temperature") = 1.0F,
        py::arg("engine_path") = std::string("model_minibatch.trt"),
        "Execute the small-model MCTS search and return action, policy distribution, and Q-values.");

    m.def(
        "leela_mcts_search",
        [](const std::string& fen,
           const std::vector<std::string>& history,
           int simulations,
           float cpuct,
           float temperature,
           const std::string& engine_path) {
            chessrl::mcts::MctsOptions options;
            options.request.fen = fen;
            options.request.history = history;
            options.simulations = simulations;
            options.cpuct = cpuct;
            options.temperature = temperature;
            options.leela_engine_path = engine_path;
            auto summary = chessrl::mcts::run_leela_mcts(std::move(options));
            return summary_to_python(summary);
        },
        py::arg("fen"),
        py::arg("history") = std::vector<std::string>{},
        py::arg("simulations") = 800,
        py::arg("cpuct") = 1.5F,
        py::arg("temperature") = 1.0F,
        py::arg("engine_path") = std::string("leela_minibatch.trt"),
        "Execute the Leela-style MCTS search and return action, policy distribution, and Q-values.");

    m.def(
        "leela_mcts_search_with_variations",
        [](const std::string& fen,
           const std::vector<std::string>& history,
           int simulations,
           float cpuct,
           float temperature,
           const std::string& engine_path,
           int max_variations,
           int max_variation_depth) {
            chessrl::mcts::MctsOptions options;
            options.request.fen = fen;
            options.request.history = history;
            options.simulations = simulations;
            options.cpuct = cpuct;
            options.temperature = temperature;
            options.leela_engine_path = engine_path;
            options.extract_variations = true;
            options.max_variations = max_variations;
            options.max_variation_depth = max_variation_depth;
            auto summary = chessrl::mcts::run_leela_mcts(std::move(options));
            return summary_to_python(summary);
        },
        py::arg("fen"),
        py::arg("history") = std::vector<std::string>{},
        py::arg("simulations") = 800,
        py::arg("cpuct") = 1.5F,
        py::arg("temperature") = 1.0F,
        py::arg("engine_path") = std::string("leela_minibatch.trt"),
        py::arg("max_variations") = 5,
        py::arg("max_variation_depth") = 20,
        "Execute the Leela-style MCTS search with PV variation extraction and return action, policy, Q-values, and variations.");

    m.def(
        "leela_mcts_search_parallel",
        [](const std::vector<std::string>& fens,
           const std::vector<std::vector<std::string>>& histories,
           int simulations, float cpuct, float temperature,
           const std::string& engine_path, int max_batch_size,
           int max_variations, int max_variation_depth) {

            // Build options vector
            std::vector<chessrl::mcts::MctsOptions> requests;
            requests.reserve(fens.size());
            for (size_t i = 0; i < fens.size(); ++i)
            {
                chessrl::mcts::MctsOptions opts;
                opts.request.fen = fens[i];
                opts.request.history = histories[i];
                opts.simulations = simulations;
                opts.cpuct = cpuct;
                opts.temperature = temperature;
                opts.leela_engine_path = engine_path;
                opts.extract_variations = (max_variations > 0);
                opts.max_variations = max_variations;
                opts.max_variation_depth = max_variation_depth;
                requests.push_back(std::move(opts));
            }

            // Release GIL during C++ parallel computation
            std::vector<chessrl::mcts::MctsSummary> summaries;
            {
                py::gil_scoped_release release;
                summaries = chessrl::mcts::run_parallel_leela_mcts(
                    std::move(requests), engine_path, max_batch_size);
            }

            py::list results;
            for (auto& s : summaries)
            {
                results.append(summary_to_python(s));
            }
            return results;
        },
        py::arg("fens"),
        py::arg("histories"),
        py::arg("simulations") = 600,
        py::arg("cpuct") = 1.5F,
        py::arg("temperature") = 1.0F,
        py::arg("engine_path") = std::string("model_dynamic_leela.trt"),
        py::arg("max_batch_size") = 256,
        py::arg("max_variations") = 5,
        py::arg("max_variation_depth") = 20,
        "Execute parallel Leela MCTS searches sharing a single batch runner.");

    m.def(
        "adversarial_mcts_search",
        [](const std::string& fen,
           const std::vector<std::string>& history,
           int simulations,
           float cpuct,
           float temperature,
           const std::string& attacker_engine_path,
           const std::string& defender_engine_path) {
            chessrl::mcts::MctsOptions options;
            options.request.fen = fen;
            options.request.history = history;
            options.simulations = simulations;
            options.cpuct = cpuct;
            options.temperature = temperature;
            options.small_engine_path = attacker_engine_path;
            options.leela_engine_path = defender_engine_path;
            auto summary = chessrl::mcts::run_adversarial_mcts(std::move(options));
            return summary_to_python(summary);
        },
        py::arg("fen"),
        py::arg("history") = std::vector<std::string>{},
        py::arg("simulations") = 800,
        py::arg("cpuct") = 1.5F,
        py::arg("temperature") = 1.0F,
        py::arg("attacker_engine_path") = std::string("model_minibatch.trt"),
        py::arg("defender_engine_path") = std::string("leela_minibatch.trt"),
        "Execute the adversarial MCTS search (attacker vs defender engines) and return action, policy distribution, and Q-values.");
}

