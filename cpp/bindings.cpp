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

