#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "decoder_engine.hpp"
#include "batched_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_decoder_inference_cpp, m)
{
    m.doc() = "C++/libtorch thinking inference engine for ChessDecoder";

    py::class_<decoder::ThinkingInferenceEngine>(m, "ThinkingInferenceEngine")
        .def(py::init<const std::string&, const std::string&,
                       const std::string&, const std::string&>(),
             py::arg("backbone_pt_path"),
             py::arg("weights_dir"),
             py::arg("vocab_path"),
             py::arg("config_path"),
             "Construct engine from TorchScript backbone + weights.")
        // NOTE: Engine is NOT thread-safe. gil_scoped_release allows Python
        // GC/signals during long calls, but do not call from multiple threads.
        .def("predict_move", &decoder::ThinkingInferenceEngine::predictMove,
             py::arg("fen"),
             py::arg("temperature") = 0.0f,
             py::call_guard<py::gil_scoped_release>(),
             "Predict the best move for a FEN position using thinking inference.\n"
             "WARNING: Not thread-safe. Do not call from multiple threads.")
        .def("predict_move_root", &decoder::ThinkingInferenceEngine::predictMoveRoot,
             py::arg("fen"),
             py::arg("temperature") = 0.0f,
             py::call_guard<py::gil_scoped_release>(),
             "Predict move using only root board (no thinking). Single forward pass.")
        .def("last_token_ids", &decoder::ThinkingInferenceEngine::lastTokenIds,
             py::return_value_policy::reference_internal,
             "Get token IDs from the last predict_move() call.")
        .def("last_wl_entries", &decoder::ThinkingInferenceEngine::lastWlEntries,
             py::return_value_policy::reference_internal,
             "Get WL entries (position, value) from the last predict_move() call.")
        .def("last_d_entries", &decoder::ThinkingInferenceEngine::lastDEntries,
             py::return_value_policy::reference_internal,
             "Get D entries (position, value) from the last predict_move() call.")
        .def("idx_to_token", &decoder::ThinkingInferenceEngine::idxToToken,
             py::arg("idx"),
             py::return_value_policy::reference_internal,
             "Convert a token ID to its string name.")
        .def_readwrite("board_temperature", &decoder::ThinkingInferenceEngine::board_temperature)
        .def_readwrite("think_temperature", &decoder::ThinkingInferenceEngine::think_temperature)
        .def_readwrite("policy_temperature", &decoder::ThinkingInferenceEngine::policy_temperature)
        .def_readwrite("wl_temperature", &decoder::ThinkingInferenceEngine::wl_temperature)
        .def_readwrite("d_temperature", &decoder::ThinkingInferenceEngine::d_temperature)
        .def_readwrite("total_tokens", &decoder::ThinkingInferenceEngine::total_tokens)
        .def_readwrite("total_time", &decoder::ThinkingInferenceEngine::total_time)
        .def_readwrite("profiling", &decoder::ThinkingInferenceEngine::profiling)
        .def("reset_profile", &decoder::ThinkingInferenceEngine::resetProfile)
        .def_readwrite("prof_prefix_init", &decoder::ThinkingInferenceEngine::prof_prefix_init)
        .def_readwrite("prof_board_prefill", &decoder::ThinkingInferenceEngine::prof_board_prefill)
        .def_readwrite("prof_board_catchup", &decoder::ThinkingInferenceEngine::prof_board_catchup)
        .def_readwrite("prof_board_gen", &decoder::ThinkingInferenceEngine::prof_board_gen)
        .def_readwrite("prof_prefix_block", &decoder::ThinkingInferenceEngine::prof_prefix_block)
        .def_readwrite("prof_prefix_incr", &decoder::ThinkingInferenceEngine::prof_prefix_incr)
        .def_readwrite("prof_causal_incr", &decoder::ThinkingInferenceEngine::prof_causal_incr)
        .def_readwrite("prof_head_eval", &decoder::ThinkingInferenceEngine::prof_head_eval)
        .def_readwrite("prof_sync_ops", &decoder::ThinkingInferenceEngine::prof_sync_ops);

    // ── Batched inference engine ──────────────────────────────────────
    py::class_<decoder::BatchedInferenceEngine::Result>(m, "BatchedResult")
        .def_readonly("move", &decoder::BatchedInferenceEngine::Result::move)
        .def_readonly("token_ids", &decoder::BatchedInferenceEngine::Result::token_ids)
        .def_readonly("wl_entries", &decoder::BatchedInferenceEngine::Result::wl_entries)
        .def_readonly("d_entries", &decoder::BatchedInferenceEngine::Result::d_entries)
        .def_readonly("move_log_probs",
                      &decoder::BatchedInferenceEngine::Result::move_log_probs);

    py::class_<decoder::BatchedInferenceEngine>(m, "BatchedInferenceEngine")
        .def(py::init<const std::string&, const std::string&,
                       const std::string&, const std::string&, int>(),
             py::arg("backbone_pt_path"),
             py::arg("weights_dir"),
             py::arg("vocab_path"),
             py::arg("config_path"),
             py::arg("max_batch_size"),
             "Construct batched engine for parallel rollout generation.")
        .def("predict_moves", &decoder::BatchedInferenceEngine::predictMoves,
             py::arg("fens"),
             py::arg("temperature") = 0.0f,
             py::call_guard<py::gil_scoped_release>(),
             "Predict moves for a batch of FEN positions using thinking inference.")
        .def_readwrite("board_temperature", &decoder::BatchedInferenceEngine::board_temperature)
        .def_readwrite("think_temperature", &decoder::BatchedInferenceEngine::think_temperature)
        .def_readwrite("policy_temperature", &decoder::BatchedInferenceEngine::policy_temperature)
        .def_readwrite("wl_temperature", &decoder::BatchedInferenceEngine::wl_temperature)
        .def_readwrite("d_temperature", &decoder::BatchedInferenceEngine::d_temperature)
        .def_readwrite("total_tokens", &decoder::BatchedInferenceEngine::total_tokens)
        .def_readwrite("total_time", &decoder::BatchedInferenceEngine::total_time);
}
