#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "decoder_engine.hpp"
#include "batched_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_decoder_inference_cpp, m)
{
    m.doc() = "C++/libtorch thinking inference engine for ChessDecoder";

    py::class_<decoder::ThinkingSingleInferenceEngine>(m, "ThinkingSingleInferenceEngine")
        .def(py::init<const std::string&, const std::string&,
                       const std::string&, const std::string&>(),
             py::arg("backbone_pt_path"),
             py::arg("weights_dir"),
             py::arg("vocab_path"),
             py::arg("config_path"),
             "Construct engine from TorchScript backbone + weights.")
        // NOTE: Engine is NOT thread-safe. gil_scoped_release allows Python
        // GC/signals during long calls, but do not call from multiple threads.
        .def("predict_move", &decoder::ThinkingSingleInferenceEngine::predictMove,
             py::arg("fen"),
             py::arg("temperature") = 0.0f,
             py::call_guard<py::gil_scoped_release>(),
             "Predict the best move for a FEN position using thinking inference.\n"
             "WARNING: Not thread-safe. Do not call from multiple threads.")
        .def("predict_move_root", &decoder::ThinkingSingleInferenceEngine::predictMoveRoot,
             py::arg("fen"),
             py::arg("temperature") = 0.0f,
             py::call_guard<py::gil_scoped_release>(),
             "Predict move using only root board (no thinking). Single forward pass.")
        .def("last_token_ids", &decoder::ThinkingSingleInferenceEngine::lastTokenIds,
             py::return_value_policy::reference_internal,
             "Get token IDs from the last predict_move() call.")
        .def("last_wl_entries", &decoder::ThinkingSingleInferenceEngine::lastWlEntries,
             py::return_value_policy::reference_internal,
             "Get WL entries (position, value) from the last predict_move() call.")
        .def("last_d_entries", &decoder::ThinkingSingleInferenceEngine::lastDEntries,
             py::return_value_policy::reference_internal,
             "Get D entries (position, value) from the last predict_move() call.")
        .def("idx_to_token", &decoder::ThinkingSingleInferenceEngine::idxToToken,
             py::arg("idx"),
             py::return_value_policy::reference_internal,
             "Convert a token ID to its string name.")
        .def_readwrite("board_temperature", &decoder::ThinkingSingleInferenceEngine::board_temperature)
        .def_readwrite("think_temperature", &decoder::ThinkingSingleInferenceEngine::think_temperature)
        .def_readwrite("policy_temperature", &decoder::ThinkingSingleInferenceEngine::policy_temperature)
        .def_readwrite("wl_temperature", &decoder::ThinkingSingleInferenceEngine::wl_temperature)
        .def_readwrite("d_temperature", &decoder::ThinkingSingleInferenceEngine::d_temperature)
        .def_readwrite("total_tokens", &decoder::ThinkingSingleInferenceEngine::total_tokens)
        .def_readwrite("total_time", &decoder::ThinkingSingleInferenceEngine::total_time)
        .def_readwrite("profiling", &decoder::ThinkingSingleInferenceEngine::profiling)
        .def("reset_profile", &decoder::ThinkingSingleInferenceEngine::resetProfile)
        .def_readwrite("prof_prefix_init", &decoder::ThinkingSingleInferenceEngine::prof_prefix_init)
        .def_readwrite("prof_board_prefill", &decoder::ThinkingSingleInferenceEngine::prof_board_prefill)
        .def_readwrite("prof_board_catchup", &decoder::ThinkingSingleInferenceEngine::prof_board_catchup)
        .def_readwrite("prof_board_gen", &decoder::ThinkingSingleInferenceEngine::prof_board_gen)
        .def_readwrite("prof_prefix_block", &decoder::ThinkingSingleInferenceEngine::prof_prefix_block)
        .def_readwrite("prof_prefix_incr", &decoder::ThinkingSingleInferenceEngine::prof_prefix_incr)
        .def_readwrite("prof_causal_incr", &decoder::ThinkingSingleInferenceEngine::prof_causal_incr)
        .def_readwrite("prof_head_eval", &decoder::ThinkingSingleInferenceEngine::prof_head_eval)
        .def_readwrite("prof_sync_ops", &decoder::ThinkingSingleInferenceEngine::prof_sync_ops);

    // ── Batched inference engine ──────────────────────────────────────
    py::class_<decoder::ThinkingBatchedInferenceEngine::Result>(m, "BatchedResult")
        .def_readonly("move", &decoder::ThinkingBatchedInferenceEngine::Result::move)
        .def_readonly("token_ids", &decoder::ThinkingBatchedInferenceEngine::Result::token_ids)
        .def_readonly("wl_entries", &decoder::ThinkingBatchedInferenceEngine::Result::wl_entries)
        .def_readonly("d_entries", &decoder::ThinkingBatchedInferenceEngine::Result::d_entries)
        .def_readonly("move_log_probs",
                      &decoder::ThinkingBatchedInferenceEngine::Result::move_log_probs)
        .def_readonly("wl_bucket_indices",
                      &decoder::ThinkingBatchedInferenceEngine::Result::wl_bucket_indices)
        .def_readonly("d_bucket_indices",
                      &decoder::ThinkingBatchedInferenceEngine::Result::d_bucket_indices)
        .def_readonly("wl_log_probs",
                      &decoder::ThinkingBatchedInferenceEngine::Result::wl_log_probs)
        .def_readonly("d_log_probs",
                      &decoder::ThinkingBatchedInferenceEngine::Result::d_log_probs);

    py::class_<decoder::ThinkingBatchedInferenceEngine>(m, "ThinkingBatchedInferenceEngine")
        .def(py::init<const std::string&, const std::string&,
                       const std::string&, const std::string&, int>(),
             py::arg("backbone_pt_path"),
             py::arg("weights_dir"),
             py::arg("vocab_path"),
             py::arg("config_path"),
             py::arg("max_batch_size"),
             "Construct batched engine for parallel rollout generation.")
        .def("predict_moves", &decoder::ThinkingBatchedInferenceEngine::predictMoves,
             py::arg("fens"),
             py::arg("temperature") = 0.0f,
             py::call_guard<py::gil_scoped_release>(),
             "Predict moves for a batch of FEN positions using thinking inference.")
        .def_readwrite("board_temperature", &decoder::ThinkingBatchedInferenceEngine::board_temperature)
        .def_readwrite("think_temperature", &decoder::ThinkingBatchedInferenceEngine::think_temperature)
        .def_readwrite("policy_temperature", &decoder::ThinkingBatchedInferenceEngine::policy_temperature)
        .def_readwrite("wl_temperature", &decoder::ThinkingBatchedInferenceEngine::wl_temperature)
        .def_readwrite("d_temperature", &decoder::ThinkingBatchedInferenceEngine::d_temperature)
        .def_readwrite("total_tokens", &decoder::ThinkingBatchedInferenceEngine::total_tokens)
        .def_readwrite("total_time", &decoder::ThinkingBatchedInferenceEngine::total_time);
}
