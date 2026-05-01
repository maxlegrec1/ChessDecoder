#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "cutlass_engine/engine.hpp"
#include "cutlass_engine/scheduler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cutlass_decoder_cpp, m) {
    m.doc() = "CUTLASS-based ChessDecoder inference engine (FP16 + FP8) — pybind11 surface";

    py::class_<cutlass_engine::RolloutResult>(m, "RolloutResult")
        .def_readonly("move", &cutlass_engine::RolloutResult::move)
        .def_readonly("token_ids", &cutlass_engine::RolloutResult::token_ids)
        .def_readonly("log_probs", &cutlass_engine::RolloutResult::log_probs)
        .def_readonly("wl_positions", &cutlass_engine::RolloutResult::wl_positions)
        .def_readonly("wl_indices",   &cutlass_engine::RolloutResult::wl_indices)
        .def_readonly("wl_values",    &cutlass_engine::RolloutResult::wl_values)
        .def_readonly("wl_log_probs", &cutlass_engine::RolloutResult::wl_log_probs)
        .def_readonly("d_positions",  &cutlass_engine::RolloutResult::d_positions)
        .def_readonly("d_indices",    &cutlass_engine::RolloutResult::d_indices)
        .def_readonly("d_values",     &cutlass_engine::RolloutResult::d_values)
        .def_readonly("d_log_probs",  &cutlass_engine::RolloutResult::d_log_probs)
        .def_readonly("final_wl_index", &cutlass_engine::RolloutResult::final_wl_index)
        .def_readonly("final_wl_value", &cutlass_engine::RolloutResult::final_wl_value)
        .def_readonly("final_d_index",  &cutlass_engine::RolloutResult::final_d_index)
        .def_readonly("final_d_value",  &cutlass_engine::RolloutResult::final_d_value);

    py::class_<cutlass_engine::ThinkingEngine>(m, "ThinkingEngine")
        .def(py::init<const std::string&, const std::string&,
                      const std::string&, const std::string&, int>(),
             py::arg("backbone_pt"), py::arg("weights_dir"),
             py::arg("vocab_json"), py::arg("config_json"),
             py::arg("batch_size"))
        .def_property("board_temperature", nullptr,
                      &cutlass_engine::ThinkingEngine::set_board_temperature)
        .def_property("think_temperature", nullptr,
                      &cutlass_engine::ThinkingEngine::set_think_temperature)
        .def_property("policy_temperature", nullptr,
                      &cutlass_engine::ThinkingEngine::set_policy_temperature)
        .def_property("wl_temperature", nullptr,
                      &cutlass_engine::ThinkingEngine::set_wl_temperature)
        .def_property("d_temperature", nullptr,
                      &cutlass_engine::ThinkingEngine::set_d_temperature)
        .def("predict_moves",
             &cutlass_engine::ThinkingEngine::predict_moves,
             py::arg("fens"), py::arg("temperature") = 1.0f)
        .def("update_weights",
             &cutlass_engine::ThinkingEngine::update_weights,
             py::arg("weights_dir"));
}
