// Pybind11 bindings for the V2 inference + MCTS engine.
//
// Exposes:
//   BoardForward (load TS, run forward)
//   Vocab        (load JSON, tokenize FEN)
//   V2Mcts       (PUCT search returning action + visit policy + WDL)
#include <memory>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "board_forward.hpp"
#include "mcts_v2.hpp"
#include "vocab_v2.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_v2_inference_cpp, m) {
  m.doc() = "V2 ChessDecoder inference + PUCT MCTS (libtorch TorchScript)";

  // ---- Vocab ----
  py::class_<v2::Vocab, std::shared_ptr<v2::Vocab>>(m, "Vocab")
      .def_static("from_json", &v2::Vocab::from_json, py::arg("path"))
      .def("fen_to_board_ids",
           [](const v2::Vocab& self, const std::string& fen) {
             auto arr = self.fen_to_board_ids(fen);
             return std::vector<int64_t>(arr.begin(), arr.end());
           })
      .def("move_sub_id_to_uci", &v2::Vocab::move_sub_id_to_uci)
      .def("uci_to_move_sub_id", &v2::Vocab::uci_to_move_sub_id);

  // ---- BoardForward ----
  py::class_<v2::LeafEval>(m, "LeafEval")
      .def_readonly("policy", &v2::LeafEval::policy)
      .def_readonly("w", &v2::LeafEval::w)
      .def_readonly("d", &v2::LeafEval::d)
      .def_readonly("l", &v2::LeafEval::l);

  py::class_<v2::BoardForward, std::shared_ptr<v2::BoardForward>>(
      m, "BoardForward")
      .def(py::init<const std::string&, const std::string&>(),
           py::arg("module_path"), py::arg("device") = "cuda:0")
      .def("forward_one", &v2::BoardForward::forward_one)
      .def("forward_batch", &v2::BoardForward::forward_batch);

  // ---- MCTS ----
  py::class_<v2::MctsConfig>(m, "MctsConfig")
      .def(py::init<>())
      .def_readwrite("simulations", &v2::MctsConfig::simulations)
      .def_readwrite("cpuct", &v2::MctsConfig::cpuct)
      .def_readwrite("temperature", &v2::MctsConfig::temperature)
      .def_readwrite("max_batch_leaves", &v2::MctsConfig::max_batch_leaves);

  py::class_<v2::MctsResult>(m, "MctsResult")
      .def_readonly("action", &v2::MctsResult::action)
      .def_readonly("policy", &v2::MctsResult::policy)
      .def_readonly("q_values", &v2::MctsResult::q_values)
      .def_readonly("root_w", &v2::MctsResult::root_w)
      .def_readonly("root_d", &v2::MctsResult::root_d)
      .def_readonly("root_l", &v2::MctsResult::root_l)
      .def_readonly("sims_done", &v2::MctsResult::sims_done);

  py::class_<v2::V2Mcts>(m, "V2Mcts")
      .def(py::init<std::shared_ptr<v2::BoardForward>,
                    std::shared_ptr<v2::Vocab>, v2::MctsConfig>(),
           py::arg("net"), py::arg("vocab"), py::arg("config") = v2::MctsConfig{})
      .def("search", &v2::V2Mcts::search, py::arg("fen"));
}
