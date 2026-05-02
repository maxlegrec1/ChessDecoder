#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "cutlass_engine/engine.hpp"
#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/scheduler.hpp"

namespace py = pybind11;
using namespace cutlass_engine;

// ---- Kernel test bindings ----------------------------------------------
//
// These accept device pointers as Python ints (torch tensor.data_ptr()).
// Used by tests/test_kernels.py to validate each kernel against a torch
// reference. No libtorch linkage needed; we just receive device pointers.
//
// The kernels themselves are in src/kernels/*.cu; these wrappers exist
// purely so the test harness can drive them from Python.

namespace test_api {

void rmsnorm_fp16(std::uintptr_t x, std::uintptr_t w, std::uintptr_t y,
                  int M, int E, float eps) {
    cutlass_engine::rmsnorm_fp16(reinterpret_cast<const __half*>(x),
                                 reinterpret_cast<const __half*>(w),
                                 reinterpret_cast<__half*>(y),
                                 M, E, eps, /*stream=*/0);
    cudaDeviceSynchronize();
}

void rmsnorm_residual_fp16(std::uintptr_t x, std::uintptr_t r,
                           std::uintptr_t w, std::uintptr_t y,
                           std::uintptr_t out_r, int M, int E, float eps) {
    cutlass_engine::rmsnorm_residual_fp16(
        reinterpret_cast<const __half*>(x),
        reinterpret_cast<const __half*>(r),
        reinterpret_cast<const __half*>(w),
        reinterpret_cast<__half*>(y),
        reinterpret_cast<__half*>(out_r),
        M, E, eps, /*stream=*/0);
    cudaDeviceSynchronize();
}

void rope_apply_qk_fp16(std::uintptr_t Q, std::uintptr_t K, std::uintptr_t pos,
                        std::uintptr_t cos_t, std::uintptr_t sin_t,
                        int M, int NH, int HD, int rope_max) {
    cutlass_engine::rope_apply_qk_fp16(
        reinterpret_cast<__half*>(Q),
        reinterpret_cast<__half*>(K),
        reinterpret_cast<const std::int32_t*>(pos),
        reinterpret_cast<const float*>(cos_t),
        reinterpret_cast<const float*>(sin_t),
        M, NH, HD, rope_max, /*stream=*/0);
    cudaDeviceSynchronize();
}

void swiglu_fp16(std::uintptr_t gate_up, std::uintptr_t y, int M, int d_ff) {
    cutlass_engine::swiglu_fp16(
        reinterpret_cast<const __half*>(gate_up),
        reinterpret_cast<__half*>(y),
        M, d_ff, /*stream=*/0);
    cudaDeviceSynchronize();
}

void mish_inplace_fp16(std::uintptr_t x, int N) {
    cutlass_engine::mish_inplace_fp16(reinterpret_cast<__half*>(x), N, /*stream=*/0);
    cudaDeviceSynchronize();
}

void argmax_fp16(std::uintptr_t logits, std::uintptr_t out, int B, int V) {
    cutlass_engine::argmax_fp16(reinterpret_cast<const __half*>(logits),
                                reinterpret_cast<std::int32_t*>(out),
                                B, V, /*stream=*/0);
    cudaDeviceSynchronize();
}

void gemm_fp16(std::uintptr_t A, std::uintptr_t Bw, std::uintptr_t C,
               int M, int N, int K) {
    cutlass_engine::gemm_fp16(reinterpret_cast<const __half*>(A),
                              reinterpret_cast<const __half*>(Bw),
                              /*bias=*/nullptr,
                              reinterpret_cast<__half*>(C),
                              M, N, K, nullptr, 0, /*stream=*/0);
    cudaDeviceSynchronize();
}

void fmha_decode_dispatch(std::uintptr_t Q, std::uintptr_t Kc, std::uintptr_t Vc,
                          std::uintptr_t past_len, std::uintptr_t active,
                          std::uintptr_t O, int B, int NH, int HD,
                          int max_seq_len, int layer_idx, float scale) {
    cutlass_engine::fmha_decode_dispatch(
        reinterpret_cast<const __half*>(Q),
        reinterpret_cast<const __half*>(Kc),
        reinterpret_cast<const __half*>(Vc),
        reinterpret_cast<const std::int32_t*>(past_len),
        reinterpret_cast<const std::int32_t*>(active),
        reinterpret_cast<__half*>(O),
        B, NH, HD, max_seq_len, layer_idx, scale, /*stream=*/0);
    cudaDeviceSynchronize();
}

void fmha_prefill_dispatch(std::uintptr_t Q, std::uintptr_t K, std::uintptr_t V,
                           std::uintptr_t block_id, std::uintptr_t active,
                           std::uintptr_t O, int B, int S, int NH, int HD,
                           float scale) {
    cutlass_engine::fmha_prefill_dispatch(
        reinterpret_cast<const __half*>(Q),
        reinterpret_cast<const __half*>(K),
        reinterpret_cast<const __half*>(V),
        reinterpret_cast<const std::int32_t*>(block_id),
        reinterpret_cast<const std::int32_t*>(active),
        reinterpret_cast<__half*>(O),
        B, S, NH, HD, scale, /*stream=*/0);
    cudaDeviceSynchronize();
}

void fmha_prefill_cutlass_causal(std::uintptr_t Q, std::uintptr_t K, std::uintptr_t V,
                                 std::uintptr_t O, int B, int S, int NH, int HD,
                                 float scale, std::uintptr_t workspace,
                                 std::uintptr_t lse_buf) {
    cutlass_engine::fmha_prefill_cutlass_causal(
        reinterpret_cast<const __half*>(Q),
        reinterpret_cast<const __half*>(K),
        reinterpret_cast<const __half*>(V),
        reinterpret_cast<__half*>(O),
        B, S, NH, HD, scale,
        reinterpret_cast<void*>(workspace),
        reinterpret_cast<void*>(lse_buf),
        /*stream=*/0);
    cudaDeviceSynchronize();
}

std::size_t fmha_prefill_cutlass_workspace_bytes(int B, int S, int NH, int HD) {
    return cutlass_engine::fmha_prefill_cutlass_workspace_bytes(B, S, NH, HD);
}

std::size_t fmha_prefill_cutlass_lse_elements(int B, int S, int NH) {
    return cutlass_engine::fmha_prefill_cutlass_lse_elements(B, S, NH);
}

}  // namespace test_api

PYBIND11_MODULE(_cutlass_decoder_cpp, m) {
    m.doc() = "CUTLASS-based ChessDecoder inference engine (FP16 + FP8) — pybind11 surface";

    py::class_<cutlass_engine::RolloutResult>(m, "RolloutResult")
        .def_readonly("move", &cutlass_engine::RolloutResult::move)
        .def_readonly("token_ids", &cutlass_engine::RolloutResult::token_ids)
        .def_readonly("block_ids", &cutlass_engine::RolloutResult::block_ids)
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
        .def_readonly("final_d_value",  &cutlass_engine::RolloutResult::final_d_value)
        .def_readonly("ended_thinking",&cutlass_engine::RolloutResult::ended_thinking)
        .def_readonly("truncated",     &cutlass_engine::RolloutResult::truncated);

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
        .def("predict_moves_thinking",
             &cutlass_engine::ThinkingEngine::predict_moves_thinking,
             py::arg("fens"), py::arg("temperature") = 0.0f,
             py::arg("max_seq_len_cap") = 1024,
             py::arg("max_iters") = 16)
        .def("update_weights",
             &cutlass_engine::ThinkingEngine::update_weights,
             py::arg("weights_dir"))
        .def("forward_decode_test",
             &cutlass_engine::ThinkingEngine::forward_decode_test,
             py::arg("ids"), py::arg("pos"), py::arg("active"),
             py::arg("past_len"), py::arg("out_h"))
        .def("forward_decode_partial",
             &cutlass_engine::ThinkingEngine::forward_decode_partial,
             py::arg("ids"), py::arg("pos"), py::arg("active"),
             py::arg("past_len"), py::arg("stop_after_layer"),
             py::arg("out_h_in"), py::arg("out_residual"));

    // ---- Kernel test surface ------------------------------------------
    auto kernels = m.def_submodule("kernels", "Direct kernel calls for unit testing.");
    kernels.def("rmsnorm_fp16", &test_api::rmsnorm_fp16,
                py::arg("x"), py::arg("w"), py::arg("y"),
                py::arg("M"), py::arg("E"), py::arg("eps"));
    kernels.def("rmsnorm_residual_fp16", &test_api::rmsnorm_residual_fp16,
                py::arg("x"), py::arg("residual"), py::arg("w"),
                py::arg("y"), py::arg("out_residual"),
                py::arg("M"), py::arg("E"), py::arg("eps"));
    kernels.def("rope_apply_qk_fp16", &test_api::rope_apply_qk_fp16,
                py::arg("Q"), py::arg("K"), py::arg("pos"),
                py::arg("cos"), py::arg("sin"),
                py::arg("M"), py::arg("num_heads"), py::arg("head_dim"),
                py::arg("rope_max_seq"));
    kernels.def("swiglu_fp16", &test_api::swiglu_fp16,
                py::arg("gate_up"), py::arg("y"),
                py::arg("M"), py::arg("d_ff"));
    kernels.def("mish_inplace_fp16", &test_api::mish_inplace_fp16,
                py::arg("x"), py::arg("N"));
    kernels.def("argmax_fp16", &test_api::argmax_fp16,
                py::arg("logits"), py::arg("idx_out"),
                py::arg("B"), py::arg("V"));
    kernels.def("gemm_fp16", &test_api::gemm_fp16,
                py::arg("A"), py::arg("B_w"), py::arg("C"),
                py::arg("M"), py::arg("N"), py::arg("K"));
    kernels.def("fmha_decode_dispatch", &test_api::fmha_decode_dispatch,
                py::arg("Q"), py::arg("K_cache"), py::arg("V_cache"),
                py::arg("past_len"), py::arg("active"), py::arg("O"),
                py::arg("B"), py::arg("NH"), py::arg("HD"),
                py::arg("max_seq_len"), py::arg("layer_idx"), py::arg("scale"));
    kernels.def("fmha_prefill_dispatch", &test_api::fmha_prefill_dispatch,
                py::arg("Q"), py::arg("K"), py::arg("V"),
                py::arg("block_id"), py::arg("active"), py::arg("O"),
                py::arg("B"), py::arg("S"), py::arg("NH"), py::arg("HD"),
                py::arg("scale"));
    kernels.def("fmha_prefill_cutlass_causal", &test_api::fmha_prefill_cutlass_causal,
                py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
                py::arg("B"), py::arg("S"), py::arg("NH"), py::arg("HD"),
                py::arg("scale"), py::arg("workspace"), py::arg("lse_buf"));
    kernels.def("fmha_prefill_cutlass_workspace_bytes",
                &test_api::fmha_prefill_cutlass_workspace_bytes,
                py::arg("B"), py::arg("S"), py::arg("NH"), py::arg("HD"));
    kernels.def("fmha_prefill_cutlass_lse_elements",
                &test_api::fmha_prefill_cutlass_lse_elements,
                py::arg("B"), py::arg("S"), py::arg("NH"));
}
