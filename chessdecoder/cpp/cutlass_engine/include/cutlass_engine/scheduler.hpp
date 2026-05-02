#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "cutlass_engine/config.hpp"
#include "cutlass_engine/state_machine.hpp"

namespace cutlass_engine {

// Per-rollout result mirroring the existing _decoder_inference_cpp.BatchedResult
// shape (so chessdecoder/rl/rollout.py doesn't need to change).
struct RolloutResult {
    std::string move;
    std::vector<int32_t> token_ids;
    std::vector<int32_t> block_ids;       // matches Python ThinkingResult
    // Sampled-move bookkeeping (Phase K). Parallel arrays:
    //   move_positions[i] is the index in token_ids where the move was placed
    //   move_log_probs[i] is the log-prob of that sampled token under the
    //   policy (thinking_policy_head for variation moves, policy_head for
    //   the final move). Computed inline by log_prob_at_idx_fp16.
    std::vector<int32_t> move_positions;
    std::vector<float>   move_log_probs;
    // Value bookkeeping (same as existing engine; *_log_probs are now
    // populated parallel to *_positions).
    std::vector<int32_t> wl_positions;
    std::vector<int32_t> wl_indices;
    std::vector<float>   wl_values;
    std::vector<float>   wl_log_probs;
    std::vector<int32_t> d_positions;
    std::vector<int32_t> d_indices;
    std::vector<float>   d_values;
    std::vector<float>   d_log_probs;
    int32_t final_wl_index{-1};
    float   final_wl_value{0};
    int32_t final_d_index{-1};
    float   final_d_value{0};
    bool ended_thinking{false};
    bool truncated{false};
};

// Per-slot scheduling state. CPU-only — the engine reads state[B] each step
// to decide which slots to launch in which mode.
struct SlotInfo {
    SlotState state{SlotState::Idle};
    int fen_id{-1};            // index into the input fen vector
    int board_step{0};         // 0..67 inside Board state
    int thinking_step{0};      // count of variation iterations elapsed
};

// Continuous-batch scheduler. Holds:
//   - pending queue of fen_ids waiting for a slot
//   - per-slot state machine
//   - results vector indexed by fen_id (in submission order)
//
// Methods are CPU-side; engine.cu drives the kernel launches.
class Scheduler {
public:
    void initialize(int batch_size, int num_fens);

    // Enqueue all FENs at indices [0, num_fens).
    void enqueue_all();

    // Refill all idle slots from the queue. Returns the slot indices that
    // were transitioned from Idle to Init.
    std::vector<int> refill_idle_slots();

    // Mark slot as done. Caller commits the result via mutable_result(b).
    void mark_done(int slot);

    // Returns true when no slots active and queue empty.
    bool all_done() const;

    int batch_size() const { return B_; }
    SlotInfo& slot(int b) { return slots_[b]; }
    const SlotInfo& slot(int b) const { return slots_[b]; }

    RolloutResult& mutable_result(int fen_id) { return results_[fen_id]; }
    std::vector<RolloutResult>&& take_results() { return std::move(results_); }

private:
    int B_{0};
    std::vector<SlotInfo> slots_;
    std::deque<int> pending_;
    std::vector<RolloutResult> results_;
};

}  // namespace cutlass_engine
