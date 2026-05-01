#pragma once

#include <cstdint>

namespace cutlass_engine {

// Per-slot state in the thinking-trace generation.  Same machine as the
// existing engine: MOVE → WL_D → BOARD → AFTER_BOARD → AFTER_END_VAR → FINAL.
//
// Each state corresponds to a different combination of (forward mode, head
// to evaluate, sampler rule, transition rule).
enum class SlotState : uint8_t {
    Idle = 0,           // empty slot, available to refill from queue
    Init = 1,           // needs prefill on the new FEN
    Move = 2,           // sample a (variation) move from thinking_policy_head
    WlD = 3,            // sample WL_idx then D_idx
    Board = 4,          // generate 67 board tokens via board_head (decode loop)
    AfterBoard = 5,     // decide end_var
    AfterEndVar = 6,    // decide end_think
    Final = 7,          // produce the final move via policy_head
    Done = 8,           // result committed; will become Idle next refill check
};

constexpr int kNumSlotStates = 9;

const char* slot_state_name(SlotState s);

}  // namespace cutlass_engine
