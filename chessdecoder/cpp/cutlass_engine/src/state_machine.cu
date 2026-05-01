#include "cutlass_engine/state_machine.hpp"

namespace cutlass_engine {

const char* slot_state_name(SlotState s) {
    switch (s) {
        case SlotState::Idle:        return "Idle";
        case SlotState::Init:        return "Init";
        case SlotState::Move:        return "Move";
        case SlotState::WlD:         return "WlD";
        case SlotState::Board:       return "Board";
        case SlotState::AfterBoard:  return "AfterBoard";
        case SlotState::AfterEndVar: return "AfterEndVar";
        case SlotState::Final:       return "Final";
        case SlotState::Done:        return "Done";
    }
    return "Unknown";
}

}  // namespace cutlass_engine
