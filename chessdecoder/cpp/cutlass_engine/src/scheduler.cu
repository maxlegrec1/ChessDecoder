#include "cutlass_engine/scheduler.hpp"

namespace cutlass_engine {

void Scheduler::initialize(int batch_size, int num_fens) {
    B_ = batch_size;
    slots_.assign(B_, SlotInfo{});
    pending_.clear();
    results_.assign(num_fens, RolloutResult{});
}

void Scheduler::enqueue_all() {
    for (int i = 0; i < (int)results_.size(); ++i) pending_.push_back(i);
}

std::vector<int> Scheduler::refill_idle_slots() {
    std::vector<int> refilled;
    for (int b = 0; b < B_ && !pending_.empty(); ++b) {
        if (slots_[b].state == SlotState::Idle) {
            int fid = pending_.front();
            pending_.pop_front();
            slots_[b] = SlotInfo{};
            slots_[b].state = SlotState::Init;
            slots_[b].fen_id = fid;
            refilled.push_back(b);
        }
    }
    return refilled;
}

void Scheduler::mark_done(int slot) {
    slots_[slot].state = SlotState::Idle;
    slots_[slot].fen_id = -1;
    slots_[slot].board_step = 0;
    slots_[slot].thinking_step = 0;
}

bool Scheduler::all_done() const {
    if (!pending_.empty()) return false;
    for (auto& s : slots_) {
        if (s.state != SlotState::Idle) return false;
    }
    return true;
}

}  // namespace cutlass_engine
