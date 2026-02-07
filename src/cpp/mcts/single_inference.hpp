#pragma once

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "chess-library/include/chess.hpp"

#include "inference.hpp"

namespace chessrl::mcts::single
{

template <typename Result>
class BatchTicket
{
public:
    BatchTicket() = default;
    explicit BatchTicket(std::shared_future<Result> future)
        : future_(std::move(future))
    {
    }

    bool valid() const noexcept
    {
        return future_.valid();
    }

    bool ready() const
    {
        if (!future_.valid())
        {
            return false;
        }
        return future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    Result get() const
    {
        return future_.get();
    }

private:
    std::shared_future<Result> future_;
};

struct SmallEncodedPosition
{
    std::vector<float> features;
};

struct BatchStatistics
{
    size_t batches{0};
    size_t positions{0};
    double total_inference_ms{0.0};
};

SmallEncodedPosition encode_small_position(const chess::Board& board);

struct LeelaEncodedPosition
{
    std::vector<float> features;
    bool is_black_to_move{false};
    std::string castling_rights;
};

LeelaEncodedPosition encode_leela_position(std::string_view origin_fen, const std::vector<std::string>& move_history);

struct SmallBatchConfig
{
    std::string engine_path{"model_minibatch.trt"};
    int max_batch_size{32};
    bool enable_telemetry{false};
};

class SmallBatchRunner
{
public:
    using Ticket = BatchTicket<SmallPolicyValue>;

    explicit SmallBatchRunner(SmallBatchConfig config = {});

    Ticket enqueue(SmallEncodedPosition position);
    void process_pending();
    size_t pending() const;
    BatchStatistics statistics() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

struct LeelaBatchConfig
{
    std::string engine_path{"leela_minibatch.trt"};
    int max_batch_size{32};
    bool enable_telemetry{false};
    int flush_threshold{-1};      // -1 = use max_batch_size
    int collect_timeout_us{0};    // 0 = no timeout (original behavior)
};

class LeelaBatchRunner
{
public:
    using Ticket = BatchTicket<LeelaPolicyValue>;

    explicit LeelaBatchRunner(LeelaBatchConfig config = {});

    Ticket enqueue(LeelaEncodedPosition position);
    void process_pending();
    size_t pending() const;
    BatchStatistics statistics() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace chessrl::mcts::single


