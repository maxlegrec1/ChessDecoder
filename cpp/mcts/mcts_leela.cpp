#include "inference.hpp"
#include "mcts/common.hpp"
#include "mcts/leela_runner.hpp"
#include "mcts/single_inference.hpp"
#include "mcts/summary.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <unordered_map>
#include <vector>

namespace chessrl::mcts
{
namespace
{

struct MovePrior
{
    chess::Move move;
    int policy_index{-1};
    float prior{0.0F};
    float logit{0.0F};
};

std::string mirror_uci_ranks(const std::string& move)
{
    auto mirror_rank = [](char c) -> char {
        if (c < '1' || c > '8')
        {
            return c;
        }
        const int rank = c - '0';
        const int mirrored = 9 - rank;
        return static_cast<char>('0' + mirrored);
    };

    if (move.size() < 4)
    {
        return move;
    }

    std::string mirrored = move;
    mirrored[1] = mirror_rank(mirrored[1]);
    mirrored[3] = mirror_rank(mirrored[3]);
    return mirrored;
}

std::string to_pseudo_castling(const std::string& move)
{
    if (move == "e1g1")
    {
        return "e1h1";
    }
    if (move == "e1c1")
    {
        return "e1a1";
    }
    if (move == "e8g8")
    {
        return "e8h8";
    }
    if (move == "e8c8")
    {
        return "e8a8";
    }
    return move;
}

std::array<float, 3> softmax3(const std::array<float, 3>& logits)
{
    float max_logit = std::max({logits[0], logits[1], logits[2]});
    std::array<float, 3> probs{};
    float sum = 0.0F;
    for (int i = 0; i < 3; ++i)
    {
        const float value = std::exp(logits[static_cast<size_t>(i)] - max_logit);
        probs[static_cast<size_t>(i)] = value;
        sum += value;
    }
    if (sum <= 0.0F)
    {
        return {1.0F / 3.0F, 1.0F / 3.0F, 1.0F / 3.0F};
    }
    const float inv_sum = 1.0F / sum;
    for (float& p : probs)
    {
        p *= inv_sum;
    }
    return probs;
}

std::optional<float> terminal_value(const chess::Board& board)
{
    const auto result = board.isGameOver().second;
    switch (result)
    {
    case chess::GameResult::WIN:
        return 1.0F;
    case chess::GameResult::LOSE:
        return -1.0F;
    case chess::GameResult::DRAW:
        return 0.0F;
    default:
        return std::nullopt;
    }
}

std::unordered_map<std::string, int> build_vocab_index(const std::vector<std::string>& vocab)
{
    std::unordered_map<std::string, int> index;
    index.reserve(vocab.size());
    for (size_t i = 0; i < vocab.size(); ++i)
    {
        index.emplace(vocab[i], static_cast<int>(i));
    }
    return index;
}

int read_batch_size_env(const char* specific)
{
    if (specific)
    {
        if (const char* value = std::getenv(specific))
        {
            const int numeric = std::atoi(value);
            if (numeric > 0)
            {
                return numeric;
            }
        }
    }
    if (const char* value = std::getenv("CHESSRL_MCTS_BATCH_SIZE"))
    {
        const int numeric = std::atoi(value);
        if (numeric > 0)
        {
            return numeric;
        }
    }
    return 32;
}

bool telemetry_enabled_env()
{
    if (const char* value = std::getenv("CHESSRL_MCTS_TELEMETRY"))
    {
        std::string lower(value);
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return lower != "0" && lower != "false" && lower != "off";
    }
    return false;
}

single::LeelaBatchConfig make_leela_batch_config(const MctsOptions& options)
{
    single::LeelaBatchConfig cfg;
    if (options.leela_engine_path.empty())
    {
        cfg.engine_path = "leela_minibatch.trt";
    }
    else
    {
        cfg.engine_path = options.leela_engine_path;
    }
    cfg.max_batch_size = read_batch_size_env("CHESSRL_MCTS_LEELA_BATCH_SIZE");
    cfg.enable_telemetry = telemetry_enabled_env();
    return cfg;
}

class LeelaMctsRunner
{
public:
    explicit LeelaMctsRunner(MctsOptions options)
        : options_(std::move(options))
        , origin_fen_(options_.request.fen)
        , vocab_(leela_policy_vocabulary())
        , vocab_index_(build_vocab_index(vocab_))
        , leela_batch_config_(make_leela_batch_config(options_))
        , leela_batch_runner_(leela_batch_config_)
        , leela_batch_flush_threshold_(static_cast<size_t>(leela_batch_config_.max_batch_size))
        , cooperative_mode_(false)
        , rng_(std::random_device{}())
    {
        const chess::Board board = apply_history(options_.request);
        tree_.add_root(board, options_.request.history);
        const size_t expected_nodes = 1 + static_cast<size_t>(options_.simulations) * 2;
        tree_.reserve(expected_nodes);
        path_buffer_.reserve(64);
    }

    // Parallel mode: shared external runner
    LeelaMctsRunner(MctsOptions options, single::LeelaBatchRunner external_runner)
        : options_(std::move(options))
        , origin_fen_(options_.request.fen)
        , vocab_(leela_policy_vocabulary())
        , vocab_index_(build_vocab_index(vocab_))
        , leela_batch_config_()
        , leela_batch_runner_(std::move(external_runner))
        , leela_batch_flush_threshold_(0)
        , cooperative_mode_(true)
        , rng_(std::random_device{}())
    {
        const chess::Board board = apply_history(options_.request);
        tree_.add_root(board, options_.request.history);
        const size_t expected_nodes = 1 + static_cast<size_t>(options_.simulations) * 2;
        tree_.reserve(expected_nodes);
        path_buffer_.reserve(64);
    }

    MctsSummary run()
    {
        evaluate_node(0);
        for (int i = 0; i < options_.simulations; ++i)
        {
            run_simulation();
        }
        MctsSummary summary = summarize();
        if (leela_batch_config_.enable_telemetry)
        {
            auto stats = leela_batch_runner_.statistics();
            if (stats.batches > 0)
            {
                std::clog << "[telemetry][leela] batches=" << stats.batches
                          << " positions=" << stats.positions
                          << " infer_ms=" << stats.total_inference_ms << '\n';
            }
        }
        return summary;
    }

private:
    void run_simulation()
    {
        path_buffer_.clear();
        path_buffer_.push_back(0);
        std::vector<int>& path = path_buffer_;
        int node_index = 0;

        while (true)
        {
            TreeNode& node = tree_.nodes[node_index];

            if (node.terminal)
            {
                float value = terminal_value(node.board).value_or(node.value_prior);
                backpropagate(path, value);
                return;
            }

            if (!node.expanded || node.children.empty())
            {
                float value = evaluate_node(node_index);
                backpropagate(path, value);
                return;
            }

            const int child_index = select_child(node_index);
            if (child_index < 0)
            {
                float value = evaluate_node(node_index);
                backpropagate(path, value);
                return;
            }

            node_index = child_index;
            path.push_back(node_index);
        }
    }

    int select_child(int node_index) const
    {
        const TreeNode& node = tree_.nodes[node_index];
        if (node.children.empty())
        {
            return -1;
        }

        const float parent_visits = static_cast<float>(std::max(1, node.visit_count));
        const float sqrt_visits = std::sqrt(parent_visits);

        int best_child = -1;
        float best_score = -std::numeric_limits<float>::infinity();

        for (size_t i = 0; i < node.children.size(); ++i)
        {
            const int child_index = node.children[i];
            const TreeNode& child = tree_.nodes[child_index];
            const float prior = node.child_priors[i];
            const float child_mean = child.visit_count > 0 ? child.value_sum / static_cast<float>(child.visit_count) : 0.0F;
            const float q = -child_mean;
            const float u = options_.cpuct * prior * sqrt_visits / (1.0F + static_cast<float>(child.visit_count));
            const float score = q + u;
            if (score > best_score)
            {
                best_score = score;
                best_child = child_index;
            }
        }

        return best_child;
    }

    void backpropagate(const std::vector<int>& path, float value)
    {
        float current = value;
        for (auto it = path.rbegin(); it != path.rend(); ++it)
        {
            TreeNode& node = tree_.nodes[*it];
            node.visit_count += 1;
            node.value_sum += current;
            current = -current;
        }
    }

    std::shared_ptr<const single::LeelaEncodedPosition> acquire_leela_encoding(
        const HistoryHandle& history,
        const std::vector<std::string>& move_history)
    {
        if (!history)
        {
            return std::make_shared<single::LeelaEncodedPosition>(single::encode_leela_position(origin_fen_, move_history));
        }
        const HistoryEntry* key = history.get();
        std::lock_guard<std::mutex> lock(leela_cache_mutex_);
        auto it = leela_encoding_cache_.find(key);
        if (it != leela_encoding_cache_.end())
        {
            if (auto existing = it->second.lock())
            {
                return existing;
            }
        }
        auto fresh = std::make_shared<single::LeelaEncodedPosition>(single::encode_leela_position(origin_fen_, move_history));
        leela_encoding_cache_[key] = fresh;
        return fresh;
    }

    float evaluate_node(int node_index)
    {
        TreeNode& node = tree_.nodes[node_index];
        if (node.expanded && !node.children.empty())
        {
            return node.value_prior;
        }

        if (auto terminal = terminal_value(node.board))
        {
            node.expanded = true;
            node.value_prior = *terminal;
            if (*terminal > 0.0F)
            {
                node.wdl = {1.0F, 0.0F, 0.0F};
            }
            else if (*terminal < 0.0F)
            {
                node.wdl = {0.0F, 0.0F, 1.0F};
            }
            else
            {
                node.wdl = {0.0F, 1.0F, 0.0F};
            }
            return node.value_prior;
        }

        const std::vector<std::string> history = history_to_vector(node.history);
        auto encoded = acquire_leela_encoding(node.history, history);
        auto ticket = leela_batch_runner_.enqueue(*encoded);

        LeelaPolicyValue eval;
        if (cooperative_mode_)
        {
            // PARALLEL: block on future. Worker thread handles processing.
            eval = ticket.get();
        }
        else
        {
            // SINGLE-TREE: caller helps flush (original behavior)
            if (leela_batch_runner_.pending() >= leela_batch_flush_threshold_)
            {
                leela_batch_runner_.process_pending();
            }
            while (!ticket.ready())
            {
                leela_batch_runner_.process_pending();
            }
            eval = ticket.get();
        }
        if (eval.policy_logits.size() != chessrl::leela_policy_size())
        {
            throw std::runtime_error("Single-batch Leela inference returned unexpected policy size.");
        }

        const bool board_black = node.board.sideToMove() == chess::Color::BLACK;
        if (eval.is_black_to_move != board_black)
        {
            throw std::runtime_error("Mismatch between board orientation and Leela evaluation result.");
        }

        const std::array<float, 3> value_probs_raw = softmax3(eval.value_q_logits);
        node.wdl = value_probs_raw;
        node.value_prior = node.wdl[0] - node.wdl[2];
        node.expanded = true;

        std::vector<MovePrior> priors = compute_priors(node.board, eval);
        if (priors.empty())
        {
            return node.value_prior;
        }

        if (node.parent == -1 && options_.use_dirichlet)
        {
            apply_dirichlet_noise(priors);
        }

        const chess::Board parent_board = node.board;
        const float value_prior = node.value_prior;
        for (const auto& entry : priors)
        {
            chess::Board child_board = parent_board;
            child_board.makeMove(entry.move);
            tree_.add_child(node_index, entry.move, entry.prior, child_board);
        }

        TreeNode& refreshed = tree_.nodes[node_index];
        refreshed.expanded = true;

        return value_prior;
    }

    std::vector<MovePrior> compute_priors(const chess::Board& board, const LeelaPolicyValue& eval) const
    {
        chess::Movelist legal_moves;
        chess::movegen::legalmoves(legal_moves, board);

        std::vector<MovePrior> entries;
        entries.reserve(legal_moves.size());

        float max_logit = -std::numeric_limits<float>::infinity();
        int mapped_count = 0;
        const bool is_black = board.sideToMove() == chess::Color::BLACK;

        for (const auto& move : legal_moves)
        {
            MovePrior entry;
            entry.move = move;
            entry.logit = -std::numeric_limits<float>::infinity();

            std::string uci = chess::uci::moveToUci(move, board.chess960());
            std::string key = uci;
            if (!key.empty() && key.back() == 'n')
            {
                key.pop_back();
            }
            key = to_pseudo_castling(key);
            if (is_black)
            {
                key = mirror_uci_ranks(key);
            }

            auto it = vocab_index_.find(key);
            if (it != vocab_index_.end())
            {
                entry.policy_index = it->second;
                if (entry.policy_index >= 0 && entry.policy_index < static_cast<int>(eval.policy_logits.size()))
                {
                    entry.logit = eval.policy_logits[static_cast<size_t>(entry.policy_index)];
                    max_logit = std::max(max_logit, entry.logit);
                    ++mapped_count;
                }
                else
                {
                    entry.policy_index = -1;
                }
            }

            entries.push_back(std::move(entry));
        }

        if (entries.empty())
        {
            return entries;
        }

        if (mapped_count == 0)
        {
            const float uniform = 1.0F / static_cast<float>(entries.size());
            for (auto& entry : entries)
            {
                entry.prior = uniform;
            }
            return entries;
        }

        if (options_.temperature <= 1e-5F)
        {
            auto best_it = std::max_element(entries.begin(), entries.end(), [](const MovePrior& a, const MovePrior& b) {
                return a.logit < b.logit;
            });
            for (auto& entry : entries)
            {
                const bool is_best = (entry.policy_index >= 0) && (&entry == &*best_it);
                entry.prior = is_best ? 1.0F : 0.0F;
            }
            return entries;
        }

        const float inv_temp = 1.0F / options_.temperature;
        double sum = 0.0;
        for (auto& entry : entries)
        {
            if (entry.policy_index < 0)
            {
                entry.prior = 0.0F;
                continue;
            }
            const double scaled = static_cast<double>((entry.logit - max_logit) * inv_temp);
            const double exp_val = std::exp(scaled);
            entry.prior = static_cast<float>(exp_val);
            sum += exp_val;
        }

        if (sum <= 0.0)
        {
            const float uniform = 1.0F / static_cast<float>(mapped_count);
            for (auto& entry : entries)
            {
                if (entry.policy_index >= 0)
                {
                    entry.prior = uniform;
                }
            }
            return entries;
        }

        const float inv_sum = static_cast<float>(1.0 / sum);
        for (auto& entry : entries)
        {
            if (entry.policy_index >= 0)
            {
                entry.prior *= inv_sum;
            }
        }

        return entries;
    }

    void apply_dirichlet_noise(std::vector<MovePrior>& priors)
    {
        if (priors.empty())
        {
            return;
        }

        std::gamma_distribution<float> gamma(options_.dirichlet_alpha, 1.0F);
        dirichlet_noise_.resize(priors.size());
        float sum = 0.0F;
        for (float& n : dirichlet_noise_)
        {
            n = gamma(rng_);
            sum += n;
        }
        if (sum <= 0.0F)
        {
            return;
        }

        const float epsilon = options_.dirichlet_epsilon;
        for (size_t i = 0; i < priors.size(); ++i)
        {
            const float dirichlet = dirichlet_noise_[i] / sum;
            priors[i].prior = (1.0F - epsilon) * priors[i].prior + epsilon * dirichlet;
        }
    }

    MctsSummary summarize() const
    {
        const TreeNode& root = tree_.nodes.front();

        std::vector<std::pair<std::string, float>> distribution;
        distribution.reserve(root.children.size());

        struct Aggregated
        {
            std::string uci;
            int visit_sum{0};
            float prior_sum{0.0F};
            float value_sum{0.0F};
            int representative_child{-1};
        };

        std::vector<Aggregated> aggregated;
        aggregated.reserve(root.children.size());
        std::unordered_map<std::string, size_t> index;

        int total_visits = 0;
        float total_priors = 0.0F;

        for (size_t i = 0; i < root.children.size(); ++i)
        {
            const int child_index = root.children[i];
            const TreeNode& child = tree_.nodes[child_index];
            const chess::Move move = root.child_moves[i];
            const std::string uci = chess::uci::moveToUci(move, root.board.chess960());

            auto [it, inserted] = index.emplace(uci, aggregated.size());
            if (inserted)
            {
                aggregated.push_back(Aggregated{uci});
            }

            Aggregated& entry = aggregated[it->second];
            entry.visit_sum += child.visit_count;
            entry.prior_sum += root.child_priors[i];
            entry.value_sum += child.value_sum;
            if (entry.representative_child < 0 || child.visit_count > tree_.nodes[entry.representative_child].visit_count)
            {
                entry.representative_child = child_index;
            }

            total_visits += child.visit_count;
            total_priors += root.child_priors[i];
        }

        if (aggregated.empty())
        {
            return MctsSummary{
                .action = {},
                .policy = {},
                .value = root.wdl,
                .rollouts = {},
            };
        }

        const float visit_sum = static_cast<float>(total_visits);
        const float prior_total = total_priors;

        if (visit_sum <= 0.0F && prior_total <= 0.0F)
        {
            const float uniform = 1.0F / static_cast<float>(aggregated.size());
            for (const auto& entry : aggregated)
            {
                distribution.emplace_back(entry.uci, uniform);
            }
        }
        else
        {
            const float normalizer = visit_sum > 0.0F ? visit_sum : prior_total;
            for (const auto& entry : aggregated)
            {
                const float value = visit_sum > 0.0F ? static_cast<float>(entry.visit_sum) : entry.prior_sum;
                distribution.emplace_back(entry.uci, value / normalizer);
            }
        }

        std::vector<std::pair<std::string, float>> sorted = distribution;
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            if (a.second == b.second)
            {
                return a.first < b.first;
            }
            return a.second > b.second;
        });

        size_t best_index = 0;
        float best_score = -1.0F;
        if (visit_sum > 0.0F)
        {
            for (const auto& entry : aggregated)
            {
                const float visits = static_cast<float>(entry.visit_sum);
                if (visits > best_score)
                {
                    best_score = visits;
                    best_index = index.at(entry.uci);
                }
            }
        }
        else
        {
            for (const auto& entry : aggregated)
            {
                if (entry.prior_sum > best_score)
                {
                    best_score = entry.prior_sum;
                    best_index = index.at(entry.uci);
                }
            }
        }

        const std::string action = aggregated[best_index].uci;

        MctsSummary summary;
        summary.action = action;
        summary.policy = std::move(sorted);
        summary.value = root.wdl;
        summary.rollouts.reserve(aggregated.size());
        for (const auto& entry : aggregated)
        {
            RolloutDetail detail;
            detail.root_move = entry.uci;
            detail.visit_count = entry.visit_sum;
            if (visit_sum > 0.0F)
            {
                detail.visit_fraction = static_cast<float>(entry.visit_sum) / visit_sum;
            }
            detail.prior = entry.prior_sum;
            detail.average_value = entry.visit_sum > 0 ? -entry.value_sum / static_cast<float>(entry.visit_sum) : 0.0F;
            if (entry.representative_child >= 0)
            {
                detail.continuation = principal_variation(tree_, entry.representative_child);
            }
            summary.rollouts.push_back(std::move(detail));
        }

        if (options_.extract_variations)
        {
            std::vector<Aggregated> sorted_agg = aggregated;
            std::sort(sorted_agg.begin(), sorted_agg.end(), [](const Aggregated& a, const Aggregated& b) {
                return a.visit_sum > b.visit_sum;
            });

            const int k = std::min(options_.max_variations, static_cast<int>(sorted_agg.size()));
            summary.variations.reserve(static_cast<size_t>(k));
            for (int i = 0; i < k; ++i)
            {
                const Aggregated& agg = sorted_agg[static_cast<size_t>(i)];
                VariationLine var;
                var.root_move = agg.uci;
                var.visit_count = agg.visit_sum;
                var.visit_fraction = visit_sum > 0.0F ? static_cast<float>(agg.visit_sum) / visit_sum : 0.0F;
                var.prior = agg.prior_sum;
                if (agg.representative_child >= 0)
                {
                    var.nodes = principal_variation_detailed(tree_, agg.representative_child, options_.max_variation_depth);
                }
                summary.variations.push_back(std::move(var));
            }
        }

        return summary;
    }

    MctsOptions options_;
    std::string origin_fen_;
    SearchTree tree_;
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> vocab_index_;
    single::LeelaBatchConfig leela_batch_config_;
    single::LeelaBatchRunner leela_batch_runner_;
    size_t leela_batch_flush_threshold_{0};
    bool cooperative_mode_{false};
    mutable std::mutex leela_cache_mutex_;
    std::unordered_map<const HistoryEntry*, std::weak_ptr<const single::LeelaEncodedPosition>> leela_encoding_cache_;
    std::mt19937 rng_;
    std::vector<int> path_buffer_;
    std::vector<float> dirichlet_noise_;
};

} // namespace
} // namespace chessrl::mcts

namespace chessrl::mcts
{

MctsSummary run_leela_mcts(MctsOptions options)
{
    LeelaMctsRunner runner(std::move(options));
    return runner.run();
}

std::vector<MctsSummary> run_parallel_leela_mcts(
    std::vector<MctsOptions> requests,
    const std::string& engine_path,
    int max_batch_size)
{
    if (requests.empty())
    {
        return {};
    }

    // One shared batch runner with dynamic engine
    single::LeelaBatchConfig cfg;
    cfg.engine_path = engine_path;
    cfg.max_batch_size = max_batch_size;
    cfg.enable_telemetry = telemetry_enabled_env();
    cfg.flush_threshold = 1;        // wake on first item
    cfg.collect_timeout_us = 500;   // 500us collect window
    single::LeelaBatchRunner shared_runner(cfg);

    const size_t n = requests.size();
    std::vector<MctsSummary> results(n);
    std::vector<std::exception_ptr> errors(n, nullptr);

    // One thread per MCTS tree
    std::vector<std::thread> threads;
    threads.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        threads.emplace_back([i, &requests, &results, &errors, &shared_runner]() {
            try
            {
                LeelaMctsRunner runner(std::move(requests[i]), shared_runner);
                results[i] = runner.run();
            }
            catch (...)
            {
                errors[i] = std::current_exception();
            }
        });
    }

    for (auto& t : threads)
    {
        t.join();
    }

    // Print telemetry once
    if (cfg.enable_telemetry)
    {
        auto stats = shared_runner.statistics();
        if (stats.batches > 0)
        {
            std::clog << "[telemetry][parallel-leela] batches=" << stats.batches
                      << " positions=" << stats.positions
                      << " infer_ms=" << stats.total_inference_ms << '\n';
        }
    }

    // Re-throw first error
    for (size_t i = 0; i < n; ++i)
    {
        if (errors[i])
        {
            std::rethrow_exception(errors[i]);
        }
    }

    return results;
}

#ifndef CHESSRL_MCTS_NO_STANDALONE
int main(int argc, char** argv)
{
    try
    {
        auto options = chessrl::mcts::parse_cli(argc, argv);
        auto summary = chessrl::mcts::run_leela_mcts(std::move(options));

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{\n";
        if (summary.action.empty())
        {
            std::cout << "  \"action\": null,\n";
        }
        else
        {
            std::cout << "  \"action\": \"" << summary.action << "\",\n";
        }
        std::cout << "  \"policy\": {\n";
        for (size_t i = 0; i < summary.policy.size(); ++i)
        {
            const auto& [move, prob] = summary.policy[i];
            std::cout << "    \"" << move << "\": " << prob;
            if (i + 1 < summary.policy.size())
            {
                std::cout << ",";
            }
            std::cout << "\n";
        }
        std::cout << "  },\n";
        std::cout << "  \"value\": [" << summary.value[0] << ", " << summary.value[1] << ", " << summary.value[2] << "]\n";
        std::cout << "}\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
#endif // CHESSRL_MCTS_NO_STANDALONE

} // namespace chessrl::mcts

