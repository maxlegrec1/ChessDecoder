#include "inference.hpp"
#include "mcts/adversarial_runner.hpp"
#include "mcts/common.hpp"
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

single::SmallBatchConfig make_small_batch_config(const MctsOptions& options)
{
    single::SmallBatchConfig cfg;
    if (options.small_engine_path.empty())
    {
        cfg.engine_path = "model_minibatch.trt";
    }
    else
    {
        cfg.engine_path = options.small_engine_path;
    }
    cfg.max_batch_size = read_batch_size_env("CHESSRL_MCTS_SMALL_BATCH_SIZE");
    cfg.enable_telemetry = telemetry_enabled_env();
    return cfg;
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

class AdversarialMctsRunner
{
public:
    explicit AdversarialMctsRunner(MctsOptions options)
        : options_(std::move(options))
        , origin_fen_(options_.request.fen)
        , small_vocab_(small_policy_vocabulary())
        , small_vocab_index_(build_vocab_index(small_vocab_))
        , leela_vocab_(leela_policy_vocabulary())
        , leela_vocab_index_(build_vocab_index(leela_vocab_))
        , small_batch_config_(make_small_batch_config(options_))
        , small_batch_runner_(small_batch_config_)
        , small_batch_flush_threshold_(static_cast<size_t>(small_batch_config_.max_batch_size))
        , leela_batch_config_(make_leela_batch_config(options_))
        , leela_batch_runner_(leela_batch_config_)
        , leela_batch_flush_threshold_(static_cast<size_t>(leela_batch_config_.max_batch_size))
        , rng_(std::random_device{}())
    {
        const chess::Board board = apply_history(options_.request);
        attacker_color_ = board.sideToMove();
        tree_.add_root(board, options_.request.history);
        const size_t expected_nodes = 1 + static_cast<size_t>(options_.simulations) * 3;
        tree_.reserve(expected_nodes);
        path_buffer_.reserve(64);
    }

    MctsSummary run()
    {
        ensure_attacker_root_evaluated();
        for (int i = 0; i < options_.simulations; ++i)
        {
            run_simulation();
        }
        refine_wdl_recursive();
        MctsSummary summary = summarize();
        if (small_batch_config_.enable_telemetry)
        {
            auto stats = small_batch_runner_.statistics();
            if (stats.batches > 0)
            {
                std::clog << "[telemetry][adversarial][small] batches=" << stats.batches
                          << " positions=" << stats.positions
                          << " infer_ms=" << stats.total_inference_ms << '\n';
            }
        }
        if (leela_batch_config_.enable_telemetry)
        {
            auto stats = leela_batch_runner_.statistics();
            if (stats.batches > 0)
            {
                std::clog << "[telemetry][adversarial][leela] batches=" << stats.batches
                          << " positions=" << stats.positions
                          << " infer_ms=" << stats.total_inference_ms << '\n';
            }
        }
        return summary;
    }

private:
    void ensure_attacker_root_evaluated()
    {
        TreeNode& root = tree_.nodes.front();
        if (!root.expanded)
        {
            evaluate_attacker_node(0);
        }
    }

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

            const bool attacker_turn = node.board.sideToMove() == attacker_color_;

            if (!node.expanded)
            {
                if (attacker_turn)
                {
                    float value = evaluate_attacker_node(node_index);
                    backpropagate(path, value);
                    return;
                }
                else
                {
                    evaluate_victim_node(node_index);
                }
            }

            if (attacker_turn)
            {
                const int child_index = select_attacker_child(node_index);
                if (child_index < 0)
                {
                    float value = node.value_prior;
                    backpropagate(path, value);
                    return;
                }
                node_index = child_index;
                path.push_back(node_index);
            }
            else
            {
                const int child_index = sample_victim_child(node_index);
                if (child_index < 0)
                {
                    float value = node.value_prior;
                    backpropagate(path, value);
                    return;
                }
                node_index = child_index;
                path.push_back(node_index);
            }
        }
    }

    float evaluate_attacker_node(int node_index)
    {
        TreeNode& node = tree_.nodes[node_index];
        if (node.expanded && !node.children.empty())
        {
            return node.value_prior;
        }

        auto encoded_ptr = acquire_small_encoding(node.history, node.board);
        auto ticket = small_batch_runner_.enqueue(*encoded_ptr);
        if (small_batch_runner_.pending() >= small_batch_flush_threshold_)
        {
            small_batch_runner_.process_pending();
        }
        while (!ticket.ready())
        {
            small_batch_runner_.process_pending();
        }
        SmallPolicyValue eval = ticket.get();
        if (eval.policy_logits.size() != chessrl::small_policy_size())
        {
            throw std::runtime_error("Single-batch small inference returned unexpected policy size.");
        }

        node.wdl = orient_small_wdl(node.board, eval.wdl);
        node.value_prior = node.wdl[0] - node.wdl[2];

        std::vector<MovePrior> priors = compute_small_priors(node.board, eval);
        if (priors.empty())
        {
            node.expanded = true;
            node.terminal = true;
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

    void evaluate_victim_node(int node_index)
    {
        TreeNode& node = tree_.nodes[node_index];
        if (node.expanded && !node.children.empty())
        {
            return;
        }

        auto small_encoded_ptr = acquire_small_encoding(node.history, node.board);
        auto small_ticket = small_batch_runner_.enqueue(*small_encoded_ptr);
        if (small_batch_runner_.pending() >= small_batch_flush_threshold_)
        {
            small_batch_runner_.process_pending();
        }
        while (!small_ticket.ready())
        {
            small_batch_runner_.process_pending();
        }
        SmallPolicyValue small_eval = small_ticket.get();
        if (small_eval.policy_logits.size() != chessrl::small_policy_size())
        {
            throw std::runtime_error("Single-batch small inference returned unexpected policy size.");
        }

        const std::vector<std::string> history = history_to_vector(node.history);
        auto leela_encoded_ptr = acquire_leela_encoding(node.history, history);
        auto leela_ticket = leela_batch_runner_.enqueue(*leela_encoded_ptr);
        if (leela_batch_runner_.pending() >= leela_batch_flush_threshold_)
        {
            leela_batch_runner_.process_pending();
        }
        while (!leela_ticket.ready())
        {
            leela_batch_runner_.process_pending();
        }
        LeelaPolicyValue leela_eval = leela_ticket.get();
        if (leela_eval.policy_logits.size() != chessrl::leela_policy_size())
        {
            throw std::runtime_error("Single-batch Leela inference returned unexpected policy size.");
        }

        node.wdl = orient_small_wdl(node.board, small_eval.wdl);
        node.value_prior = node.wdl[0] - node.wdl[2];

        std::vector<MovePrior> priors = compute_leela_priors(node.board, leela_eval);
        if (priors.empty())
        {
            node.expanded = true;
            node.terminal = true;
            return;
        }

        const chess::Board parent_board = node.board;
        for (const auto& entry : priors)
        {
            chess::Board child_board = parent_board;
            child_board.makeMove(entry.move);
            tree_.add_child(node_index, entry.move, entry.prior, child_board);
        }

        TreeNode& refreshed = tree_.nodes[node_index];
        refreshed.expanded = true;
    }

    int select_attacker_child(int node_index) const
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

    int sample_victim_child(int node_index)
    {
        TreeNode& node = tree_.nodes[node_index];
        if (node.children.empty())
        {
            return -1;
        }

        std::vector<double> weights;
        weights.reserve(node.child_priors.size());
        double sum = 0.0;
        for (float prior : node.child_priors)
        {
            const double w = std::max(0.0F, prior);
            weights.push_back(w);
            sum += w;
        }

        if (sum <= 0.0)
        {
            const double uniform = 1.0 / static_cast<double>(weights.size());
            std::fill(weights.begin(), weights.end(), uniform);
        }

        std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
        const size_t choice = distribution(rng_);
        return node.children[choice];
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

    std::vector<MovePrior> compute_small_priors(const chess::Board& board, const SmallPolicyValue& eval) const
    {
        chess::Movelist legal_moves;
        chess::movegen::legalmoves(legal_moves, board);

        std::vector<MovePrior> entries;
        entries.reserve(legal_moves.size());

        float max_logit = -std::numeric_limits<float>::infinity();
        int mapped_count = 0;

        for (const auto& move : legal_moves)
        {
            MovePrior entry;
            entry.move = move;
            entry.logit = -std::numeric_limits<float>::infinity();

            std::string key = chess::uci::moveToUci(move, board.chess960());
            if (!key.empty() && key.back() == 'n')
            {
                key.pop_back();
            }
            auto it = small_vocab_index_.find(key);
            if (it != small_vocab_index_.end())
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

    std::vector<MovePrior> compute_leela_priors(const chess::Board& board, const LeelaPolicyValue& eval) const
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

            auto it = leela_vocab_index_.find(key);
            if (it != leela_vocab_index_.end())
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

        const double temperature = std::max(1e-5, static_cast<double>(options_.temperature));
        const double inv_temp = 1.0 / temperature;
        double sum = 0.0;
        for (auto& entry : entries)
        {
            if (entry.policy_index < 0)
            {
                entry.prior = 0.0F;
                continue;
            }
            const double scaled = static_cast<double>(entry.logit - max_logit) * inv_temp;
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

    void refine_wdl_recursive()
    {
        // Post-order traversal to refine WDL from bottom up
        std::vector<bool> visited(tree_.nodes.size(), false);
        refine_wdl_node(0, visited);
    }

    std::array<float, 3> refine_wdl_node(int node_index, std::vector<bool>& visited)
    {
        TreeNode& node = tree_.nodes[node_index];
        
        // If already visited, return current refined WDL
        if (visited[static_cast<size_t>(node_index)])
        {
            return node.wdl;
        }
        
        visited[static_cast<size_t>(node_index)] = true;
        
        // For terminal nodes or nodes without children, use original WDL
        if (node.terminal || node.children.empty())
        {
            return node.wdl;
        }
        
        // Recursively refine children first, then aggregate
        std::array<float, 3> wdl_sum{0.0F, 0.0F, 0.0F};
        int total_visits = 0;
        
        for (const int child_index : node.children)
        {
            if (child_index < 0 || child_index >= static_cast<int>(tree_.nodes.size()))
            {
                continue;
            }
            
            const TreeNode& child = tree_.nodes[child_index];
            
            // Only aggregate if child has been visited
            if (child.visit_count > 0)
            {
                // Recursively refine child's WDL
                std::array<float, 3> child_refined_wdl = refine_wdl_node(child_index, visited);
                
                // Child's WDL is from opponent's perspective, flip it for parent's perspective
                // [Win, Draw, Loss] from child -> [Loss, Draw, Win] for parent
                const float visits = static_cast<float>(child.visit_count);
                wdl_sum[0] += visits * child_refined_wdl[2];  // Opponent loss = our win
                wdl_sum[1] += visits * child_refined_wdl[1];  // Draw stays draw
                wdl_sum[2] += visits * child_refined_wdl[0];  // Opponent win = our loss
                
                total_visits += child.visit_count;
            }
        }
        
        // If we have aggregated visits, compute weighted average
        if (total_visits > 0)
        {
            const float inv_visits = 1.0F / static_cast<float>(total_visits);
            node.wdl[0] = wdl_sum[0] * inv_visits;
            node.wdl[1] = wdl_sum[1] * inv_visits;
            node.wdl[2] = wdl_sum[2] * inv_visits;
        }
        // Otherwise keep original WDL from neural network
        
        return node.wdl;
    }

    MctsSummary summarize() const
    {
        const TreeNode& root = tree_.nodes.front();

        struct AggregatedMove
        {
            std::string uci;
            int visit_total{0};
            float prior_total{0.0F};
            float value_total{0.0F};
            int representative_child{-1};
            int occurrence_count{0};
        };

        std::unordered_map<std::string, size_t> aggregate_index;
        std::vector<AggregatedMove> aggregated;
        aggregated.reserve(root.children.size());

        int total_visits = 0;
        float prior_sum_total = 0.0F;

        for (size_t i = 0; i < root.children.size(); ++i)
        {
            const int child_index = root.children[i];
            const TreeNode& child = tree_.nodes[child_index];
            const chess::Move move = root.child_moves[i];
            const std::string uci = chess::uci::moveToUci(move, root.board.chess960());
            const int visits = child.visit_count;
            const float prior = root.child_priors[i];

            auto [it, inserted] = aggregate_index.emplace(uci, aggregated.size());
            if (inserted)
            {
                AggregatedMove entry;
                entry.uci = uci;
                entry.visit_total = visits;
                entry.prior_total = prior;
                entry.value_total = child.value_sum;
                entry.representative_child = child_index;
                entry.occurrence_count = 1;
                aggregated.push_back(std::move(entry));
            }
            else
            {
                AggregatedMove& entry = aggregated[it->second];
                entry.visit_total += visits;
                entry.prior_total += prior;
                entry.value_total += child.value_sum;
                entry.occurrence_count += 1;
                if (entry.representative_child < 0 || child.visit_count > tree_.nodes[entry.representative_child].visit_count)
                {
                    entry.representative_child = child_index;
        }
            }

            total_visits += visits;
            prior_sum_total += prior;
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

        float prior_sum_average = 0.0F;
        for (const auto& entry : aggregated)
        {
            if (entry.occurrence_count > 0)
            {
                prior_sum_average += entry.prior_total / static_cast<float>(entry.occurrence_count);
            }
        }

        std::vector<std::pair<std::string, float>> distribution;
        distribution.reserve(aggregated.size());

        if (visit_sum <= 0.0F)
        {
            if (prior_sum_average <= 0.0F)
            {
                const float uniform = 1.0F / static_cast<float>(aggregated.size());
                for (const auto& entry : aggregated)
                {
                    distribution.emplace_back(entry.uci, uniform);
                }
            }
            else
            {
                for (const auto& entry : aggregated)
                {
                    const float avg_prior = entry.occurrence_count > 0 ? entry.prior_total / static_cast<float>(entry.occurrence_count) : 0.0F;
                    distribution.emplace_back(entry.uci, avg_prior / prior_sum_average);
                }
            }
        }
        else
        {
            if (prior_sum_average <= 0.0F)
            {
                const float uniform = 1.0F / static_cast<float>(aggregated.size());
                for (const auto& entry : aggregated)
                {
                    distribution.emplace_back(entry.uci, uniform);
                }
            }
            else
            {
                for (const auto& entry : aggregated)
            {
                    const float avg_prior = entry.occurrence_count > 0 ? entry.prior_total / static_cast<float>(entry.occurrence_count) : 0.0F;
                    distribution.emplace_back(entry.uci, avg_prior / prior_sum_average);
                }
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
            for (size_t i = 0; i < aggregated.size(); ++i)
            {
                const float visits = static_cast<float>(aggregated[i].visit_total);
                if (visits > best_score)
                {
                    best_score = visits;
                    best_index = i;
                }
            }
        }
        else
        {
            for (size_t i = 0; i < aggregated.size(); ++i)
            {
                const float prior = aggregated[i].prior_total;
                if (prior > best_score)
                {
                    best_score = prior;
                    best_index = i;
                }
            }
        }

        const std::string action = distribution[best_index].first;

        MctsSummary summary;
        summary.action = action;
        summary.policy = std::move(sorted);
        summary.value = root.wdl;
        summary.rollouts.reserve(aggregated.size());
        for (const auto& entry : aggregated)
        {
            RolloutDetail detail;
            detail.root_move = entry.uci;
            detail.visit_count = entry.visit_total;
            if (visit_sum > 0.0F)
            {
                detail.visit_fraction = static_cast<float>(entry.visit_total) / visit_sum;
            }
            else if (prior_sum_average > 0.0F)
            {
                const float avg_prior
                    = entry.occurrence_count > 0 ? entry.prior_total / static_cast<float>(entry.occurrence_count) : 0.0F;
                detail.visit_fraction = avg_prior / prior_sum_average;
            }
            detail.prior
                = entry.occurrence_count > 0 ? entry.prior_total / static_cast<float>(entry.occurrence_count) : 0.0F;
            detail.average_value
                = entry.visit_total > 0 ? -entry.value_total / static_cast<float>(entry.visit_total) : 0.0F;
            if (entry.representative_child >= 0)
            {
                detail.continuation = principal_variation(tree_, entry.representative_child);
            }
            summary.rollouts.push_back(std::move(detail));
        }
        return summary;
    }

    std::shared_ptr<const single::SmallEncodedPosition> acquire_small_encoding(
        const HistoryHandle& history,
        const chess::Board& board)
    {
        if (!history)
        {
            return std::make_shared<single::SmallEncodedPosition>(single::encode_small_position(board));
        }
        const HistoryEntry* key = history.get();
        std::lock_guard<std::mutex> lock(small_cache_mutex_);
        auto it = small_encoding_cache_.find(key);
        if (it != small_encoding_cache_.end())
        {
            if (auto existing = it->second.lock())
            {
                return existing;
            }
        }
        auto fresh = std::make_shared<single::SmallEncodedPosition>(single::encode_small_position(board));
        small_encoding_cache_[key] = fresh;
        return fresh;
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

    MctsOptions options_;
    std::string origin_fen_;
    SearchTree tree_;
    chess::Color attacker_color_{chess::Color::WHITE};
    std::vector<std::string> small_vocab_;
    std::unordered_map<std::string, int> small_vocab_index_;
    std::vector<std::string> leela_vocab_;
    std::unordered_map<std::string, int> leela_vocab_index_;
    single::SmallBatchConfig small_batch_config_;
    single::SmallBatchRunner small_batch_runner_;
    size_t small_batch_flush_threshold_{0};
    single::LeelaBatchConfig leela_batch_config_;
    single::LeelaBatchRunner leela_batch_runner_;
    size_t leela_batch_flush_threshold_{0};
    mutable std::mutex small_cache_mutex_;
    std::unordered_map<const HistoryEntry*, std::weak_ptr<const single::SmallEncodedPosition>> small_encoding_cache_;
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

MctsSummary run_adversarial_mcts(MctsOptions options)
{
    AdversarialMctsRunner runner(std::move(options));
    return runner.run();
}

#ifndef CHESSRL_MCTS_NO_STANDALONE
int main(int argc, char** argv)
{
    try
    {
        auto options = chessrl::mcts::parse_cli(argc, argv);
        auto summary = chessrl::mcts::run_adversarial_mcts(std::move(options));

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



