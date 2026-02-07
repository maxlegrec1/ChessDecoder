#include "mcts/common.hpp"

#include <algorithm>
#include <cctype>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace chessrl::mcts
{

struct HistoryEntry
{
    std::shared_ptr<const HistoryEntry> parent;
    std::string move;
    int length{0};
};

namespace
{

HistoryHandle build_history_chain(const std::vector<std::string>& history)
{
    HistoryHandle tail;
    for (const auto& move : history)
    {
        auto entry = std::make_shared<HistoryEntry>();
        entry->parent = tail;
        entry->move = move;
        entry->length = tail ? tail->length + 1 : 1;
        tail = std::move(entry);
    }
    return tail;
}

bool starts_with_flag(const std::string& value)
{
    return value.rfind("--", 0) == 0;
}

bool parse_bool(const std::string& value, const std::string& name)
{
    const std::string lowered = [&]() {
        std::string tmp = value;
        std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return tmp;
    }();

    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on")
    {
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off")
    {
        return false;
    }
    throw std::invalid_argument("Failed to parse boolean value for " + name + ": " + value);
}

int parse_int(const std::string& value, const std::string& name)
{
    try
    {
        return std::stoi(value);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("Failed to parse integer value for " + name + ": " + value);
    }
}

float parse_float(const std::string& value, const std::string& name)
{
    try
    {
        return std::stof(value);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("Failed to parse float value for " + name + ": " + value);
    }
}

} // namespace

std::array<float, 3> orient_small_wdl(const chess::Board& board, const std::array<float, 3>& raw_wdl)
{
    if (board.sideToMove() == chess::Color::BLACK)
    {
        return raw_wdl;
    }

    return {raw_wdl[2], raw_wdl[1], raw_wdl[0]};
}

std::vector<std::string> principal_variation(const SearchTree& tree, int node_index)
{
    std::vector<std::string> line;
    if (node_index < 0 || node_index >= static_cast<int>(tree.nodes.size()))
    {
        return line;
    }

    int current = node_index;
    while (current >= 0 && current < static_cast<int>(tree.nodes.size()))
    {
        const TreeNode& node = tree.nodes[current];
        if (node.children.empty())
        {
            break;
        }

        size_t best_index = 0;
        bool has_visits = false;
        float best_visits = -1.0F;

        for (size_t i = 0; i < node.children.size(); ++i)
        {
            const int child_index = node.children[i];
            if (child_index < 0 || child_index >= static_cast<int>(tree.nodes.size()))
            {
                continue;
            }
            const TreeNode& child = tree.nodes[child_index];
            const float visits = static_cast<float>(child.visit_count);
            if (visits > best_visits)
            {
                best_visits = visits;
                best_index = i;
                has_visits = visits > 0.0F;
            }
        }

        if (!has_visits)
        {
            float best_prior = -1.0F;
            bool found_prior = false;
            for (size_t i = 0; i < node.children.size(); ++i)
            {
                const float prior = node.child_priors[i];
                if (prior > best_prior)
                {
                    best_prior = prior;
                    best_index = i;
                    found_prior = true;
                }
            }
            if (!found_prior)
            {
                break;
            }
        }

        const chess::Move move = node.child_moves[best_index];
        line.push_back(chess::uci::moveToUci(move, tree.nodes[current].board.chess960()));
        current = node.children[best_index];
    }

    return line;
}

std::vector<PVNodeDetail> principal_variation_detailed(const SearchTree& tree, int node_index, int max_depth)
{
    std::vector<PVNodeDetail> line;
    if (node_index < 0 || node_index >= static_cast<int>(tree.nodes.size()))
    {
        return line;
    }

    int current = node_index;
    int depth = 0;
    while (current >= 0 && current < static_cast<int>(tree.nodes.size()) && depth < max_depth)
    {
        const TreeNode& node = tree.nodes[current];
        if (node.children.empty())
        {
            break;
        }

        size_t best_index = 0;
        bool has_visits = false;
        float best_visits = -1.0F;

        for (size_t i = 0; i < node.children.size(); ++i)
        {
            const int child_index = node.children[i];
            if (child_index < 0 || child_index >= static_cast<int>(tree.nodes.size()))
            {
                continue;
            }
            const TreeNode& child = tree.nodes[child_index];
            const float visits = static_cast<float>(child.visit_count);
            if (visits > best_visits)
            {
                best_visits = visits;
                best_index = i;
                has_visits = visits > 0.0F;
            }
        }

        if (!has_visits)
        {
            float best_prior = -1.0F;
            bool found_prior = false;
            for (size_t i = 0; i < node.children.size(); ++i)
            {
                const float prior = node.child_priors[i];
                if (prior > best_prior)
                {
                    best_prior = prior;
                    best_index = i;
                    found_prior = true;
                }
            }
            if (!found_prior)
            {
                break;
            }
        }

        const chess::Move move = node.child_moves[best_index];
        const std::string move_uci = chess::uci::moveToUci(move, node.board.chess960());

        PVNodeDetail detail;
        detail.fen = node.board.getFen();
        detail.move = move_uci;
        detail.wdl = node.wdl;
        detail.visit_count = node.visit_count;
        line.push_back(std::move(detail));

        current = node.children[best_index];
        ++depth;
    }

    return line;
}

std::vector<std::string> history_to_vector(const HistoryHandle& tail)
{
    if (!tail)
    {
        return {};
    }
    std::vector<std::string> moves(static_cast<size_t>(tail->length));
    auto current = tail;
    for (int index = tail->length - 1; index >= 0; --index)
    {
        moves[static_cast<size_t>(index)] = current->move;
        current = current->parent;
    }
    return moves;
}

MctsOptions parse_cli(int argc, char** argv)
{
    std::unordered_map<std::string, std::string> flags;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (!starts_with_flag(arg))
        {
            throw std::invalid_argument("Unexpected positional argument: " + arg);
        }

        std::string key = arg.substr(2);
        if (key.empty())
        {
            throw std::invalid_argument("Empty flag encountered.");
        }

        std::string value;
        if (i + 1 < argc && !starts_with_flag(argv[i + 1]))
        {
            value = argv[++i];
        }
        else
        {
            value = "true";
        }

        flags[key] = value;
    }

    auto lookup = [&](const std::string& name) -> std::optional<std::string> {
        auto it = flags.find(name);
        if (it != flags.end())
        {
            return it->second;
        }
        return std::nullopt;
    };

    MctsOptions options;

    if (auto fen = lookup("fen"))
    {
        options.request.fen = *fen;
    }
    else
    {
        throw std::invalid_argument("Missing required flag: --fen");
    }

    if (auto history = lookup("history"))
    {
        options.request.history = parse_history_argument(*history);
    }

    if (auto sims = lookup("simulations"))
    {
        options.simulations = parse_int(*sims, "--simulations");
    }
    if (auto cpuct = lookup("cpuct"))
    {
        options.cpuct = parse_float(*cpuct, "--cpuct");
    }
    if (auto temperature = lookup("temperature"))
    {
        options.temperature = parse_float(*temperature, "--temperature");
    }
    if (auto vloss = lookup("virtual-loss"))
    {
        options.virtual_loss = parse_float(*vloss, "--virtual-loss");
    }
    if (auto alpha = lookup("dirichlet-alpha"))
    {
        options.dirichlet_alpha = parse_float(*alpha, "--dirichlet-alpha");
    }
    if (auto epsilon = lookup("dirichlet-epsilon"))
    {
        options.dirichlet_epsilon = parse_float(*epsilon, "--dirichlet-epsilon");
    }
    if (auto use_dirichlet = lookup("use-dirichlet"))
    {
        options.use_dirichlet = parse_bool(*use_dirichlet, "--use-dirichlet");
    }
    if (auto small_engine = lookup("small-engine"))
    {
        options.small_engine_path = *small_engine;
    }
    if (auto leela_engine = lookup("leela-engine"))
    {
        options.leela_engine_path = *leela_engine;
    }

    if (options.simulations <= 0)
    {
        throw std::invalid_argument("--simulations must be positive");
    }
    if (options.cpuct <= 0.0F)
    {
        throw std::invalid_argument("--cpuct must be positive");
    }
    if (options.temperature < 0.0F)
    {
        throw std::invalid_argument("--temperature must be non-negative");
    }
    if (options.virtual_loss < 0.0F)
    {
        throw std::invalid_argument("--virtual-loss must be non-negative");
    }
    if (options.dirichlet_alpha <= 0.0F)
    {
        throw std::invalid_argument("--dirichlet-alpha must be positive");
    }
    if (options.dirichlet_epsilon < 0.0F || options.dirichlet_epsilon > 1.0F)
    {
        throw std::invalid_argument("--dirichlet-epsilon must be between 0 and 1");
    }

    return options;
}

std::vector<std::string> parse_history_argument(const std::string& raw)
{
    std::vector<std::string> moves;
    std::stringstream ss(raw);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c) { return std::isspace(c) != 0; }), token.end());
        if (!token.empty())
        {
            moves.push_back(token);
        }
    }
    return moves;
}

chess::Board apply_history(const SearchInput& input)
{
    const std::string fenToUse = input.fen.empty() ? std::string(chess::constants::STARTPOS) : input.fen;
    chess::Board board(fenToUse);

    for (const auto& moveStr : input.history)
    {
        chess::Move move = chess::uci::uciToMove(board, moveStr);
        if (move == chess::Move::NO_MOVE)
        {
            throw std::runtime_error("Failed to apply move from history: " + moveStr);
        }
        board.makeMove(move);
    }

    return board;
}

int SearchTree::add_root(const chess::Board& board, const std::vector<std::string>& history)
{
    nodes.clear();
    TreeNode root;
    root.board = board;
    root.terminal = root.board.isGameOver().second != chess::GameResult::NONE;
    root.prior = 1.0F;
    root.history = build_history_chain(history);

    nodes.push_back(std::move(root));
    return 0;
}

void SearchTree::reserve(size_t count)
{
    nodes.reserve(count);
}

int SearchTree::add_child(
    int parent_index,
    chess::Move move,
    float prior,
    const chess::Board& child_board)
{
    if (parent_index < 0 || parent_index >= static_cast<int>(nodes.size()))
    {
        throw std::out_of_range("Parent index out of range in SearchTree::add_child");
    }

    TreeNode node;
    node.parent = parent_index;
    node.move_from_parent = move;
    node.prior = prior;
    node.board = child_board;
    node.terminal = node.board.isGameOver().second != chess::GameResult::NONE;
    if (node.terminal)
    {
        // Initialize terminal node WDL from the perspective of side to move on this node
        const auto result = node.board.isGameOver().second;
        switch (result)
        {
        case chess::GameResult::WIN:
            node.wdl = {1.0F, 0.0F, 0.0F};
            break;
        case chess::GameResult::LOSE:
            node.wdl = {0.0F, 0.0F, 1.0F};
            break;
        case chess::GameResult::DRAW:
            node.wdl = {0.0F, 1.0F, 0.0F};
            break;
        default:
            break;
        }
    }
    node.history = nodes[parent_index].history;

    const bool chess960 = nodes[parent_index].board.chess960();
    const std::string move_uci = chess::uci::moveToUci(move, chess960);
    if (!move_uci.empty())
    {
        auto entry = std::make_shared<HistoryEntry>();
        entry->parent = node.history;
        entry->move = move_uci;
        entry->length = node.history ? node.history->length + 1 : 1;
        node.history = std::move(entry);
    }

    nodes.push_back(std::move(node));
    const int child_index = static_cast<int>(nodes.size()) - 1;

    TreeNode& parent = nodes[parent_index];
    parent.children.push_back(child_index);
    parent.child_moves.push_back(move);
    parent.child_priors.push_back(prior);

    return child_index;
}

} // namespace chessrl::mcts


