#include "inference.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cctype>

#include "chess-library/include/chess.hpp"
#include "cuda_fp16.h"

namespace chessrl
{
namespace
{

constexpr int kSmallBoardSize = 8;
constexpr int kSmallFeaturePlanes = 19;
constexpr int kSmallPolicySize = 1929;
constexpr int kSmallBatchSize = 1024;
constexpr int kSmallValueSize = 3;

constexpr int kLeelaBoardSize = 8;
constexpr int kLeelaFeaturePlanes = 112;
constexpr int kLeelaPolicySize = 1858;
constexpr int kLeelaBatchSize = 1024;
constexpr int kLeelaValueSize = 3;
constexpr int kLeelaHistoryLength = 8;

constexpr float kIllegalScore = -1e9F;

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

size_t getElementSize(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        throw std::runtime_error("Unsupported TensorRT data type");
    }
}

size_t volume(const nvinfer1::Dims& dims)
{
    return std::accumulate(
        dims.d, dims.d + dims.nbDims, static_cast<size_t>(1), [](size_t a, int64_t b) { return a * static_cast<size_t>(b); });
}

class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t bytes)
    {
        if (cudaMalloc(&mPtr, bytes) != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed.");
        }
    }

    ~DeviceBuffer()
    {
        if (mPtr)
        {
            cudaFree(mPtr);
        }
    }

    void* data() const noexcept { return mPtr; }

private:
    void* mPtr{nullptr};
};

class CudaStream
{
public:
    CudaStream()
    {
        if (cudaStreamCreate(&mStream) != cudaSuccess)
        {
            throw std::runtime_error("Failed to create CUDA stream.");
        }
    }

    ~CudaStream()
    {
        if (mStream)
        {
            cudaStreamDestroy(mStream);
        }
    }

    cudaStream_t get() const noexcept { return mStream; }

private:
    cudaStream_t mStream{};
};

std::string resolveEnginePath(const std::string& engine_path, const std::string& default_name)
{
    std::string resolved = engine_path.empty() ? default_name : engine_path;
    try
    {
        resolved = std::filesystem::absolute(resolved).string();
    }
    catch (...)
    {
    }
    return resolved;
}

namespace small_impl
{

struct RawOutputs
{
    std::vector<float> policy;
    std::vector<float> wdl;
};

class EngineContext
{
public:
    explicit EngineContext(std::string path)
        : enginePath(std::move(path))
    {
        std::ifstream engineFile(enginePath, std::ios::binary);
        if (!engineFile)
        {
            throw std::runtime_error(std::string("Failed to open engine file: ") + enginePath);
        }

        engineFile.seekg(0, std::ifstream::end);
        const auto fsize = engineFile.tellg();
        engineFile.seekg(0, std::ifstream::beg);

        std::vector<char> engineData(static_cast<size_t>(fsize));
        engineFile.read(engineData.data(), fsize);

        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime)
        {
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }

        engine.reset(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!engine)
        {
            throw std::runtime_error("Failed to deserialize TensorRT engine.");
        }

        const int nbTensors = engine->getNbIOTensors();
        std::vector<std::string> outputNames;
        outputNames.reserve(nbTensors);

        for (int i = 0; i < nbTensors; ++i)
        {
            const char* name = engine->getIOTensorName(i);
            const auto mode = engine->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT)
            {
                inputName = name;
            }
            else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
            {
                outputNames.emplace_back(name);
            }
        }

        for (const auto& name : outputNames)
        {
            const auto dims = engine->getTensorShape(name.c_str());
            const size_t total = volume(dims);
            if (total == static_cast<size_t>(kSmallBatchSize) * kSmallPolicySize && policyOutputName.empty())
            {
                policyOutputName = name;
            }
            else if (total == static_cast<size_t>(kSmallBatchSize) * kSmallValueSize && valueOutputName.empty())
            {
                valueOutputName = name;
            }
        }

        if (policyOutputName.empty() && !outputNames.empty())
        {
            policyOutputName = outputNames.front();
        }
        if (valueOutputName.empty() && outputNames.size() >= 2)
        {
            valueOutputName = outputNames[1];
        }

        if (inputName.empty() || policyOutputName.empty() || valueOutputName.empty())
        {
            throw std::runtime_error("Failed to determine input/output tensor names for small model.");
        }
    }

    std::string enginePath;
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::string inputName;
    std::string policyOutputName;
    std::string valueOutputName;
};

EngineContext& getEngineContext(const std::string& engine_path)
{
    static auto* mutex = new std::mutex();
    static auto* contexts = new std::unordered_map<std::string, std::unique_ptr<EngineContext>>();

    std::string resolvedPath = resolveEnginePath(engine_path, "model.trt");

    std::lock_guard<std::mutex> lock(*mutex);
    auto it = contexts->find(resolvedPath);
    if (it == contexts->end())
    {
        auto ctx = std::make_unique<EngineContext>(resolvedPath);
        auto [inserted, _] = contexts->emplace(resolvedPath, std::move(ctx));
        return *inserted->second;
    }
    return *it->second;
}

class PolicyVocab
{
public:
    PolicyVocab()
    {
        std::ifstream file(kVocabPath);
        if (!file)
        {
            throw std::runtime_error(std::string("Failed to open policy vocab file: ") + kVocabPath);
        }

        std::string line;
        while (std::getline(file, line))
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            tokens.emplace_back(line);
            if (!line.empty())
            {
                toIndex[line] = static_cast<int>(tokens.size() - 1);
            }
        }

        if (tokens.size() != kSmallPolicySize)
        {
            throw std::runtime_error("Unexpected small policy vocab size: " + std::to_string(tokens.size()));
        }
    }

    const std::string& at(int idx) const { return tokens.at(static_cast<size_t>(idx)); }

    std::optional<int> indexOf(const std::string& token) const
    {
        auto it = toIndex.find(token);
        if (it == toIndex.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

    const std::vector<std::string>& tokensRef() const { return tokens; }

private:
    static constexpr const char* kVocabPath = "cpp/policy_index.txt";
    std::vector<std::string> tokens;
    std::unordered_map<std::string, int> toIndex;
};

PolicyVocab& getPolicyVocab()
{
    static PolicyVocab vocab;
    return vocab;
}

size_t encodeIndex(int batch, int row, int col, int plane)
{
    return (((static_cast<size_t>(batch) * kSmallBoardSize + static_cast<size_t>(row)) * kSmallBoardSize
             + static_cast<size_t>(col))
            * kSmallFeaturePlanes)
        + static_cast<size_t>(plane);
}

struct EncodedBatch
{
    std::vector<float> features;
    std::vector<chess::Board> boards;
};

EncodedBatch encodeBatch(const std::vector<std::string>& fens)
{
    if (static_cast<int>(fens.size()) != kSmallBatchSize)
    {
        throw std::runtime_error("TensorRT engine expects batch size " + std::to_string(kSmallBatchSize)
                                 + ", but got " + std::to_string(fens.size()));
    }

    EncodedBatch encoded;
    encoded.features.assign(fens.size() * kSmallBoardSize * kSmallBoardSize * kSmallFeaturePlanes, 0.0F);
    encoded.boards.reserve(fens.size());

    for (size_t idx = 0; idx < fens.size(); ++idx)
    {
        chess::Board board(fens[idx]);
        encoded.boards.push_back(board);

        const auto fillPlane = [&](int plane, float value) {
            for (int r = 0; r < kSmallBoardSize; ++r)
            {
                for (int c = 0; c < kSmallBoardSize; ++c)
                {
                    encoded.features[encodeIndex(static_cast<int>(idx), r, c, plane)] = value;
                }
            }
        };

        const std::array<std::pair<chess::PieceType, int>, 6> typeOrder{{
            {chess::PieceType::PAWN, 0},
            {chess::PieceType::KNIGHT, 1},
            {chess::PieceType::BISHOP, 2},
            {chess::PieceType::ROOK, 3},
            {chess::PieceType::QUEEN, 4},
            {chess::PieceType::KING, 5},
        }};

        for (auto color : {chess::Color::WHITE, chess::Color::BLACK})
        {
            const int base = color == chess::Color::WHITE ? 0 : 6;
            for (const auto& [pieceType, offset] : typeOrder)
            {
                chess::Bitboard bb = board.pieces(pieceType, color);
                while (bb)
                {
                    const int squareIndex = static_cast<int>(bb.pop());
                    const int rank = squareIndex / kSmallBoardSize;
                    const int file = squareIndex % kSmallBoardSize;
                    const int plane = base + offset;
                    encoded.features[encodeIndex(
                        static_cast<int>(idx), kSmallBoardSize - 1 - rank, file, plane)]
                        = 1.0F;
                }
            }
        }

        fillPlane(12, board.sideToMove() == chess::Color::WHITE ? 1.0F : 0.0F);

        const auto enPassant = board.enpassantSq();
        if (enPassant.is_valid())
        {
            const int idxSq = enPassant.index();
            const int rank = idxSq / kSmallBoardSize;
            const int file = idxSq % kSmallBoardSize;
            encoded.features[encodeIndex(
                static_cast<int>(idx), kSmallBoardSize - 1 - rank, file, 13)]
                = 1.0F;
        }

        const auto castling = board.castlingRights();
        fillPlane(
            14,
            castling.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE) ? 1.0F : 0.0F);
        fillPlane(
            15,
            castling.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0F : 0.0F);
        fillPlane(
            16,
            castling.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE) ? 1.0F : 0.0F);
        fillPlane(
            17,
            castling.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0F : 0.0F);

        const float fiftyMove = std::min(board.halfMoveClock() / 100.0F, 1.0F);
        fillPlane(18, fiftyMove);
    }

    return encoded;
}

RawOutputs runInference(EngineContext& ctx, const std::vector<float>& input)
{
    auto& engine = *ctx.engine;

    const auto inputDims = engine.getTensorShape(ctx.inputName.c_str());
    const auto policyDims = engine.getTensorShape(ctx.policyOutputName.c_str());
    const auto valueDims = engine.getTensorShape(ctx.valueOutputName.c_str());

    const auto expectedInput = nvinfer1::Dims4{kSmallBatchSize, kSmallBoardSize, kSmallBoardSize, kSmallFeaturePlanes};
    if (volume(inputDims) != volume(expectedInput))
    {
        throw std::runtime_error("Small-model input tensor has unexpected size.");
    }

    const auto expectedPolicy = nvinfer1::Dims2{kSmallBatchSize, kSmallPolicySize};
    if (volume(policyDims) != volume(expectedPolicy))
    {
        throw std::runtime_error("Small-model policy tensor has unexpected size.");
    }

    const auto expectedValue = nvinfer1::Dims2{kSmallBatchSize, kSmallValueSize};
    if (volume(valueDims) != volume(expectedValue))
    {
        throw std::runtime_error("Small-model value tensor has unexpected size.");
    }

    const auto inputType = engine.getTensorDataType(ctx.inputName.c_str());
    const auto policyType = engine.getTensorDataType(ctx.policyOutputName.c_str());
    const auto valueType = engine.getTensorDataType(ctx.valueOutputName.c_str());

    std::unique_ptr<nvinfer1::IExecutionContext> executionContext(engine.createExecutionContext());
    if (!executionContext)
    {
        throw std::runtime_error("Failed to create small-model execution context.");
    }

    if (!executionContext->setInputShape(ctx.inputName.c_str(), expectedInput))
    {
        throw std::runtime_error("Failed to set small-model input shape.");
    }

    DeviceBuffer inputDevice(volume(inputDims) * getElementSize(inputType));
    DeviceBuffer policyDevice(volume(policyDims) * getElementSize(policyType));
    DeviceBuffer valueDevice(volume(valueDims) * getElementSize(valueType));
    CudaStream stream;

    std::vector<__half> inputHalf;
    std::vector<__half> policyHalf;
    std::vector<__half> valueHalf;

    const void* inputHostData = nullptr;
    size_t inputBytes = 0;

    if (inputType == nvinfer1::DataType::kFLOAT)
    {
        inputHostData = input.data();
        inputBytes = input.size() * sizeof(float);
    }
    else if (inputType == nvinfer1::DataType::kHALF)
    {
        inputHalf.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i)
        {
            inputHalf[i] = __float2half(input[i]);
        }
        inputHostData = inputHalf.data();
        inputBytes = inputHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported small-model input tensor type.");
    }

    if (cudaMemcpyAsync(inputDevice.data(), inputHostData, inputBytes, cudaMemcpyHostToDevice, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (small H2D) failed.");
    }

    if (!executionContext->setTensorAddress(ctx.inputName.c_str(), inputDevice.data())
        || !executionContext->setTensorAddress(ctx.policyOutputName.c_str(), policyDevice.data())
        || !executionContext->setTensorAddress(ctx.valueOutputName.c_str(), valueDevice.data()))
    {
        throw std::runtime_error("Failed to set small-model tensor addresses.");
    }

    if (!executionContext->enqueueV3(stream.get()))
    {
        throw std::runtime_error("Small-model TensorRT enqueue failed.");
    }

    RawOutputs outputs;
    outputs.policy.resize(volume(policyDims));
    outputs.wdl.resize(volume(valueDims));

    void* policyHostData = nullptr;
    size_t policyBytes = 0;

    if (policyType == nvinfer1::DataType::kFLOAT)
    {
        policyHostData = outputs.policy.data();
        policyBytes = outputs.policy.size() * sizeof(float);
    }
    else if (policyType == nvinfer1::DataType::kHALF)
    {
        policyHalf.resize(outputs.policy.size());
        policyHostData = policyHalf.data();
        policyBytes = policyHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported small-model policy tensor type.");
    }

    void* valueHostData = nullptr;
    size_t valueBytes = 0;

    if (valueType == nvinfer1::DataType::kFLOAT)
    {
        valueHostData = outputs.wdl.data();
        valueBytes = outputs.wdl.size() * sizeof(float);
    }
    else if (valueType == nvinfer1::DataType::kHALF)
    {
        valueHalf.resize(outputs.wdl.size());
        valueHostData = valueHalf.data();
        valueBytes = valueHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported small-model value tensor type.");
    }

    if (cudaMemcpyAsync(policyHostData, policyDevice.data(), policyBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (small policy D2H) failed.");
    }

    if (cudaMemcpyAsync(valueHostData, valueDevice.data(), valueBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (small value D2H) failed.");
    }

    if (cudaStreamSynchronize(stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaStreamSynchronize (small) failed.");
    }

    if (policyType == nvinfer1::DataType::kHALF)
    {
        for (size_t i = 0; i < outputs.policy.size(); ++i)
        {
            outputs.policy[i] = __half2float(policyHalf[i]);
        }
    }

    if (valueType == nvinfer1::DataType::kHALF)
    {
        for (size_t i = 0; i < outputs.wdl.size(); ++i)
        {
            outputs.wdl[i] = __half2float(valueHalf[i]);
        }
    }

    return outputs;
}

std::vector<std::string> selectMoves(
    const std::vector<float>& logits, const std::vector<chess::Board>& boards, float temperature)
{
    if (boards.size() * kSmallPolicySize != logits.size())
    {
        throw std::runtime_error("Mismatch between small-model logits and batch size.");
    }

    auto& vocab = getPolicyVocab();
    std::vector<std::string> moves;
    moves.reserve(boards.size());

    std::mt19937 rng(std::random_device{}());

    std::vector<float> row(kSmallPolicySize);
    std::vector<char> legalMask(kSmallPolicySize);
    std::vector<float> scaled(kSmallPolicySize);
    std::vector<double> probs(kSmallPolicySize);

    for (size_t idx = 0; idx < boards.size(); ++idx)
    {
        const float* rowStart = logits.data() + static_cast<ptrdiff_t>(idx * kSmallPolicySize);
        std::copy(rowStart, rowStart + kSmallPolicySize, row.begin());
        std::fill(legalMask.begin(), legalMask.end(), 0);

        chess::Movelist legalMoves;
        chess::movegen::legalmoves(legalMoves, boards[idx]);

        std::vector<int> legalIndices;
        legalIndices.reserve(legalMoves.size());

        for (const auto& move : legalMoves)
        {
            std::string uci = chess::uci::moveToUci(move, boards[idx].chess960());
            if (!uci.empty() && uci.back() == 'n')
            {
                uci.pop_back();
            }

            if (auto mapped = vocab.indexOf(uci))
            {
                if (!legalMask[*mapped])
                {
                    legalMask[*mapped] = 1;
                    legalIndices.push_back(*mapped);
                }
            }
        }

        if (!legalIndices.empty())
        {
            for (int i = 0; i < kSmallPolicySize; ++i)
            {
                if (!legalMask[i])
                {
                    row[i] = kIllegalScore;
                }
            }
        }

        int selectedIndex = -1;

        if (temperature <= 0.0F || legalIndices.size() <= 1)
        {
            selectedIndex = static_cast<int>(std::distance(row.begin(), std::max_element(row.begin(), row.end())));
        }
        else
        {
            float maxVal = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < kSmallPolicySize; ++i)
            {
                const float val = row[i] / temperature;
                scaled[i] = val;
                if (val > maxVal)
                {
                    maxVal = val;
                }
            }

            double sum = 0.0;
            for (int i = 0; i < kSmallPolicySize; ++i)
            {
                const double expVal = std::exp(static_cast<double>(scaled[i] - maxVal));
                probs[i] = expVal;
                sum += expVal;
            }

            if (sum <= 0.0)
            {
                selectedIndex = static_cast<int>(std::distance(row.begin(), std::max_element(row.begin(), row.end())));
            }
            else
            {
                for (double& p : probs)
                {
                    p /= sum;
                }
                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                selectedIndex = dist(rng);
            }
        }

        if (selectedIndex < 0 || selectedIndex >= kSmallPolicySize)
        {
            throw std::runtime_error("Failed to select small-model move.");
        }

        moves.push_back(vocab.at(selectedIndex));
    }

    return moves;
}

SmallInferenceResult runOnce(const std::vector<std::string>& fens, float temperature, EngineContext& ctx)
{
    using clock = std::chrono::high_resolution_clock;
    const auto totalStart = clock::now();

    const auto encodeStart = clock::now();
    EncodedBatch encoded = encodeBatch(fens);
    const auto encodeEnd = clock::now();

    const auto inferStart = clock::now();
    RawOutputs rawOutputs = runInference(ctx, encoded.features);
    const auto inferEnd = clock::now();

    const auto sampleStart = clock::now();
    std::vector<std::string> moves = selectMoves(rawOutputs.policy, encoded.boards, temperature);
    const auto sampleEnd = clock::now();

    if (rawOutputs.wdl.size() != encoded.boards.size() * static_cast<size_t>(kSmallValueSize))
    {
        throw std::runtime_error("Mismatch between small-model value outputs and batch size.");
    }

    std::vector<std::array<float, kSmallValueSize>> wdlVectors;
    wdlVectors.reserve(encoded.boards.size());
    for (size_t idx = 0; idx < encoded.boards.size(); ++idx)
    {
        std::array<float, kSmallValueSize> entry{};
        for (int j = 0; j < kSmallValueSize; ++j)
        {
            entry[static_cast<size_t>(j)]
                = rawOutputs.wdl[idx * static_cast<size_t>(kSmallValueSize) + static_cast<size_t>(j)];
        }
        wdlVectors.push_back(entry);
    }

    const auto totalEnd = clock::now();

    auto durationMs = [](auto start, auto end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    InferenceTimings timings{
        .encodeMs = durationMs(encodeStart, encodeEnd),
        .inferMs = durationMs(inferStart, inferEnd),
        .sampleMs = durationMs(sampleStart, sampleEnd),
        .totalMs = durationMs(totalStart, totalEnd),
    };

    return SmallInferenceResult{std::move(moves), std::move(wdlVectors), timings};
}

} // namespace small_impl

namespace leela_impl
{

struct RawOutputs
{
    std::vector<float> policy;
    std::vector<float> valueWinner;
    std::vector<float> valueQ;
};

class EngineContext
{
public:
    explicit EngineContext(std::string path)
        : enginePath(std::move(path))
    {
        std::ifstream engineFile(enginePath, std::ios::binary);
        if (!engineFile)
        {
            throw std::runtime_error(std::string("Failed to open engine file: ") + enginePath);
        }

        engineFile.seekg(0, std::ifstream::end);
        const auto fsize = engineFile.tellg();
        engineFile.seekg(0, std::ifstream::beg);

        std::vector<char> engineData(static_cast<size_t>(fsize));
        engineFile.read(engineData.data(), fsize);

        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime)
        {
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }

        engine.reset(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!engine)
        {
            throw std::runtime_error("Failed to deserialize TensorRT engine.");
        }

        const int nbTensors = engine->getNbIOTensors();
        std::vector<std::string> outputNames;
        outputNames.reserve(nbTensors);

        for (int i = 0; i < nbTensors; ++i)
        {
            const char* name = engine->getIOTensorName(i);
            const auto mode = engine->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT)
            {
                inputName = name;
            }
            else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
            {
                outputNames.emplace_back(name);
            }
        }

        for (const auto& name : outputNames)
        {
            const auto dims = engine->getTensorShape(name.c_str());
            const size_t total = volume(dims);
            if (total == static_cast<size_t>(kLeelaBatchSize) * kLeelaPolicySize && policyOutputName.empty())
            {
                policyOutputName = name;
            }
            else if (total == static_cast<size_t>(kLeelaBatchSize) * kLeelaValueSize)
            {
                if (valueWinnerOutputName.empty())
                {
                    valueWinnerOutputName = name;
                }
                else if (valueQOutputName.empty())
                {
                    valueQOutputName = name;
                }
            }
        }

        if (policyOutputName.empty() && !outputNames.empty())
        {
            policyOutputName = outputNames.front();
        }
        if (valueWinnerOutputName.empty() && outputNames.size() >= 2)
        {
            valueWinnerOutputName = outputNames[0] == policyOutputName ? outputNames[1] : outputNames[0];
        }
        if (valueQOutputName.empty() && outputNames.size() >= 3)
        {
            for (const auto& candidate : outputNames)
            {
                if (candidate != policyOutputName && candidate != valueWinnerOutputName)
                {
                    valueQOutputName = candidate;
                    break;
                }
            }
        }

        if (inputName.empty() || policyOutputName.empty() || valueWinnerOutputName.empty() || valueQOutputName.empty())
        {
            throw std::runtime_error("Failed to determine input/output tensor names for Leela model.");
        }
    }

    std::string enginePath;
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::string inputName;
    std::string policyOutputName;
    std::string valueWinnerOutputName;
    std::string valueQOutputName;
};

EngineContext& getEngineContext(const std::string& engine_path)
{
    static auto* mutex = new std::mutex();
    static auto* contexts = new std::unordered_map<std::string, std::unique_ptr<EngineContext>>();

    std::string resolvedPath = resolveEnginePath(engine_path, "leela.trt");

    std::lock_guard<std::mutex> lock(*mutex);
    auto it = contexts->find(resolvedPath);
    if (it == contexts->end())
    {
        auto ctx = std::make_unique<EngineContext>(resolvedPath);
        auto [inserted, _] = contexts->emplace(resolvedPath, std::move(ctx));
        return *inserted->second;
    }
    return *it->second;
}

class PolicyVocab
{
public:
    PolicyVocab()
    {
        std::ifstream file(kVocabPath);
        if (!file)
        {
            throw std::runtime_error(std::string("Failed to open Leela policy vocab file: ") + kVocabPath);
        }

        std::string line;
        while (std::getline(file, line))
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            tokens.emplace_back(line);
            if (!line.empty())
            {
                toIndex[line] = static_cast<int>(tokens.size() - 1);
            }
        }

        if (tokens.size() != kLeelaPolicySize)
        {
            throw std::runtime_error("Unexpected Leela policy vocab size: " + std::to_string(tokens.size()));
        }
    }

    const std::string& at(int idx) const { return tokens.at(static_cast<size_t>(idx)); }

    std::optional<int> indexOf(const std::string& token) const
    {
        auto it = toIndex.find(token);
        if (it == toIndex.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

    const std::vector<std::string>& tokensRef() const { return tokens; }

private:
    static constexpr const char* kVocabPath = "cpp/leela_policy_index.txt";
    std::vector<std::string> tokens;
    std::unordered_map<std::string, int> toIndex;
};

PolicyVocab& getPolicyVocab()
{
    static PolicyVocab vocab;
    return vocab;
}

std::vector<std::string> splitByChar(const std::string& input, char delimiter)
{
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string part;
    while (std::getline(ss, part, delimiter))
    {
        result.push_back(part);
    }
    return result;
}

std::vector<std::string> splitFen(const std::string& fen)
{
    std::vector<std::string> parts;
    std::istringstream ss(fen);
    std::string part;
    while (ss >> part)
    {
        parts.push_back(part);
    }
    return parts;
}

std::string reorderCastling(const std::string& rights)
{
    if (rights.empty() || rights == "-")
    {
        return "-";
    }

    std::string ordered;
    auto appendIfPresent = [&](char ch) {
        if (rights.find(ch) != std::string::npos)
        {
            ordered.push_back(ch);
        }
    };

    appendIfPresent('K');
    appendIfPresent('Q');
    appendIfPresent('k');
    appendIfPresent('q');

    return ordered.empty() ? std::string("-") : ordered;
}

std::string mirrorFen(const chess::Board& board)
{
    const std::string fen = board.getFen();
    auto fields = splitFen(fen);
    if (fields.size() < 4)
    {
        throw std::runtime_error("Invalid FEN encountered while mirroring board.");
    }

    auto ranks = splitByChar(fields[0], '/');
    std::reverse(ranks.begin(), ranks.end());
    for (auto& rank : ranks)
    {
        for (char& ch : rank)
        {
            if (std::isalpha(static_cast<unsigned char>(ch)))
            {
                if (std::islower(static_cast<unsigned char>(ch)))
                {
                    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
                }
                else
                {
                    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                }
            }
        }
    }

    std::ostringstream placementStream;
    for (size_t i = 0; i < ranks.size(); ++i)
    {
        if (i > 0)
        {
            placementStream << '/';
        }
        placementStream << ranks[i];
    }
    const std::string placement = placementStream.str();

    std::string stm = fields[1];
    if (stm == "w")
    {
        stm = "b";
    }
    else if (stm == "b")
    {
        stm = "w";
    }
    else
    {
        throw std::runtime_error("Invalid active color in FEN while mirroring board.");
    }

    std::string castling = fields[2];
    if (castling != "-")
    {
        std::string converted;
        converted.reserve(castling.size());
        for (char c : castling)
        {
            if (std::isalpha(static_cast<unsigned char>(c)))
            {
                if (std::islower(static_cast<unsigned char>(c)))
                {
                    converted.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
                }
                else
                {
                    converted.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                }
            }
        }
        castling = reorderCastling(converted);
    }

    std::string ep = fields[3];
    if (ep != "-")
    {
        if (ep.size() == 2 && ep[0] >= 'a' && ep[0] <= 'h' && ep[1] >= '1' && ep[1] <= '8')
        {
            const int rank = ep[1] - '0';
            const int mirroredRank = 9 - rank;
            ep[1] = static_cast<char>('0' + mirroredRank);
        }
        else
        {
            ep = "-";
        }
    }

    const std::string halfmove = fields.size() > 4 ? fields[4] : "0";
    const std::string fullmove = fields.size() > 5 ? fields[5] : "1";

    std::ostringstream os;
    os << placement << ' ' << stm << ' ' << castling << ' ' << ep << ' ' << halfmove << ' ' << fullmove;
    return os.str();
}

chess::Board mirrorBoard(const chess::Board& board)
{
    const std::string mirroredFen = mirrorFen(board);
    return chess::Board(mirroredFen);
}

std::string mirrorUciRanks(const std::string& move)
{
    if (move.size() < 4)
    {
        return move;
    }
    auto mirrorRankChar = [](char c) -> char {
        if (c < '1' || c > '8')
        {
            return c;
        }
        const int rank = c - '0';
        const int mirrored = 9 - rank;
        return static_cast<char>('0' + mirrored);
    };
    std::string mirrored = move;
    mirrored[1] = mirrorRankChar(mirrored[1]);
    mirrored[3] = mirrorRankChar(mirrored[3]);
    return mirrored;
}

std::string adjustCastling(const std::string& move, const std::string& rights)
{
    if (move == "e1h1" && rights.find('K') != std::string::npos)
    {
        return "e1g1";
    }
    if (move == "e1a1" && rights.find('Q') != std::string::npos)
    {
        return "e1c1";
    }
    if (move == "e8h8" && rights.find('k') != std::string::npos)
    {
        return "e8g8";
    }
    if (move == "e8a8" && rights.find('q') != std::string::npos)
    {
        return "e8c8";
    }
    return move;
}

std::string extractCastlingRights(const chess::Board& board)
{
    const auto fields = splitFen(board.getFen());
    if (fields.size() >= 3)
    {
        return fields[2];
    }
    return "-";
}

size_t encodeIndex(int batch, int plane, int row, int col)
{
    return ((((static_cast<size_t>(batch) * kLeelaFeaturePlanes) + static_cast<size_t>(plane))
             * kLeelaBoardSize)
            + static_cast<size_t>(row))
        * kLeelaBoardSize
        + static_cast<size_t>(col);
}

void writeConstantPlane(std::vector<float>& features, int batchIdx, int planeIdx, float value)
{
    if (value == 0.0F)
    {
        return;
    }
    for (int row = 0; row < kLeelaBoardSize; ++row)
    {
        for (int col = 0; col < kLeelaBoardSize; ++col)
        {
            features[encodeIndex(batchIdx, planeIdx, row, col)] = value;
        }
    }
}

void writePiecePlanes(const chess::Board& board, int batchIdx, int planeOffset, std::vector<float>& features)
{
    const std::array<chess::Color, 2> colors{chess::Color::WHITE, chess::Color::BLACK};
    const std::array<chess::PieceType, 6> pieceTypes{
        chess::PieceType::PAWN,
        chess::PieceType::KNIGHT,
        chess::PieceType::BISHOP,
        chess::PieceType::ROOK,
        chess::PieceType::QUEEN,
        chess::PieceType::KING};

    int plane = planeOffset;
    for (const auto color : colors)
    {
        for (const auto pieceType : pieceTypes)
        {
            chess::Bitboard bb = board.pieces(pieceType, color);
            while (bb)
            {
                const int squareIndex = static_cast<int>(bb.pop());
                const int rank = squareIndex / kLeelaBoardSize;
                const int file = squareIndex % kLeelaBoardSize;
                features[encodeIndex(batchIdx, plane, rank, file)] = 1.0F;
            }
            ++plane;
        }
    }
}

struct EncodedBatch
{
    std::vector<float> features;
    std::vector<chess::Board> originalBoards;
    std::vector<chess::Board> maskBoards;
    std::vector<bool> isBlackToMove;
    std::vector<std::string> castlingRights;
};

EncodedBatch encodeBatch(
    const std::vector<std::string>& originFens,
    const std::vector<std::vector<std::string>>& moveLists)
{
    if (originFens.size() != moveLists.size())
    {
        throw std::invalid_argument("originFens and moveLists must have the same length.");
    }
    if (static_cast<int>(originFens.size()) != kLeelaBatchSize)
    {
        throw std::runtime_error("TensorRT engine expects batch size " + std::to_string(kLeelaBatchSize)
                                 + ", but got " + std::to_string(originFens.size()));
    }

    EncodedBatch encoded;
    const size_t featureCount
        = originFens.size() * kLeelaFeaturePlanes * kLeelaBoardSize * kLeelaBoardSize;
    encoded.features.assign(featureCount, 0.0F);
    encoded.originalBoards.resize(originFens.size());
    encoded.maskBoards.resize(originFens.size());
    encoded.isBlackToMove.resize(originFens.size());
    encoded.castlingRights.resize(originFens.size());

    for (size_t idx = 0; idx < originFens.size(); ++idx)
    {
        const std::string fenToUse
            = originFens[idx].empty() ? std::string(chess::constants::STARTPOS) : originFens[idx];

        chess::Board board(fenToUse);
        std::vector<chess::Board> states;
        states.reserve(moveLists[idx].size() + 1);
        states.push_back(board);

        for (const auto& uci : moveLists[idx])
        {
            const chess::Move move = chess::uci::uciToMove(board, uci);
            if (move == chess::Move::NO_MOVE)
            {
                throw std::runtime_error("Failed to parse UCI move: " + uci);
            }
            board.makeMove(move);
            states.push_back(board);
        }

        const chess::Board& finalBoard = states.back();
        encoded.originalBoards[idx] = finalBoard;
        const bool isBlack = finalBoard.sideToMove() == chess::Color::BLACK;
        encoded.isBlackToMove[idx] = isBlack;
        encoded.castlingRights[idx] = extractCastlingRights(finalBoard);

        std::array<std::optional<chess::Board>, kLeelaHistoryLength> history{};
        for (int h = 0; h < kLeelaHistoryLength; ++h)
        {
            if (states.size() > static_cast<size_t>(h))
            {
                history[h] = states[states.size() - 1 - static_cast<size_t>(h)];
            }
        }

        auto featureHistory = history;
        if (isBlack)
        {
            for (auto& snapshot : featureHistory)
            {
                if (snapshot)
                {
                    snapshot = mirrorBoard(*snapshot);
                }
            }
        }

        for (int h = 0; h < kLeelaHistoryLength; ++h)
        {
            if (featureHistory[h])
            {
                const int planeOffset = h * 13;
                writePiecePlanes(*featureHistory[h], static_cast<int>(idx), planeOffset, encoded.features);
            }
        }

        const auto& currentSnapshot = featureHistory[0];
        if (!currentSnapshot)
        {
            throw std::runtime_error("Leela encoding expected current snapshot to be available.");
        }

        const auto castling = currentSnapshot->castlingRights();
        writeConstantPlane(
            encoded.features,
            static_cast<int>(idx),
            104,
            castling.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0F : 0.0F);
        writeConstantPlane(
            encoded.features,
            static_cast<int>(idx),
            105,
            castling.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE) ? 1.0F : 0.0F);
        writeConstantPlane(
            encoded.features,
            static_cast<int>(idx),
            106,
            castling.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0F : 0.0F);
        writeConstantPlane(
            encoded.features,
            static_cast<int>(idx),
            107,
            castling.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE) ? 1.0F : 0.0F);

        if (isBlack)
        {
            writeConstantPlane(encoded.features, static_cast<int>(idx), 108, 1.0F);
        }
        writeConstantPlane(encoded.features, static_cast<int>(idx), 111, 1.0F);

        encoded.maskBoards[idx] = isBlack ? mirrorBoard(finalBoard) : finalBoard;
    }

    return encoded;
}

RawOutputs runInference(EngineContext& ctx, const std::vector<float>& input)
{
    auto& engine = *ctx.engine;

    const auto inputDims = engine.getTensorShape(ctx.inputName.c_str());
    const auto policyDims = engine.getTensorShape(ctx.policyOutputName.c_str());
    const auto winnerDims = engine.getTensorShape(ctx.valueWinnerOutputName.c_str());
    const auto qDims = engine.getTensorShape(ctx.valueQOutputName.c_str());

    const auto expectedInput = nvinfer1::Dims4{kLeelaBatchSize, kLeelaFeaturePlanes, kLeelaBoardSize, kLeelaBoardSize};
    if (volume(inputDims) != volume(expectedInput))
    {
        throw std::runtime_error("Leela-model input tensor has unexpected size.");
    }

    const auto expectedPolicy = nvinfer1::Dims2{kLeelaBatchSize, kLeelaPolicySize};
    if (volume(policyDims) != volume(expectedPolicy))
    {
        throw std::runtime_error("Leela-model policy tensor has unexpected size.");
    }

    const auto expectedValue = nvinfer1::Dims2{kLeelaBatchSize, kLeelaValueSize};
    if (volume(winnerDims) != volume(expectedValue) || volume(qDims) != volume(expectedValue))
    {
        throw std::runtime_error("Leela-model value tensors have unexpected size.");
    }

    const auto inputType = engine.getTensorDataType(ctx.inputName.c_str());
    const auto policyType = engine.getTensorDataType(ctx.policyOutputName.c_str());
    const auto winnerType = engine.getTensorDataType(ctx.valueWinnerOutputName.c_str());
    const auto qType = engine.getTensorDataType(ctx.valueQOutputName.c_str());

    std::unique_ptr<nvinfer1::IExecutionContext> executionContext(engine.createExecutionContext());
    if (!executionContext)
    {
        throw std::runtime_error("Failed to create Leela-model execution context.");
    }

    if (!executionContext->setInputShape(ctx.inputName.c_str(), expectedInput))
    {
        throw std::runtime_error("Failed to set Leela-model input shape.");
    }

    DeviceBuffer inputDevice(volume(inputDims) * getElementSize(inputType));
    DeviceBuffer policyDevice(volume(policyDims) * getElementSize(policyType));
    DeviceBuffer winnerDevice(volume(winnerDims) * getElementSize(winnerType));
    DeviceBuffer qDevice(volume(qDims) * getElementSize(qType));
    CudaStream stream;

    std::vector<__half> inputHalf;
    std::vector<__half> policyHalf;
    std::vector<__half> winnerHalf;
    std::vector<__half> qHalf;

    const void* inputHostData = nullptr;
    size_t inputBytes = 0;

    if (inputType == nvinfer1::DataType::kFLOAT)
    {
        inputHostData = input.data();
        inputBytes = input.size() * sizeof(float);
    }
    else if (inputType == nvinfer1::DataType::kHALF)
    {
        inputHalf.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i)
        {
            inputHalf[i] = __float2half(input[i]);
        }
        inputHostData = inputHalf.data();
        inputBytes = inputHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported Leela-model input tensor type.");
    }

    if (cudaMemcpyAsync(inputDevice.data(), inputHostData, inputBytes, cudaMemcpyHostToDevice, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (Leela H2D) failed.");
    }

    if (!executionContext->setTensorAddress(ctx.inputName.c_str(), inputDevice.data())
        || !executionContext->setTensorAddress(ctx.policyOutputName.c_str(), policyDevice.data())
        || !executionContext->setTensorAddress(ctx.valueWinnerOutputName.c_str(), winnerDevice.data())
        || !executionContext->setTensorAddress(ctx.valueQOutputName.c_str(), qDevice.data()))
    {
        throw std::runtime_error("Failed to set Leela-model tensor addresses.");
    }

    if (!executionContext->enqueueV3(stream.get()))
    {
        throw std::runtime_error("Leela-model TensorRT enqueue failed.");
    }

    RawOutputs outputs;
    outputs.policy.resize(volume(policyDims));
    outputs.valueWinner.resize(volume(winnerDims));
    outputs.valueQ.resize(volume(qDims));

    void* policyHostData = nullptr;
    size_t policyBytes = 0;

    if (policyType == nvinfer1::DataType::kFLOAT)
    {
        policyHostData = outputs.policy.data();
        policyBytes = outputs.policy.size() * sizeof(float);
    }
    else if (policyType == nvinfer1::DataType::kHALF)
    {
        policyHalf.resize(outputs.policy.size());
        policyHostData = policyHalf.data();
        policyBytes = policyHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported Leela-model policy tensor type.");
    }

    void* winnerHostData = nullptr;
    size_t winnerBytes = 0;

    if (winnerType == nvinfer1::DataType::kFLOAT)
    {
        winnerHostData = outputs.valueWinner.data();
        winnerBytes = outputs.valueWinner.size() * sizeof(float);
    }
    else if (winnerType == nvinfer1::DataType::kHALF)
    {
        winnerHalf.resize(outputs.valueWinner.size());
        winnerHostData = winnerHalf.data();
        winnerBytes = winnerHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported Leela-model value_winner tensor type.");
    }

    void* qHostData = nullptr;
    size_t qBytes = 0;

    if (qType == nvinfer1::DataType::kFLOAT)
    {
        qHostData = outputs.valueQ.data();
        qBytes = outputs.valueQ.size() * sizeof(float);
    }
    else if (qType == nvinfer1::DataType::kHALF)
    {
        qHalf.resize(outputs.valueQ.size());
        qHostData = qHalf.data();
        qBytes = qHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported Leela-model value_q tensor type.");
    }

    if (cudaMemcpyAsync(policyHostData, policyDevice.data(), policyBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (Leela policy D2H) failed.");
    }

    if (cudaMemcpyAsync(winnerHostData, winnerDevice.data(), winnerBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (Leela value_winner D2H) failed.");
    }

    if (cudaMemcpyAsync(qHostData, qDevice.data(), qBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (Leela value_q D2H) failed.");
    }

    if (cudaStreamSynchronize(stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaStreamSynchronize (Leela) failed.");
    }

    if (policyType == nvinfer1::DataType::kHALF)
    {
        for (size_t i = 0; i < outputs.policy.size(); ++i)
        {
            outputs.policy[i] = __half2float(policyHalf[i]);
        }
    }
    if (winnerType == nvinfer1::DataType::kHALF)
    {
        for (size_t i = 0; i < outputs.valueWinner.size(); ++i)
        {
            outputs.valueWinner[i] = __half2float(winnerHalf[i]);
        }
    }
    if (qType == nvinfer1::DataType::kHALF)
    {
        for (size_t i = 0; i < outputs.valueQ.size(); ++i)
        {
            outputs.valueQ[i] = __half2float(qHalf[i]);
        }
    }

    return outputs;
}

std::vector<std::string> selectMoves(
    const std::vector<float>& logits,
    const std::vector<chess::Board>& maskBoards,
    const std::vector<bool>& isBlackToMove,
    const std::vector<std::string>& castlingRights,
    float temperature)
{
    if (maskBoards.size() * kLeelaPolicySize != logits.size())
    {
        throw std::runtime_error("Mismatch between Leela-model logits and batch size.");
    }

    auto& vocab = getPolicyVocab();
    std::vector<std::string> moves;
    moves.reserve(maskBoards.size());

    std::mt19937 rng(std::random_device{}());

    std::vector<float> row(kLeelaPolicySize);
    std::vector<char> legalMask(kLeelaPolicySize);
    std::vector<float> scaled(kLeelaPolicySize);
    std::vector<double> probs(kLeelaPolicySize);

    for (size_t idx = 0; idx < maskBoards.size(); ++idx)
    {
        const float* rowStart = logits.data() + static_cast<ptrdiff_t>(idx * kLeelaPolicySize);
        std::copy(rowStart, rowStart + kLeelaPolicySize, row.begin());
        std::fill(legalMask.begin(), legalMask.end(), 0);

        chess::Movelist legalMoves;
        chess::movegen::legalmoves(legalMoves, maskBoards[idx]);

        std::unordered_set<std::string> legalSet;
        legalSet.reserve(legalMoves.size());

        std::vector<int> legalIndices;
        legalIndices.reserve(legalMoves.size());

        for (const auto& move : legalMoves)
        {
            std::string uci = chess::uci::moveToUci(move, maskBoards[idx].chess960());
            if (!uci.empty() && uci.back() == 'n')
            {
                uci.pop_back();
            }
            legalSet.insert(uci);

            if (auto mapped = vocab.indexOf(uci))
            {
                if (!legalMask[*mapped])
                {
                    legalMask[*mapped] = 1;
                    legalIndices.push_back(*mapped);
                }
            }
        }

        auto markCastlingPseudo = [&](const char* pseudo, const char* actual) {
            if (legalSet.find(actual) != legalSet.end())
            {
                if (auto mapped = vocab.indexOf(pseudo))
                {
                    if (!legalMask[*mapped])
                    {
                        legalMask[*mapped] = 1;
                        legalIndices.push_back(*mapped);
                    }
                }
            }
        };

        markCastlingPseudo("e1h1", "e1g1");
        markCastlingPseudo("e1a1", "e1c1");
        markCastlingPseudo("e8h8", "e8g8");
        markCastlingPseudo("e8a8", "e8c8");

        if (!legalIndices.empty())
        {
            for (int i = 0; i < kLeelaPolicySize; ++i)
            {
                if (!legalMask[i])
                {
                    row[i] = kIllegalScore;
                }
            }
        }

        int selectedIndex = -1;

        if (temperature <= 0.0F || legalIndices.size() <= 1)
        {
            selectedIndex = static_cast<int>(std::distance(row.begin(), std::max_element(row.begin(), row.end())));
        }
        else
        {
            float maxVal = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < kLeelaPolicySize; ++i)
            {
                const float val = row[i] / temperature;
                scaled[i] = val;
                if (val > maxVal)
                {
                    maxVal = val;
                }
            }

            double sum = 0.0;
            for (int i = 0; i < kLeelaPolicySize; ++i)
            {
                const double expVal = std::exp(static_cast<double>(scaled[i] - maxVal));
                probs[i] = expVal;
                sum += expVal;
            }

            if (sum <= 0.0)
            {
                selectedIndex = static_cast<int>(std::distance(row.begin(), std::max_element(row.begin(), row.end())));
            }
            else
            {
                for (double& p : probs)
                {
                    p /= sum;
                }
                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                selectedIndex = dist(rng);
            }
        }

        if (selectedIndex < 0 || selectedIndex >= kLeelaPolicySize)
        {
            throw std::runtime_error("Failed to select Leela-model move.");
        }

        std::string move = vocab.at(selectedIndex);
        if (isBlackToMove[idx])
        {
            move = mirrorUciRanks(move);
        }
        move = adjustCastling(move, castlingRights[idx]);
        moves.push_back(std::move(move));
    }

    return moves;
}

LeelaInferenceResult runOnce(
    const std::vector<std::string>& originFens,
    const std::vector<std::vector<std::string>>& moveLists,
    float temperature,
    EngineContext& ctx)
{
    using clock = std::chrono::high_resolution_clock;
    const auto totalStart = clock::now();

    const auto encodeStart = clock::now();
    EncodedBatch encoded = encodeBatch(originFens, moveLists);
    const auto encodeEnd = clock::now();

    const auto inferStart = clock::now();
    RawOutputs rawOutputs = runInference(ctx, encoded.features);
    const auto inferEnd = clock::now();

    const auto sampleStart = clock::now();
    std::vector<std::string> moves = selectMoves(
        rawOutputs.policy, encoded.maskBoards, encoded.isBlackToMove, encoded.castlingRights, temperature);
    const auto sampleEnd = clock::now();

    const size_t batchSize = encoded.maskBoards.size();
    const size_t expectedSize = batchSize * static_cast<size_t>(kLeelaValueSize);
    if (rawOutputs.valueWinner.size() != expectedSize || rawOutputs.valueQ.size() != expectedSize)
    {
        throw std::runtime_error("Mismatch between Leela-model value outputs and batch size.");
    }

    std::vector<std::array<float, kLeelaValueSize>> winnerVectors;
    std::vector<std::array<float, kLeelaValueSize>> qVectors;
    winnerVectors.reserve(batchSize);
    qVectors.reserve(batchSize);

    for (size_t idx = 0; idx < batchSize; ++idx)
    {
        std::array<float, kLeelaValueSize> winner{};
        std::array<float, kLeelaValueSize> q{};
        for (int j = 0; j < kLeelaValueSize; ++j)
        {
            winner[static_cast<size_t>(j)]
                = rawOutputs.valueWinner[idx * static_cast<size_t>(kLeelaValueSize) + static_cast<size_t>(j)];
            q[static_cast<size_t>(j)]
                = rawOutputs.valueQ[idx * static_cast<size_t>(kLeelaValueSize) + static_cast<size_t>(j)];
        }
        winnerVectors.push_back(winner);
        qVectors.push_back(q);
    }

    const auto totalEnd = clock::now();

    auto durationMs = [](auto start, auto end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    InferenceTimings timings{
        .encodeMs = durationMs(encodeStart, encodeEnd),
        .inferMs = durationMs(inferStart, inferEnd),
        .sampleMs = durationMs(sampleStart, sampleEnd),
        .totalMs = durationMs(totalStart, totalEnd),
    };

    return LeelaInferenceResult{std::move(moves), std::move(winnerVectors), std::move(qVectors), timings};
}

} // namespace leela_impl

} // namespace

SmallInferenceResult small_generate_moves(
    const std::vector<std::string>& fens,
    float temperature,
    int iterations,
    const std::string& engine_path)
{
    if (iterations <= 0)
    {
        throw std::invalid_argument("iterations must be >= 1");
    }

    auto& ctx = small_impl::getEngineContext(engine_path);

    InferenceTimings cumulative{};
    SmallInferenceResult lastResult;

    for (int i = 0; i < iterations; ++i)
    {
        SmallInferenceResult result = small_impl::runOnce(fens, temperature, ctx);
        cumulative.encodeMs += result.timings.encodeMs;
        cumulative.inferMs += result.timings.inferMs;
        cumulative.sampleMs += result.timings.sampleMs;
        cumulative.totalMs += result.timings.totalMs;

        if (i == iterations - 1)
        {
            lastResult = std::move(result);
        }
    }

    lastResult.timings.encodeMs = cumulative.encodeMs / static_cast<double>(iterations);
    lastResult.timings.inferMs = cumulative.inferMs / static_cast<double>(iterations);
    lastResult.timings.sampleMs = cumulative.sampleMs / static_cast<double>(iterations);
    lastResult.timings.totalMs = cumulative.totalMs / static_cast<double>(iterations);

    return lastResult;
}

LeelaInferenceResult leela_generate_from_move_lists(
    const std::vector<std::string>& origin_fens,
    const std::vector<std::vector<std::string>>& move_lists,
    float temperature,
    int iterations,
    const std::string& engine_path)
{
    if (iterations <= 0)
    {
        throw std::invalid_argument("iterations must be >= 1");
    }

    auto& ctx = leela_impl::getEngineContext(engine_path);

    InferenceTimings cumulative{};
    LeelaInferenceResult lastResult;

    for (int i = 0; i < iterations; ++i)
    {
        LeelaInferenceResult result = leela_impl::runOnce(origin_fens, move_lists, temperature, ctx);
        cumulative.encodeMs += result.timings.encodeMs;
        cumulative.inferMs += result.timings.inferMs;
        cumulative.sampleMs += result.timings.sampleMs;
        cumulative.totalMs += result.timings.totalMs;

        if (i == iterations - 1)
        {
            lastResult = std::move(result);
        }
    }

    lastResult.timings.encodeMs = cumulative.encodeMs / static_cast<double>(iterations);
    lastResult.timings.inferMs = cumulative.inferMs / static_cast<double>(iterations);
    lastResult.timings.sampleMs = cumulative.sampleMs / static_cast<double>(iterations);
    lastResult.timings.totalMs = cumulative.totalMs / static_cast<double>(iterations);

    return lastResult;
}

size_t small_policy_size()
{
    return kSmallPolicySize;
}

size_t leela_policy_size()
{
    return kLeelaPolicySize;
}

const std::vector<std::string>& small_policy_vocabulary()
{
    return small_impl::getPolicyVocab().tokensRef();
}

const std::vector<std::string>& leela_policy_vocabulary()
{
    return leela_impl::getPolicyVocab().tokensRef();
}

} // namespace chessrl
