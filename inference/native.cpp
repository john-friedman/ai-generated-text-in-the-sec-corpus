#include <dlfcn.h>
using HMODULE = void*;
inline HMODULE LoadLibraryA(char const* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
inline void* GetProcAddress(HMODULE module, char const* name) { return dlsym(module, name); }
#ifndef BENCH_BATCH
#define BENCH_BATCH 40
#endif
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>

static thread_local std::string error;
struct Logger : nvinfer1::ILogger {
    void log(Severity s, char const* message) noexcept override {
        if (s <= Severity::kERROR) error = message;
    }
};
template<class T> T symbol(HMODULE dll, char const* name) {
    auto p = GetProcAddress(dll, name);
    if (!p) throw std::runtime_error(std::string("Missing native symbol: ") + name);
    return reinterpret_cast<T>(p);
}
struct Cuda {
    HMODULE dll;
    decltype(&cudaSetDevice) set_device;
    decltype(&cudaMalloc) malloc_device;
    decltype(&cudaFree) free_device;
    decltype(&cudaHostAlloc) malloc_host;
    decltype(&cudaFreeHost) free_host;
    decltype(&cudaMemcpyAsync) copy;
    decltype(&cudaStreamCreateWithFlags) stream_new;
    decltype(&cudaStreamDestroy) stream_free;
    decltype(&cudaStreamSynchronize) sync;
    decltype(&cudaStreamWaitEvent) wait;
    decltype(&cudaEventCreateWithFlags) event_new;
    decltype(&cudaEventRecord) record;
    decltype(&cudaEventDestroy) event_free;
    decltype(&cudaGetErrorString) describe;
    decltype(&cudaStreamBeginCapture) begin_capture;
    decltype(&cudaStreamEndCapture) end_capture;
    decltype(&cudaGraphInstantiateWithFlags) instantiate;
    decltype(&cudaGraphLaunch) launch_graph;
    decltype(&cudaGraphDestroy) destroy_graph;
    decltype(&cudaGraphExecDestroy) destroy_graph_exec;
    Cuda(char const* path) {
        dll = LoadLibraryA(path);
        if (!dll) throw std::runtime_error("Cannot load CUDA runtime");
#define LOAD(field, name) field = symbol<decltype(field)>(dll, #name)
        LOAD(set_device, cudaSetDevice); LOAD(malloc_device, cudaMalloc); LOAD(free_device, cudaFree);
        LOAD(malloc_host, cudaHostAlloc); LOAD(free_host, cudaFreeHost); LOAD(copy, cudaMemcpyAsync);
        LOAD(stream_new, cudaStreamCreateWithFlags); LOAD(stream_free, cudaStreamDestroy);
        LOAD(sync, cudaStreamSynchronize); LOAD(wait, cudaStreamWaitEvent);
        LOAD(event_new, cudaEventCreateWithFlags); LOAD(record, cudaEventRecord);
        LOAD(event_free, cudaEventDestroy); LOAD(describe, cudaGetErrorString);
        LOAD(begin_capture, cudaStreamBeginCapture); LOAD(end_capture, cudaStreamEndCapture);
        LOAD(instantiate, cudaGraphInstantiateWithFlags); LOAD(launch_graph, cudaGraphLaunch);
        LOAD(destroy_graph, cudaGraphDestroy); LOAD(destroy_graph_exec, cudaGraphExecDestroy);
#undef LOAD
    }
    void check(cudaError_t value) { if (value != cudaSuccess) throw std::runtime_error(describe(value)); }
};
struct Buffer {
    Cuda* cuda;
    int32_t *host = nullptr, *device = nullptr;
    float *output = nullptr, *device_output = nullptr;
    size_t capacity = 0, rows = 0;
    cudaEvent_t ready = nullptr;
    bool busy = false;
    Buffer(Cuda* c, size_t n, size_t r): cuda(c), capacity(n), rows(r) {
        c->check(c->malloc_host((void**)&host, n * 8, cudaHostAllocDefault));
        c->check(c->malloc_device((void**)&device, n * 8));
        c->check(c->malloc_host((void**)&output, r * 16, cudaHostAllocDefault));
        c->check(c->malloc_device((void**)&device_output, r * 16));
        c->check(c->event_new(&ready, cudaEventDisableTiming));
    }
    ~Buffer() {
        if (ready) cuda->event_free(ready);
        if (host) cuda->free_host(host);
        if (device) cuda->free_device(device);
        if (output) cuda->free_host(output);
        if (device_output) cuda->free_device(device_output);
    }
};
struct Captured {
    Cuda* cuda;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaGraphExec_t executable = nullptr;
    explicit Captured(Cuda* c): cuda(c) {}
    ~Captured() { if (executable) cuda->destroy_graph_exec(executable); }
};
struct Native {
    Cuda cuda;
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t transfer = nullptr, compute = nullptr;
    std::mutex mutex;
    std::vector<std::unique_ptr<Buffer>> buffers;
    std::unordered_map<size_t, std::unique_ptr<Captured>> graphs;
    void* workspace = nullptr;
    int32_t* graph_input = nullptr;
    float* graph_output = nullptr;
    bool graphs_enabled = true;
    bool graph_widths[513];
    static constexpr size_t graph_elements = BENCH_BATCH * 512;
    Native(char const* plan, char const* runtime_path, char const* infer_path): cuda(runtime_path) {
        std::fill(std::begin(graph_widths), std::end(graph_widths), true);
        cuda.check(cuda.set_device(0));
        auto dll = LoadLibraryA(infer_path);
        if (!dll) throw std::runtime_error("Cannot load TensorRT runtime DLL");
        auto create = symbol<void*(*)(void*, int32_t)>(dll, "createInferRuntime_INTERNAL");
        runtime.reset(static_cast<nvinfer1::IRuntime*>(create(&logger, NV_TENSORRT_VERSION)));
        if (!runtime) throw std::runtime_error("Cannot create TensorRT runtime: " + error);
        std::ifstream file(plan, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
        engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine) throw std::runtime_error("Cannot deserialize TensorRT plan: " + error);
        context.reset(engine->createExecutionContext());
        if (!context) throw std::runtime_error("Cannot create TensorRT execution context");
        cuda.check(cuda.stream_new(&transfer, cudaStreamNonBlocking));
        cuda.check(cuda.stream_new(&compute, cudaStreamNonBlocking));
        auto option = std::getenv("SEC_CUDA_GRAPHS");
        graphs_enabled = !option || std::string(option) != "0";
        if (graphs_enabled) {
            cuda.check(cuda.malloc_device(&workspace, engine->getDeviceMemorySizeV2()));
            cuda.check(cuda.malloc_device((void**)&graph_input, graph_elements * 8));
            cuda.check(cuda.malloc_device((void**)&graph_output, BENCH_BATCH * 16));
        }
    }
    // One fixed context per graph. All graphs execute sequentially on compute,
    // so their immutable I/O addresses and activation workspace can be shared.
    Captured* capture(size_t width) {
        auto found = graphs.find(width);
        if (found != graphs.end()) return found->second.get();
        auto graph = std::make_unique<Captured>(&cuda);
        graph->context.reset(engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
        if (!graph->context) throw std::runtime_error("Cannot create graph context");
        auto& ctx = graph->context;
        ctx->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kNONE);
        ctx->setDeviceMemoryV2(workspace, engine->getDeviceMemorySizeV2());
        nvinfer1::Dims dims{}; dims.nbDims=2; dims.d[0]=BENCH_BATCH; dims.d[1]=width;
        if (!ctx->setInputShape("input_ids", dims) || !ctx->setInputShape("attention_mask", dims))
            throw std::runtime_error("Unsupported graph shape");
        ctx->setTensorAddress("input_ids", graph_input);
        ctx->setTensorAddress("attention_mask", graph_input+graph_elements);
        ctx->setTensorAddress("logits", graph_output);
        if (!ctx->enqueueV3(compute)) throw std::runtime_error("Graph warmup failed: " + error);
        cuda.check(cuda.sync(compute));
        cuda.check(cuda.begin_capture(compute, cudaStreamCaptureModeThreadLocal));
        bool queued = ctx->enqueueV3(compute);
        cudaGraph_t raw = nullptr;
        auto ended = cuda.end_capture(compute, &raw);
        if (queued && ended == cudaSuccess && raw) {
            if (cuda.instantiate(&graph->executable, raw, 0) != cudaSuccess)
                graph->executable = nullptr;
        }
        if (raw) cuda.destroy_graph(raw);
        auto result = graph.get();
        graphs.emplace(width, std::move(graph));
        return result;
    }
    ~Native() {
        if (transfer) cuda.sync(transfer);
        if (compute) cuda.sync(compute);
        buffers.clear();
        graphs.clear();
        if (workspace) cuda.free_device(workspace);
        if (graph_input) cuda.free_device(graph_input);
        if (graph_output) cuda.free_device(graph_output);
        if (transfer) cuda.stream_free(transfer);
        if (compute) cuda.stream_free(compute);
    }
};
struct Plan { size_t start, count, width, offset; };
extern "C" {
char const* sec_error() { return error.c_str(); }
void* sec_create(char const* plan, char const* cuda, char const* infer) {
    try { return new Native(plan, cuda, infer); } catch (std::exception const& e) { error=e.what(); return nullptr; }
}
void sec_destroy(Native* n) { delete n; }
Buffer* sec_acquire(Native* n, size_t count, size_t rows) {
    try {
        n->cuda.check(n->cuda.set_device(0));
        std::lock_guard<std::mutex> lock(n->mutex);
        for (auto& b : n->buffers) if (!b->busy && b->capacity >= count && b->rows >= rows) {
            b->busy = true; return b.get();
        }
        // A bounded Rust job queue bounds the number of busy slots. Replace a
        // free undersized slot instead of retaining a slot for every input size.
        auto buffer = std::make_unique<Buffer>(&n->cuda, count, rows);
        buffer->busy = true;
        auto result = buffer.get();
        for (auto& old : n->buffers) if (!old->busy) { old = std::move(buffer); return result; }
        n->buffers.push_back(std::move(buffer));
        return result;
    } catch (std::exception const& e) { error=e.what(); return nullptr; }
}
int32_t* sec_host(Buffer* b, int mask) { return b->host + (mask ? b->capacity : 0); }
int sec_upload(Native* n, Buffer* b, size_t count) {
    try {
        n->cuda.check(n->cuda.copy(b->device, b->host, count*4, cudaMemcpyHostToDevice, n->transfer));
        n->cuda.check(n->cuda.copy(b->device+b->capacity, b->host+b->capacity, count*4, cudaMemcpyHostToDevice, n->transfer));
        n->cuda.check(n->cuda.record(b->ready, n->transfer)); return 1;
    } catch (std::exception const& e) { error=e.what(); return 0; }
}
int sec_score(Native* n, Buffer* b, Plan const* plans, size_t count, size_t rows, float* result) {
    try {
        n->cuda.check(n->cuda.set_device(0));
        n->cuda.check(n->cuda.wait(n->compute, b->ready, 0));
        for (size_t i=0; i<count; ++i) {
            auto const& p = plans[i];
            if (n->graphs_enabled && p.count == BENCH_BATCH && n->graph_widths[p.width]) {
                n->cuda.check(n->cuda.copy(n->graph_input, b->device+p.offset, p.count*p.width*4, cudaMemcpyDeviceToDevice, n->compute));
                n->cuda.check(n->cuda.copy(n->graph_input+Native::graph_elements, b->device+b->capacity+p.offset, p.count*p.width*4, cudaMemcpyDeviceToDevice, n->compute));
                auto graph = n->capture(p.width);
                if (graph->executable) {
                    n->cuda.check(n->cuda.launch_graph(graph->executable, n->compute));
                } else if (!graph->context->enqueueV3(n->compute)) {
                    throw std::runtime_error("Uncaptured shape enqueue failed: " + error);
                }
                n->cuda.check(n->cuda.copy(b->device_output+p.start*4, n->graph_output, p.count*16, cudaMemcpyDeviceToDevice, n->compute));
                continue;
            }
            nvinfer1::Dims dims{}; dims.nbDims=2; dims.d[0]=p.count; dims.d[1]=p.width;
            if (!n->context->setInputShape("input_ids", dims) || !n->context->setInputShape("attention_mask", dims))
                throw std::runtime_error("Unsupported TensorRT shape");
            n->context->setTensorAddress("input_ids", b->device+p.offset);
            n->context->setTensorAddress("attention_mask", b->device+b->capacity+p.offset);
            n->context->setTensorAddress("logits", b->device_output+p.start*4);
            if (!n->context->enqueueV3(n->compute)) throw std::runtime_error("TensorRT enqueue failed: " + error);
        }
        n->cuda.check(n->cuda.copy(b->output, b->device_output, rows*16, cudaMemcpyDeviceToHost, n->compute));
        n->cuda.check(n->cuda.sync(n->compute));
        std::copy(b->output, b->output+rows*4, result); return 1;
    } catch (std::exception const& e) { error=e.what(); return 0; }
}
void sec_release(Native* n, Buffer* b) {
    std::lock_guard<std::mutex> lock(n->mutex); b->busy=false;
}
}

extern "C" int sec_no_annotations(Native* n) {
    return n->context->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kNONE);
}
extern "C" size_t sec_graph_count(Native* n) {
    size_t count = 0;
    for (auto const& item : n->graphs) if (item.second->executable) ++count;
    return count;
}

extern "C" void sec_graph_widths(Native* n, uint32_t const* widths, size_t count) {
    std::fill(std::begin(n->graph_widths), std::end(n->graph_widths), false);
    for (size_t i=0; i<count; ++i) if (widths[i] <= 512) n->graph_widths[widths[i]] = true;
}
