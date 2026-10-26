// Weight-free WASM SIMD candidate controls. This executable is not a model probe
// and is excluded from ordinary builds. Run on hosted Emscripten only.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static void converter_controls() {
    std::array<float, 32> input{};
    input[0] = 127.0f;
    input[1] = 1.6f; input[2] = -1.6f;
    input[3] = 1.49f; input[4] = -1.49f;
    input[5] = 2.5f; input[6] = -2.5f;
    block_q8_0 legacy{}, nearest{}, reference{};
    quantize_row_q8_0(input.data(), &legacy, 32);
    quantize_row_q8_0_wasm_nearest(input.data(), &nearest, 32);
    quantize_row_q8_0_ref(input.data(), &reference, 32);
    require(legacy.d == nearest.d && nearest.d == reference.d, "scale preserved");
    require(legacy.qs[0] == 127 && nearest.qs[0] == 127, "unit block scale");
    require(legacy.qs[1] == 1 && legacy.qs[2] == -1, "legacy fractional truncation");
    require(nearest.qs[1] == 2 && nearest.qs[2] == -2, "nearest fractional rounding");
    require(nearest.qs[3] == 1 && nearest.qs[4] == -1, "below-half rounding");
    require(nearest.qs[5] == 2 && nearest.qs[6] == -2, "nearest-even ties");
    require(reference.qs[5] == 3 && reference.qs[6] == -3, "roundf oracle ties-away");
    input.fill(0);
    quantize_row_q8_0(input.data(), &legacy, 32);
    quantize_row_q8_0_wasm_nearest(input.data(), &nearest, 32);
    block_q8_0 zero{};
    require(std::memcmp(&legacy, &zero, sizeof zero) == 0, "legacy zero block");
    require(std::memcmp(&nearest, &zero, sizeof zero) == 0, "nearest zero block");
    std::printf("{\"control\":\"converter\",\"passed\":true}\n");
}

static void backend_controls(enum ggml_type type, bool indirect) {
    constexpr int width = 384;
    std::array<float, width> input{}, weight{};
    input[0] = 127.0f; input[1] = 1.6f; input[2] = -1.6f;
    input[3] = 2.5f; input[4] = -2.5f;
    weight[1] = 1.0f;
    const size_t row_bytes = ggml_row_size(type, width);
    const int experts = indirect ? 2 : 1;
    std::vector<uint8_t> weights(experts * row_bytes);
    for (int expert = 0; expert < experts; ++expert) {
        if (expert) weight[1] = -1.0f;
        if (type == GGML_TYPE_Q8_0) {
            quantize_row_q8_0_ref(weight.data(), (block_q8_0 *)(weights.data() + expert * row_bytes), width);
        } else {
            quantize_row_iq4_nl_ref(weight.data(), (block_iq4_nl *)(weights.data() + expert * row_bytes), width);
        }
    }
    std::array<block_q8_0, width / 32> qlegacy{}, qnearest{};
    quantize_row_q8_0(input.data(), qlegacy.data(), width);
    quantize_row_q8_0_wasm_nearest(input.data(), qnearest.data(), width);
    const auto * trait = ggml_get_type_traits_cpu(type);
    require(trait->vec_dot_type == GGML_TYPE_Q8_0, "mapped activation type");
    float expected_legacy = 0, expected_nearest = 0;
    const void * selected_weight = weights.data() + (experts - 1) * row_bytes;
    trait->vec_dot(width, &expected_legacy, 0, selected_weight, 0, qlegacy.data(), 0, 1);
    trait->vec_dot(width, &expected_nearest, 0, selected_weight, 0, qnearest.data(), 0, 1);
    require(std::fabs(expected_nearest - expected_legacy) > .01f, "control exposes rounding difference");

    ggml_context * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "metadata context");
    ggml_tensor * w = ggml_new_tensor_3d(ctx, type, width, 1, experts);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, 1);
    ggml_tensor * ids = indirect ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1) : nullptr;
    ggml_tensor * y = indirect ? ggml_mul_mat_id(ctx, w, x, ids) : ggml_mul_mat(ctx, w, x);
    ggml_set_output(y);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_t legacy = ggml_backend_cpu_init(), rounded = ggml_backend_cpu_init();
    require(legacy && rounded, "two independent backends");
    ggml_backend_cpu_set_n_threads(legacy, 1);
    ggml_backend_cpu_set_n_threads(rounded, 1);
    ggml_backend_cpu_set_wasm_q8_0_rounding(rounded, true);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, legacy);
    require(buffer != nullptr, "CPU graph buffer");
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(x, input.data(), 0, sizeof(input));
    if (ids) { const int32_t selected = 1; ggml_backend_tensor_set(ids, &selected, 0, sizeof selected); }
    auto check = [&](ggml_backend_t backend, float expected, const char * message) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "backend compute");
        float actual = 0; ggml_backend_tensor_get(y, &actual, 0, sizeof actual);
        require(std::memcmp(&actual, &expected, sizeof actual) == 0, message);
    };
    check(legacy, expected_legacy, "default-off backend");
    check(rounded, expected_nearest, "enabled backend");
    check(legacy, expected_legacy, "enabled backend did not alter legacy backend");
    ggml_backend_graph_plan_t plan = ggml_backend_graph_plan_create(rounded, graph);
    require(plan != nullptr, "cached plan");
    ggml_backend_cpu_set_wasm_q8_0_rounding(rounded, false);
    check(rounded, expected_legacy, "explicitly disabled backend");
    require(ggml_backend_graph_plan_compute(rounded, plan) == GGML_STATUS_SUCCESS, "cached plan compute");
    float actual = 0; ggml_backend_tensor_get(y, &actual, 0, sizeof actual);
    require(std::memcmp(&actual, &expected_nearest, sizeof actual) == 0, "cached plan retains its setting");
    check(legacy, expected_legacy, "cached plan did not alter other backend");
    ggml_backend_cpu_set_wasm_q8_0_rounding(rounded, true);
    check(rounded, expected_nearest, "re-enabled backend");
    ggml_cplan direct_plan = ggml_graph_plan(graph, 1, nullptr);
    require(!direct_plan.wasm_q8_0_round, "direct plan defaults off");
    ggml_backend_graph_plan_free(rounded, plan);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(rounded); ggml_backend_free(legacy);
    std::printf("{\"control\":\"backend\",\"type\":%d,\"indirect\":%s,\"legacy\":%.9g,\"nearest\":%.9g,\"passed\":true}\n",
        int(type), indirect ? "true" : "false", expected_legacy, expected_nearest);
}

int main() {
    require(ggml_cpu_has_wasm_simd() == 1, "actual WASM SIMD backend");
    converter_controls();
    for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL}) {
        backend_controls(type, false); backend_controls(type, true);
    }
    return 0;
}
