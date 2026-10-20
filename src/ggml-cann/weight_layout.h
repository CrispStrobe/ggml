// CANN weight-layout policy. SDK-free so host controls exercise production policy.
#pragma once
#include "ggml.h"
#include <cstring>
#include <map>
#include <iterator>
#include <algorithm>

inline bool ggml_cann_weight_suffix(const char* name, const char* suffix) {
    const size_t n = std::strlen(name), s = std::strlen(suffix);
    return n >= s && std::strcmp(name + n - s, suffix) == 0 && (n == s || name[n - s - 1] == '.');
}

inline bool ggml_cann_legacy_matmul_weight(const ggml_tensor* t) {
    if (t->flags & GGML_TENSOR_FLAG_WEIGHT_GENERAL)
        return false;
    const char* names[] = {"output.weight",      "attn_q.weight",   "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};
    for (auto name : names)
        if (ggml_cann_weight_suffix(ggml_get_name(t), name))
            return true;
    return false;
}

inline bool ggml_cann_nz_candidate(const ggml_tensor* t) {
    // Cache accepts only ordinary immutable 2-D float matrices. BF16/quantized
    // recipes, batched weights, transposes and slices need their own acceptance.
    if ((t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_F32) || t->ne[0] <= 1 || t->ne[1] <= 1 || t->ne[2] != 1 ||
        t->ne[3] != 1 || t->view_src || !ggml_is_contiguous(t) || t->op != GGML_OP_NONE ||
        (t->flags & GGML_TENSOR_FLAG_WEIGHT_GENERAL))
        return false;
    if (t->flags & GGML_TENSOR_FLAG_WEIGHT_MATMUL_ONLY)
        return true;
    if (ggml_cann_legacy_matmul_weight(t))
        return true;
    const char* names[] = {"attn.q.weight",   "attn.k.weight",   "attn.v.weight", "attn.o.weight",   "attn.qkv.weight",
                           "attn_qkv.weight", "ffn.gate.weight", "ffn.up.weight", "ffn.down.weight", "lm_head.weight"};
    for (auto name : names)
        if (ggml_cann_weight_suffix(ggml_get_name(t), name))
            return true;
    return false;
}

// Union of uploaded byte ranges; counts alone cannot detect overlapping writes.
struct ggml_cann_upload_coverage {
    std::map<size_t, size_t> ranges;
    bool add(size_t offset, size_t size, size_t total) {
        if (offset > total || size > total - offset || size == 0)
            return false;
        size_t start = offset, end = offset + size;
        auto it = ranges.lower_bound(start);
        if (it != ranges.begin() && std::prev(it)->second >= start)
            --it;
        while (it != ranges.end() && it->first <= end) {
            start = std::min(start, it->first);
            end = std::max(end, it->second);
            it = ranges.erase(it);
        }
        ranges[start] = end;
        return ranges.begin()->first == 0 && ranges.begin()->second == total;
    }
    void clear() { ranges.clear(); }
};
