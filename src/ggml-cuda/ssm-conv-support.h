#pragma once

#include "ggml.h"

// Keep capability detection in sync with the dispatch in ssm-conv.cu. The
// scheduler must reject other widths before reaching the kernel's abort.
static inline bool ggml_cuda_ssm_conv_supported(const ggml_tensor* op) {
    const int64_t k = op->src[1]->ne[0];
    return op->src[0]->ne[1] % 128 == 0 && (k == 2 || k == 3 || k == 4 || k == 5 || k == 7 || k == 9 || k == 15);
}
