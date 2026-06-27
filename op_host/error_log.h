#ifndef OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
#define OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_

#include <string>
#include "toolchain/slog.h"

namespace optiling {

#define OP_CHECK_IF(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                          \
    do {                                                                  \
        if ((ptr) == nullptr) {                                           \
            OPS_LOG_E(context->GetNodeName(), "%s is null", #ptr);        \
            return ge::GRAPH_FAILED;                                      \
        }                                                                 \
    } while (0)

#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)

}  // namespace optiling

template <typename T>
T CeilAlign(T a, T b)
{
    return (a + b - 1) / b * b;
}

template <typename T>
T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

#endif  // OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
