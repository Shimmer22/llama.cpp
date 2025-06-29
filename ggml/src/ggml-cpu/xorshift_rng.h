#ifndef XORSHIFT_RNG_H
#define XORSHIFT_RNG_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ================= XorShift32 伪随机数生成器 =================

// 线程局部随机数状态（每个包含此头文件的源文件都会有独立的状态）
static __thread uint32_t rng_state = 0x12345678;

// XorShift32 算法宏版本 - 内联展开，零函数调用开销
#define XORSHIFT32() ({ \
    uint32_t x = rng_state; \
    x ^= x << 13; \
    x ^= x >> 17; \
    x ^= x << 5; \
    rng_state = x; \
    x; \
})

// 初始化随机数种子的宏
#define XORSHIFT32_SEED(seed) do { \
    rng_state = (seed) ? (seed) : 0x12345678; \
} while(0)

// 获取指定范围内随机数的宏 [0, max)
#define XORSHIFT32_RANGE(max) (XORSHIFT32() % (max))

// 随机布尔值
#define XORSHIFT32_BOOL() (XORSHIFT32() & 1)

// 随机浮点数 [0.0, 1.0)
#define XORSHIFT32_FLOAT() ((float)XORSHIFT32() / (float)UINT32_MAX)

// 随机双精度浮点数 [0.0, 1.0)
#define XORSHIFT32_DOUBLE() ((double)XORSHIFT32() / (double)UINT32_MAX)

// 支持多个独立随机数流的宏
#define DECLARE_RNG_STATE(name) static __thread uint32_t rng_state_##name = 0x12345678

#define XORSHIFT32_NAMED(name) ({ \
    uint32_t x = rng_state_##name; \
    x ^= x << 13; \
    x ^= x >> 17; \
    x ^= x << 5; \
    rng_state_##name = x; \
    x; \
})

#define XORSHIFT32_NAMED_RANGE(name, max) (XORSHIFT32_NAMED(name) % (max))

#define XORSHIFT32_NAMED_SEED(name, seed) do { \
    rng_state_##name = (seed) ? (seed) : 0x12345678; \
} while(0)

#ifdef __cplusplus
}
#endif

#endif // XORSHIFT_RNG_H
