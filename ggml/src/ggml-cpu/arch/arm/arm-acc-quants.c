#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#include "../../quants.h"
#include "../../ggml-cpu-impl.h"
#include "../../ggml_profiler.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for printf
#include <math.h>   // for fabsf

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

#if defined(__ARM_NEON)
#define B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#define B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#define B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#define B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#define B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#define B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#define B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#define B8(c,s  ) B7(c,s,      c), B7(c,s,      s)

// precomputed tables for expanding 8bits to 8 bytes:
static const uint64_t table_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
static const uint64_t table_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#endif


// Generic C implementation with timing
static void ggml_vec_dot_q4_K_q8_K_c(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#ifdef GGML_PERF_ENABLE
    static __thread int64_t call_id_c = 0;
    ggml_profiler_start_sampled("ggml_vec_dot_q4_K_q8_K_c", call_id_c);
#endif
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);
    assert(n % QK_K == 0);

    const int nb = n / QK_K;

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

#ifdef GGML_PERF_ENABLE
    ggml_profiler_end_sampled("ggml_vec_dot_q4_K_q8_K_c", call_id_c++);
#endif
}

// Comparison function to evaluate three implementations
void ggml_vec_dot_q4_K_q8_K_compare(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, size_t bx,
                                    const void * GGML_RESTRICT vy, size_t by,
                                    int nrc) {
    // static __thread 用于每个线程维护自己的轮转状态
    static __thread int method_index = 0;

    float result = 0.0f;

    switch (method_index) {
        case 0:
            ggml_vec_dot_q4_K_q8_K_c(n, &result, bs, vx, bx, vy, by, nrc);
            break;
        case 1:
            ggml_vec_dot_q4_K_q8_K_arm_acc(n, &result, bs, vx, bx, vy, by, nrc);
            break;
        case 2:
            ggml_vec_dot_q4_K_q8_K(n, &result, bs, vx, bx, vy, by, nrc);
            break;
        default:
            assert(0 && "Invalid method_index");
    }

    // 下次轮换
    method_index = (method_index + 1) % 3;

    *s = result;
}


// Optimized function
void ggml_vec_dot_q4_K_q8_K_arm_acc(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#ifdef GGML_PERF_ENABLE
    static __thread int64_t call_id = 0;
    ggml_profiler_start_sampled("ggml_vec_dot_q4_K_q8_K_arm_acc", call_id);
#endif
    assert(n % QK_K == 0);
#ifdef __ARM_FEATURE_MATMUL_INT8
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#if defined __ARM_NEON
#ifdef GGML_PERF_ENABLE
    static __thread int64_t call_id_2 = 0;
    ggml_profiler_start_sampled("ggml_vec_dot_q4_K_q8_K_arm_acc: NEON", call_id_2);
#endif
    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const int32x4_t mzero = vdupq_n_s32(0);

    ggml_int8x16x2_t q4bytes;
    ggml_int8x16x2_t q8bytes;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        // 内存预取
        if (i + 1 < nb) {
            // 预取下一个块的结构体本身
            __builtin_prefetch(&x[i+1], 0, 3); // 读操作，最高优先级（加载到所有缓存级别）
            __builtin_prefetch(&y[i+1], 0, 3); // 读操作，最高优先级

            // 预取下一个块的关键数据数组
            __builtin_prefetch(x[i+1].qs, 0, 1);     // 预取量化后的 q4 数据，局部性较低（只读取一次）
            __builtin_prefetch(y[i+1].qs, 0, 1);     // 预取量化后的 q8 数据，局部性较低
            __builtin_prefetch(y[i+1].bsums, 0, 1);  // 预取 q8 的 bsums，局部性较低
            __builtin_prefetch(x[i+1].scales, 0, 1); // 预取 q4 的 scales，局部性较低
        }
        

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, 12);

        uint32x2_t mins8 = { 0 };
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);

        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
                                             vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * vaddvq_s32(prod);

        const uint8_t * scales = (const uint8_t *)utmp;

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        for (int j = 0; j < QK_K/64; ++j) {
            const ggml_uint8x16x2_t q4bits = ggml_vld1q_u8_x2(q4); q4 += 32;

            q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
            q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));

            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
            sumi1 += vaddvq_s32(p1) * scales[2*j+0];

            q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
            q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);

            sumi2 += vaddvq_s32(p2) * scales[2*j+1];
        }

        sumf += d * (sumi1 + sumi2);

    }

    *s = sumf;
#ifdef GGML_PERF_ENABLE
    ggml_profiler_end_sampled("ggml_vec_dot_q4_K_q8_K_arm_acc: NEON", call_id_2++);
#endif
// Note: The generic C implementation was here. It has been moved to its own function.
#endif
#ifdef GGML_PERF_ENABLE
    ggml_profiler_end_sampled("ggml_vec_dot_q4_K_q8_K_arm_acc", call_id++);
#endif
}
