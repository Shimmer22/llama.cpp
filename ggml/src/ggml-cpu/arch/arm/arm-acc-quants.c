#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#include "../../quants.h"
#include "../../ggml-cpu-impl.h"
#include "../../ggml_profiler.h"
#include "../../simd-mappings.h"
#include "xorshift_rng.h"

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

#endif

// Comparison function to evaluate three implementations
void ggml_vec_dot_q4_K_q8_K_compare(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, size_t bx,
                                    const void * GGML_RESTRICT vy, size_t by,
                                    int nrc) {
    float result = 0.0f;
    switch (XORSHIFT32_RANGE(3)) {
        case 0: ggml_vec_dot_q4_K_q8_K_arm_acc(n, &result, bs, vx, bx, vy, by, nrc); break;
        case 1: ggml_vec_dot_q4_K_q8_K(n, &result, bs, vx, bx, vy, by, nrc); break;
        case 2: ggml_vec_dot_q4_K_q8_K_generic(n, &result, bs, vx, bx, vy, by, nrc); break;
    }
    *s = result;
}


// Optimized function
void ggml_vec_dot_q4_K_q8_K_arm_acc(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#ifdef GGML_PERF_ENABLE
    static __thread int64_t call_id = 0;
    ggml_profiler_start_sampled("vec_dot_q4_K_q8_K_arm_acc", call_id);
#endif
    assert(n % QK_K == 0);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;
    uint32_t utmp[4];

#if defined __ARM_NEON
#ifdef GGML_PERF_ENABLE
    static __thread int64_t call_id_2 = 0;
    ggml_profiler_start_sampled("vec_dot_q4_K_q8_K_arm_acc: NEON", call_id_2);
#endif

    float sumf = 0.0f;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const int32x4_t mzero = vdupq_n_s32(0);

    // 多个累加器减少依赖链
    float32x4_t sumv = vdupq_n_f32(0.0f);
    
    const uint8_t prefetch_distance = 8;
    
    for (int i = 0; i < nb; ++i) {
        // 改进的预取策略 - 预取多个关键数据结构
        if (i + prefetch_distance < nb) {
            __builtin_prefetch(&x[i + prefetch_distance].qs, 0, 3);      // 量化数据
            __builtin_prefetch(&x[i + prefetch_distance].scales, 0, 3);  // 缩放因子
            __builtin_prefetch(&x[i + prefetch_distance].d, 0, 3);       // delta值
            __builtin_prefetch(&y[i + prefetch_distance].qs, 0, 3);      // Q8数据
            __builtin_prefetch(&y[i + prefetch_distance].bsums, 0, 3);   // bsums
        }

        // 预加载关键标量值
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

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
        
        // 预加载所有指针和scales到寄存器
        const uint8_t * GGML_RESTRICT q4_base = x[i].qs;
        const int8_t  * GGML_RESTRICT q8_base = y[i].qs;
        
        // 将scales打包成NEON寄存器以便后续使用
        const uint8x8_t scales_vec = vld1_u8(scales);
        const uint16x8_t scales_16 = vmovl_u8(scales_vec);
        
        // 多个独立的累加器，减少数据依赖
        int32x4_t acc1 = mzero, acc2 = mzero, acc3 = mzero, acc4 = mzero;
        
        // 完全展开的循环，每个迭代块优化指令混合
        
        // ===== 迭代 0 和 1 的指令交错 =====
        {
            // 预加载迭代0的数据
            const ggml_uint8x16x2_t q4bits_0 = ggml_vld1q_u8_x2(q4_base);
            const ggml_int8x16x2_t q8bytes_0a = ggml_vld1q_s8_x2(q8_base + 0);
            const ggml_int8x16x2_t q8bytes_0b = ggml_vld1q_s8_x2(q8_base + 32);
            
            // 同时预加载迭代1的数据
            const ggml_uint8x16x2_t q4bits_1 = ggml_vld1q_u8_x2(q4_base + 32);
            const ggml_int8x16x2_t q8bytes_1a = ggml_vld1q_s8_x2(q8_base + 64);
            const ggml_int8x16x2_t q8bytes_1b = ggml_vld1q_s8_x2(q8_base + 96);
            
            // 迭代0的低4位处理，同时准备迭代1的数据
            ggml_int8x16x2_t q4bytes_0, q4bytes_1;
            q4bytes_0.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits_0.val[0], m4b));
            q4bytes_1.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits_1.val[0], m4b));
            q4bytes_0.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits_0.val[1], m4b));
            q4bytes_1.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits_1.val[1], m4b));
            
            // 并行计算两个dot products
            const int32x4_t p1_0 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes_0.val[0], q8bytes_0a.val[0]), q4bytes_0.val[1], q8bytes_0a.val[1]);
            const int32x4_t p1_1 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes_1.val[0], q8bytes_1a.val[0]), q4bytes_1.val[1], q8bytes_1a.val[1]);
            
            // 提取scale值并进行向量化乘法
            const uint16_t scale0 = vgetq_lane_u16(scales_16, 0);
            const uint16_t scale2 = vgetq_lane_u16(scales_16, 2);
            acc1 = vmlaq_n_s32(acc1, p1_0, scale0);
            acc2 = vmlaq_n_s32(acc2, p1_1, scale2);
            
            // 高4位处理
            q4bytes_0.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits_0.val[0], 4));
            q4bytes_1.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits_1.val[0], 4));
            q4bytes_0.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits_0.val[1], 4));
            q4bytes_1.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits_1.val[1], 4));
            
            const int32x4_t p2_0 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes_0.val[0], q8bytes_0b.val[0]), q4bytes_0.val[1], q8bytes_0b.val[1]);
            const int32x4_t p2_1 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes_1.val[0], q8bytes_1b.val[0]), q4bytes_1.val[1], q8bytes_1b.val[1]);
            
            const uint16_t scale1 = vgetq_lane_u16(scales_16, 1);
            const uint16_t scale3 = vgetq_lane_u16(scales_16, 3);
            acc1 = vmlaq_n_s32(acc1, p2_0, scale1);
            acc2 = vmlaq_n_s32(acc2, p2_1, scale3);
        }
        
        // ===== 迭代 2 和 3 的指令交错 =====
        {
            // 预加载迭代2和3的数据
            const ggml_uint8x16x2_t q4bits_2 = ggml_vld1q_u8_x2(q4_base + 64);
            const ggml_int8x16x2_t q8bytes_2a = ggml_vld1q_s8_x2(q8_base + 128);
            const ggml_int8x16x2_t q8bytes_2b = ggml_vld1q_s8_x2(q8_base + 160);
            
            const ggml_uint8x16x2_t q4bits_3 = ggml_vld1q_u8_x2(q4_base + 96);
            const ggml_int8x16x2_t q8bytes_3a = ggml_vld1q_s8_x2(q8_base + 192);
            const ggml_int8x16x2_t q8bytes_3b = ggml_vld1q_s8_x2(q8_base + 224);
            
            // 并行处理低4位
            ggml_int8x16x2_t q4bytes_2, q4bytes_3;
            q4bytes_2.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits_2.val[0], m4b));
            q4bytes_3.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits_3.val[0], m4b));
            q4bytes_2.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits_2.val[1], m4b));
            q4bytes_3.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits_3.val[1], m4b));
            
            const int32x4_t p1_2 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes_2.val[0], q8bytes_2a.val[0]), q4bytes_2.val[1], q8bytes_2a.val[1]);
            const int32x4_t p1_3 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes_3.val[0], q8bytes_3a.val[0]), q4bytes_3.val[1], q8bytes_3a.val[1]);
            
            const uint16_t scale4 = vgetq_lane_u16(scales_16, 4);
            const uint16_t scale6 = vgetq_lane_u16(scales_16, 6);
            acc3 = vmlaq_n_s32(acc3, p1_2, scale4);
            acc4 = vmlaq_n_s32(acc4, p1_3, scale6);
            
            // 并行处理高4位
            q4bytes_2.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits_2.val[0], 4));
            q4bytes_3.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits_3.val[0], 4));
            q4bytes_2.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits_2.val[1], 4));
            q4bytes_3.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits_3.val[1], 4));
            
            const int32x4_t p2_2 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes_2.val[0], q8bytes_2b.val[0]), q4bytes_2.val[1], q8bytes_2b.val[1]);
            const int32x4_t p2_3 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes_3.val[0], q8bytes_3b.val[0]), q4bytes_3.val[1], q8bytes_3b.val[1]);
            
            const uint16_t scale5 = vgetq_lane_u16(scales_16, 5);
            const uint16_t scale7 = vgetq_lane_u16(scales_16, 7);
            acc3 = vmlaq_n_s32(acc3, p2_2, scale5);
            acc4 = vmlaq_n_s32(acc4, p2_3, scale7);
        }
        
        // 合并所有累加器
        const int32x4_t final_acc = vaddq_s32(vaddq_s32(acc1, acc2), vaddq_s32(acc3, acc4));
        const int32_t total_sum = vaddvq_s32(final_acc);
        
        // 使用FMA进行最终累加
        sumf = vfmaq_n_f32(vdupq_n_f32(sumf), vdupq_n_f32(d), total_sum)[0];
    }

    *s = sumf;

#ifdef GGML_PERF_ENABLE
    ggml_profiler_end_sampled("vec_dot_q4_K_q8_K_arm_acc: NEON", call_id_2++);
#endif
#endif
#ifdef GGML_PERF_ENABLE
    ggml_profiler_end_sampled("vec_dot_q4_K_q8_K_arm_acc", call_id++);
#endif
}
