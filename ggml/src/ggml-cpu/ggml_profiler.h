#ifndef GGML_PROFILER_H
#define GGML_PROFILER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 初始化性能分析器
void ggml_profiler_init(void);

// 低开销算子级测量 (不支持嵌套)
void ggml_profiler_start(const char* name);
void ggml_profiler_end(const char* name);

// 采样模式控制 (基于低开销测量)
void ggml_profiler_set_sampling_rate(int rate);
int  ggml_profiler_get_sampling_rate(void);
void ggml_profiler_start_sampled(const char* name, int64_t call_id);
void ggml_profiler_end_sampled(const char* name, int64_t call_id);

// 生成性能报告
void ggml_profiler_report(void);
void ggml_profiler_report_sorted(void);
void ggml_profiler_report_csv(const char* filename);
void ggml_profiler_free(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_PROFILER_H
