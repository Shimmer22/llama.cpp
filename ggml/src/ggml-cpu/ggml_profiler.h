#ifndef GGML_PROFILER_H
#define GGML_PROFILER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_profiler_init(void);
void ggml_profiler_start(const char * name);
void ggml_profiler_end(const char * name);
void ggml_profiler_report(void);
void ggml_profiler_reset(void);
void ggml_profiler_save_csv(const char * filename);

#ifdef __cplusplus
}
#endif

#endif // GGML_PROFILER_H
