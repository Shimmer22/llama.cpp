#include "../ggml/src/ggml-cpu/ggml_profiler.h"
#include <stdio.h>
#include <unistd.h> // for usleep
#include <pthread.h>

#define NUM_THREADS 8
#define GEMM_CALLS 10000
#define OTHER_CALLS 10000
#define OVERHEAD_TEST_CALLS 1000

// 模拟耗时操作
void busy_wait_us(long us) {
    usleep(us);
}

// fast 函数实现不同延时
void fast_1us() { busy_wait_us(1); }
void fast_5us() { busy_wait_us(5); }
void fast_10us() { busy_wait_us(10); }
void fast_20us() { busy_wait_us(20); }
void fast_50us() { busy_wait_us(50); }
void fast_100us() { busy_wait_us(100); }

static volatile inline void basic_test(void)
{
    // 测试分析器开销
    for (int i = 0; i < OVERHEAD_TEST_CALLS; ++i) {
        ggml_profiler_start_sampled("profiler_overhead",i);
        ggml_profiler_end_sampled("profiler_overhead",i);
    }
}

// 线程工作函数
void* worker_thread(void* arg) {
    long thread_id = (long)arg;

    // fast 操作
    for (int i = 0; i < GEMM_CALLS; ++i) {
        ggml_profiler_start_sampled("fast (1us)", i);
        fast_1us();
        ggml_profiler_end_sampled("fast (1us)", i);
    }
    for (int i = 0; i < GEMM_CALLS; ++i) {
        ggml_profiler_start_sampled("fast (5us)", i);
        fast_5us();
        ggml_profiler_end_sampled("fast (5us)", i);
    }
    for (int i = 0; i < GEMM_CALLS; ++i) {
        ggml_profiler_start_sampled("fast (10us)", i);
        fast_10us();
        ggml_profiler_end_sampled("fast (10us)", i);
    }
    for (int i = 0; i < GEMM_CALLS; ++i) {
        ggml_profiler_start_sampled("fast (20us)", i);
        fast_20us();
        ggml_profiler_end_sampled("fast (20us)", i);
    }
    for (int i = 0; i < GEMM_CALLS; ++i) {
        ggml_profiler_start_sampled("fast (50us)", i);
        fast_50us();
        ggml_profiler_end_sampled("fast (50us)", i);
    }
    for (int i = 0; i < GEMM_CALLS; ++i) {
        ggml_profiler_start_sampled("fast (100us)", i);
        fast_100us();
        ggml_profiler_end_sampled("fast (100us)", i);
    }

    basic_test();

    return NULL;
}

int main() {
    ggml_profiler_init();
    int rate = 100;
    ggml_profiler_set_sampling_rate(rate);
    printf("Profiler initialized. Sampling rate set to %d.\n", ggml_profiler_get_sampling_rate());
    
    pthread_t threads[NUM_THREADS];

    for (long i = 0; i < NUM_THREADS; ++i) {
        pthread_create(&threads[i], NULL, worker_thread, (void*)i);
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    printf("\nAll threads finished. Generating report...\n");
    ggml_profiler_report_sorted();

    return 0;
}

// clang -o test mrs_test.c ../ggml/src/ggml-cpu/ggml_profiler.c