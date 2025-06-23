#include "../ggml/src/ggml-cpu/ggml_profiler.h"
#include <stdio.h>
#include <unistd.h> // for usleep
#include <pthread.h>

#define NUM_THREADS 8
#define GEMM_CALLS 10000
#define OTHER_CALLS 10000
#define OVERHEAD_TEST_CALLS 1000 // 增加一个专门用于测试开销的调用次数

// 模拟一个耗时操作
void busy_wait_us(long us) {
    usleep(us);
}

// 模拟一个GEMM操作
void gemm_op() {
    busy_wait_us(2); // 模拟5us的GEMM计算
}

// 模拟一个向量加法操作
void vec_add_op() {
    busy_wait_us(4); // 模拟50us的向量加法
}

// 线程工作函数
void* worker_thread(void* arg) {
    long thread_id = (long)arg;
    // printf("Thread %ld started.\n", thread_id);

    // 1. 对高频函数 (gemm_op) 进行采样计时
    // 预期耗时 = 150us + 计时开销
    for (int i = 0; i < GEMM_CALLS; ++i) {
        ggml_profiler_start_sampled("gemm_op (2us)", i);
        gemm_op();
        ggml_profiler_end_sampled("gemm_op (2us)", i);
    }

    // 2. 对低频函数进行常规计时
    // 预期耗时 = 20us + 计时开销
    for (int i = 0; i < OTHER_CALLS; ++i) {
        ggml_profiler_start_sampled("vec_add_op (100us)", i);
        vec_add_op();
        ggml_profiler_end_sampled("vec_add_op (100us)", i);
    }

    // 3. 对空函数进行计时，以估算计时器本身的开销 (误差)
    // 预期耗时 = 计时开销
    for (int i = 0; i < OVERHEAD_TEST_CALLS; ++i) {
        ggml_profiler_start("profiler_overhead");
        // 此处为空，不执行任何操作
        ggml_profiler_end("profiler_overhead");
    }

    // printf("Thread %ld finished.\n", thread_id);
    return NULL;
}

int main() {
    // 初始化分析器
    ggml_profiler_init();

    // 设置采样率：每100次调用记录一次
    int rate = 100;
    ggml_profiler_set_sampling_rate(rate);
    printf("Profiler initialized. Sampling rate set to %d.\n", ggml_profiler_get_sampling_rate());
    
    pthread_t threads[NUM_THREADS];

    // 创建线程
    for (long i = 0; i < NUM_THREADS; ++i) {
        pthread_create(&threads[i], NULL, worker_thread, (void*)i);
    }

    // 等待所有线程完成
    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    printf("\nAll threads finished. Generating report...\n");

    // 打印按耗时排序的性能报告
    ggml_profiler_report_sorted();

    return 0;
}