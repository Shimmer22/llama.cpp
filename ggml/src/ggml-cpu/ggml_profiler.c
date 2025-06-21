#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

#define HASH_SIZE 128
#define MAX_THREADS 64

typedef struct {
    const char * name;
    int64_t call_count;
    double total_ns;
} profile_entry_t;

typedef struct {
    profile_entry_t items[HASH_SIZE];
    int used;
} thread_profiler_t;

static __thread thread_profiler_t thread_profiler;
static __thread uint64_t t0;

static thread_profiler_t * all_thread_profilers[MAX_THREADS];
static pthread_mutex_t all_thread_mutex = PTHREAD_MUTEX_INITIALIZER;

static uint32_t hash_string(const char * str) {
    uint32_t hash = 5381;
    int c;
    while ((c = *str++)) hash = ((hash << 5) + hash) + c;
    return hash % HASH_SIZE;
}

static inline uint64_t time_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

// 注册当前线程的 profiler（首次使用时调用）
static void register_thread_profiler() {
    pthread_mutex_lock(&all_thread_mutex);
    for (int i = 0; i < MAX_THREADS; ++i) {
        if (all_thread_profilers[i] == NULL) {
            all_thread_profilers[i] = &thread_profiler;
            break;
        }
    }
    pthread_mutex_unlock(&all_thread_mutex);
}

void ggml_profiler_init() {
    memset(all_thread_profilers, 0, sizeof(all_thread_profilers));
}

void ggml_profiler_start(const char * name) {
    if (thread_profiler.used == 0) {
        register_thread_profiler();
    }
    t0 = time_now_ns();
}

void ggml_profiler_end(const char * name) {
    uint64_t t1 = time_now_ns();
    double dt = (double)(t1 - t0);
    uint32_t hash = hash_string(name);

    for (int i = 0; i < HASH_SIZE; ++i) {
        uint32_t idx = (hash + i) % HASH_SIZE;
        if (thread_profiler.items[idx].name == NULL) {
            thread_profiler.items[idx].name = name;
            thread_profiler.items[idx].call_count = 1;
            thread_profiler.items[idx].total_ns = dt;
            thread_profiler.used++;
            break;
        } else if (strcmp(thread_profiler.items[idx].name, name) == 0) {
            thread_profiler.items[idx].call_count++;
            thread_profiler.items[idx].total_ns += dt;
            break;
        }
    }
}

void ggml_profiler_report() {
    printf("\n===== GGML Profiler Report =====\n");
    printf("%-30s %-15s %-20s %-20s\n", "Function", "Calls", "Total (ms)", "Avg (us)");

    profile_entry_t combined[HASH_SIZE] = {0};

    pthread_mutex_lock(&all_thread_mutex);
    for (int t = 0; t < MAX_THREADS; ++t) {
        thread_profiler_t * p = all_thread_profilers[t];
        if (!p) continue;
        for (int i = 0; i < HASH_SIZE; ++i) {
            if (p->items[i].name == NULL) continue;

            uint32_t hash = hash_string(p->items[i].name);
            for (int j = 0; j < HASH_SIZE; ++j) {
                uint32_t idx = (hash + j) % HASH_SIZE;
                if (combined[idx].name == NULL) {
                    combined[idx].name = p->items[i].name;
                    combined[idx].call_count = p->items[i].call_count;
                    combined[idx].total_ns = p->items[i].total_ns;
                    break;
                } else if (strcmp(combined[idx].name, p->items[i].name) == 0) {
                    combined[idx].call_count += p->items[i].call_count;
                    combined[idx].total_ns += p->items[i].total_ns;
                    break;
                }
            }
        }
    }
    pthread_mutex_unlock(&all_thread_mutex);

    for (int i = 0; i < HASH_SIZE; ++i) {
        if (combined[i].name != NULL) {
            double total_ms = combined[i].total_ns / 1e6;
            double avg_us = combined[i].total_ns / combined[i].call_count / 1e3;
            printf("%-30s %-15ld %-20.3f %-20.3f\n",
                   combined[i].name,
                   combined[i].call_count,
                   total_ms,
                   avg_us);
        }
    }

    printf("=================================\n");
}
