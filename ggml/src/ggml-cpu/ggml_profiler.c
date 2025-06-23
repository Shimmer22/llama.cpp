#include "ggml_profiler.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <stdlib.h> // For malloc/free

#define HASH_SIZE 251  // 使用质数减少哈希冲突
#define MAX_THREADS 64

// 性能分析条目 (已简化)
typedef struct {
    const char* name;
    int64_t call_count;
    double total_ns;
} profile_entry_t;

// 线程局部分析器 (已简化)
typedef struct {
    profile_entry_t items[HASH_SIZE];
    int used;
    // Add a flag to indicate if this profiler is active/registered
    int active;
} thread_profiler_t;

// 全局状态
static __thread thread_profiler_t* thread_profiler_ptr = NULL; // Now a pointer
static __thread uint64_t t0; // 用于计时的线程局部起始时间

static thread_profiler_t* all_thread_profilers[MAX_THREADS];
static pthread_mutex_t all_thread_mutex = PTHREAD_MUTEX_INITIALIZER;
static int sampling_rate = 500;
static int num_registered_threads = 0; // Keep track of registered threads

// String hashing function (unchanged)
static uint32_t hash_string(const char* str) {
    uint32_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % HASH_SIZE;
}

// Get current nanosecond time (unchanged)
static inline uint64_t time_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

// Register thread profiler
static void register_thread_profiler(void) {
    // Only register if not already registered for this thread
    if (thread_profiler_ptr == NULL) {
        // Dynamically allocate thread_profiler_t for this thread
        thread_profiler_ptr = (thread_profiler_t*)calloc(1, sizeof(thread_profiler_t));
        if (thread_profiler_ptr == NULL) {
            fprintf(stderr, "Error: Failed to allocate thread profiler for thread.\n");
            return;
        }

        pthread_mutex_lock(&all_thread_mutex);
        if (num_registered_threads < MAX_THREADS) {
            all_thread_profilers[num_registered_threads++] = thread_profiler_ptr;
            thread_profiler_ptr->active = 1; // Mark as active
        } else {
            fprintf(stderr, "Warning: Exceeded MAX_THREADS. Some thread profilers might not be registered.\n");
            free(thread_profiler_ptr); // Free if cannot register
            thread_profiler_ptr = NULL;
        }
        pthread_mutex_unlock(&all_thread_mutex);
    }
}

void ggml_profiler_init(void) {
    memset(all_thread_profilers, 0, sizeof(all_thread_profilers));
    num_registered_threads = 0;
    // Note: thread_profiler_ptr is __thread, so it's initialized to NULL for each new thread.
    // This function only initializes the global array.
}

// ================= Core functionality: Low overhead timing =================
void ggml_profiler_start(const char* name) {
    if (thread_profiler_ptr == NULL) {
        register_thread_profiler();
    }
    t0 = time_now_ns();
}

void ggml_profiler_end(const char* name) {
    if (thread_profiler_ptr == NULL || !thread_profiler_ptr->active) {
        // This should not happen if start was called, but a safeguard
        return;
    }

    uint64_t t1 = time_now_ns();
    double dt = (double)(t1 - t0);
    uint32_t hash = hash_string(name);

    // Use thread_profiler_ptr->items
    for (int i = 0; i < HASH_SIZE; ++i) {
        uint32_t idx = (hash + i) % HASH_SIZE;
        if (thread_profiler_ptr->items[idx].name == NULL) {
            thread_profiler_ptr->items[idx].name = name;
            thread_profiler_ptr->items[idx].call_count = 1;
            thread_profiler_ptr->items[idx].total_ns = dt;
            thread_profiler_ptr->used++;
            break;
        } else if (strcmp(thread_profiler_ptr->items[idx].name, name) == 0) {
            thread_profiler_ptr->items[idx].call_count++;
            thread_profiler_ptr->items[idx].total_ns += dt;
            break;
        }
    }
}

// ================= Sampling mode =================
void ggml_profiler_set_sampling_rate(int rate) {
    if (rate > 0) sampling_rate = rate;
}

int ggml_profiler_get_sampling_rate(void) {
    return sampling_rate;
}

void ggml_profiler_start_sampled(const char* name, int64_t call_id) {
    if (call_id % sampling_rate != 0) return;
    ggml_profiler_start(name);
}

void ggml_profiler_end_sampled(const char* name, int64_t call_id) {
    if (call_id % sampling_rate != 0) return;

    if (thread_profiler_ptr == NULL || !thread_profiler_ptr->active) {
        return;
    }

    uint64_t t1 = time_now_ns();
    double dt = (double)(t1 - t0);
    uint32_t hash = hash_string(name);

    for (int i = 0; i < HASH_SIZE; ++i) {
        uint32_t idx = (hash + i) % HASH_SIZE;
        if (thread_profiler_ptr->items[idx].name == NULL) {
            thread_profiler_ptr->items[idx].name = name;
            // Corrected: Scale by sampling_rate
            thread_profiler_ptr->items[idx].call_count = sampling_rate;
            thread_profiler_ptr->items[idx].total_ns = dt * sampling_rate;
            thread_profiler_ptr->used++;
            break;
        } else if (strcmp(thread_profiler_ptr->items[idx].name, name) == 0) {
            // Corrected: Scale by sampling_rate
            thread_profiler_ptr->items[idx].call_count += sampling_rate;
            thread_profiler_ptr->items[idx].total_ns += dt * sampling_rate;
            break;
        }
    }
}

// ================= Report functions =================

// Combine all thread performance data
static void combine_profiler_data(profile_entry_t combined[]) {
    pthread_mutex_lock(&all_thread_mutex);
    // Iterate only up to num_registered_threads to avoid NULL entries
    for (int t = 0; t < num_registered_threads; ++t) {
        thread_profiler_t* p = all_thread_profilers[t];
        if (!p || !p->active) continue; // Skip if not active

        for (int i = 0; i < HASH_SIZE; ++i) {
            if (p->items[i].name == NULL) continue;

            uint32_t hash = hash_string(p->items[i].name);
            for (int j = 0; j < HASH_SIZE; ++j) {
                uint32_t idx = (hash + j) % HASH_SIZE;
                if (combined[idx].name == NULL) {
                    combined[idx] = p->items[i];
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
}

// Generate performance report (unchanged except for combine_profiler_data loop)
void ggml_profiler_report(void) {
    profile_entry_t combined[HASH_SIZE] = {{0}};
    combine_profiler_data(combined);

    printf("\n===== GGML Profiler Report =====\n");
    printf("%-30s %-15s %-20s %-20s\n", "Function", "Calls", "Total (ms)", "Avg (us)");

    for (int i = 0; i < HASH_SIZE; ++i) {
        if (combined[i].name != NULL) {
            double total_ms = combined[i].total_ns / 1e6;
            double avg_us = (combined[i].call_count > 0) ? (combined[i].total_ns / combined[i].call_count / 1e3) : 0.0;
            printf("%-30s %-15ld %-20.3f %-20.3f\n",
                   combined[i].name,
                   combined[i].call_count,
                   total_ms,
                   avg_us);
        }
    }
    printf("=================================\n");
}

// Generate sorted performance report (unchanged except for combine_profiler_data loop and avg_us calculation)
void ggml_profiler_report_sorted() {
    profile_entry_t combined[HASH_SIZE] = {{0}};
    combine_profiler_data(combined);
    
    profile_entry_t entries[HASH_SIZE];
    int count = 0;
    for (int i = 0; i < HASH_SIZE; i++) {
        if (combined[i].name != NULL) {
            entries[count++] = combined[i];
        }
    }
    
    // 按平均耗时排序 (降序)
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - i - 1; j++) {
            double avg_j   = entries[j].call_count ? (entries[j].total_ns / entries[j].call_count) : 0;
            double avg_j1  = entries[j+1].call_count ? (entries[j+1].total_ns / entries[j+1].call_count) : 0;
            if (avg_j < avg_j1) {
                profile_entry_t temp = entries[j];
                entries[j] = entries[j+1];
                entries[j+1] = temp;
            }
        }
    }

    printf("\n===== GGML Profiler Sorted Report =====\n");
    printf("%-30s %-15s %-20s %-20s\n", "Function", "Calls", "Total (ms)", "Avg (us)");
    
    for (int i = 0; i < count; i++) {
        double total_ms = entries[i].total_ns / 1e6;
        double avg_us = entries[i].call_count ? entries[i].total_ns / entries[i].call_count / 1e3 : 0;
        printf("%-30s %-15ld %-20.3f %-20.3f\n",
               entries[i].name,
               entries[i].call_count,
               total_ms,
               avg_us);
    }
    printf("=============================================\n");
}


// Add a cleanup function to free allocated memory
void ggml_profiler_free(void) {
    pthread_mutex_lock(&all_thread_mutex);
    for (int i = 0; i < num_registered_threads; ++i) {
        free(all_thread_profilers[i]);
        all_thread_profilers[i] = NULL;
    }
    num_registered_threads = 0;
    pthread_mutex_unlock(&all_thread_mutex);
}
