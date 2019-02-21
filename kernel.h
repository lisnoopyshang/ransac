#include <atomic>

#include "common.h"


typedef struct {
    int *model_candidate;
	int *outliers_candidate;
	flowvector *flowvectors;
    int flowvector_count;
	int *random_numbers;
	int max_iter;
	int error_threshold;
	float convergence_threshold;
    std::atomic_int *g_out_id;
	int n_threads;
	int n_tasks;
	float alpha;
} Thread_params;

void run_cpu_threads(Thread_params *p2);
