#include "kernel.h"
#include "common.h"
#include "timer.h"
#include "verify.h"
#include "stdio.h"
#include <string.h>
#include <thread>
#include <assert.h>
// Params ---------------------------------------------------------------------
struct Params {

    int         platform;
    int         device;
    int         n_work_items;
    int         n_work_groups;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    float       alpha;
    const char *file_name;
    int         max_iter;
    int         error_threshold;
    float       convergence_threshold;

    Params(int argc, char **argv) {
        platform              = 0;
        device                = 0;
        n_work_items          = 256;
        n_work_groups         = 64;
        n_threads             = 4;
        n_warmup              = 5;
        n_reps                = 50;
        alpha                 = 0;
        file_name             = "v.csv";
        max_iter              = 2000;
        error_threshold       = 3;
        convergence_threshold = 0.75;
       
    }

};

// Input ----------------------------------------------------------------------
int read_input_size(const Params &p) {
    FILE *File = NULL;
    File       = fopen(p.file_name, "r");
    if(File == NULL) {
        puts("Error al abrir el fichero");
        exit(-1);
    }

    int n;
    fscanf(File, "%d", &n);

    fclose(File);
//	printf("%d",n);5922 items
//	getchar();
    return n;
}

void read_input(flowvector *v, int *r, const Params &p) {

    int ic = 0;

    // Open input file
    FILE *File = NULL;
    File       = fopen(p.file_name, "r");
    if(File == NULL) {
        puts("Error opening file!");
        exit(-1);
    }

    int n;
    fscanf(File, "%d", &n);

    while(fscanf(File, "%d,%d,%d,%d", &v[ic].x, &v[ic].y, &v[ic].vx, &v[ic].vy) == 4) {
        ic++;// row
        if(ic > n) {
            puts("Error: inconsistent file data!");
            exit(-1);
        }
    }
    if(ic < n) {
        puts("Error: inconsistent file data!");
        exit(-1);
    }

    srand(time(NULL));
    for(int i = 0; i < 2 * p.max_iter; i++) {
        r[i] = ((int)rand()) % n;
    }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
	
    const Params p(argc, argv);
    // Allocate
    
    int n_flow_vectors = read_input_size(p);
    int best_model     = -1;
    int best_outliers  = n_flow_vectors;

    Timer timer =  Timer();

    flowvector *     h_flow_vector_array  = (flowvector *)malloc(n_flow_vectors * sizeof(flowvector));
    int *            h_random_numbers     = (int *)malloc(2 * p.max_iter * sizeof(int));
    int *            h_model_candidate    = (int *)malloc(p.max_iter * sizeof(int));
    int *            h_outliers_candidate = (int *)malloc(p.max_iter * sizeof(int));
    //float *          h_model_param_local  = (float *)malloc(4 * p.max_iter * sizeof(float));
    std::atomic_int *h_g_out_id           = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    // Initialize
    read_input(h_flow_vector_array, h_random_numbers, p);
    //clFinish(ocl.clCommandQueue);



    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {//ִ��55�Σ��ֵ�ϸ����
}    
	
    
        // Reset
        memset((void *)h_model_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)h_outliers_candidate, 0, p.max_iter * sizeof(int));
        //memset((void *)h_model_param_local, 0, 4 * p.max_iter * sizeof(float));

        h_g_out_id[0] = 0;
        // Launch FPGA threads

        timer.start("Thread");
        // Launch CPU threads
		Thread_params *p2 = (Thread_params *)malloc(sizeof(Thread_params));//Ϊɶ��ô���壿��
		p2[0].model_candidate		= h_model_candidate;
		p2[0].outliers_candidate	= h_outliers_candidate;
		//p2[0].model_param_local		= h_model_param_local;
		p2[0].flowvectors			= h_flow_vector_array;
		p2[0].flowvector_count		= n_flow_vectors;
		p2[0].random_numbers		= h_random_numbers;
		p2[0].max_iter				= p.max_iter;
		p2[0].error_threshold		= p.error_threshold;
		p2[0].convergence_threshold	= p.convergence_threshold;
		p2[0].g_out_id				= h_g_out_id;
		p2[0].n_threads				= p.n_threads;
		p2[0].n_tasks				= p.max_iter;
		p2[0].alpha					= p.alpha;


		
		std::thread main_thread(run_cpu_threads, p2);


		main_thread.join();
        timer.stop("Thread");

        // Copy back
        
		int d_candidates = 0;
		
		h_g_out_id[0] += d_candidates;




        // Post-processing (chooses the best model among the candidates)
        for(int i = 0; i < h_g_out_id[0]; i++) {
            if(h_outliers_candidate[i] < best_outliers) {
                best_outliers = h_outliers_candidate[i];
                best_model    = h_model_candidate[i];
            }
        }
    

	int iter = best_model;
	flowvector fv[2];
	int rand_num = h_random_numbers[iter * 2 + 0];
	fv[0]        = h_flow_vector_array[rand_num];
    rand_num     = h_random_numbers[iter * 2 + 1];
	fv[1]        = h_flow_vector_array[rand_num];                                                                                                                                 


	printf("\nThread result:\n");
	printf("randnum %d fv0 (%d,%d)  (%d,%d) \n",h_random_numbers[iter * 2 + 0],fv[0].x,fv[0].y,fv[0].vx,fv[0].vy);
	printf("randnum %d fv1 (%d,%d)  (%d,%d) \n",h_random_numbers[iter * 2 + 1],fv[1].x,fv[1].y,fv[1].vx,fv[1].vy);
	printf("%d %d \n",best_model,best_outliers);
    timer.print("Thread",1);



    // Verify answer
    timer.start("Verify");
    verify(h_flow_vector_array, n_flow_vectors, h_random_numbers, p.max_iter, p.error_threshold,
        p.convergence_threshold, h_g_out_id[0], best_outliers);
    timer.stop("Verify");
    timer.print("Verify",1);
    // Free memory   


    free(h_model_candidate);
    free(h_outliers_candidate);
    //free(h_model_param_local);
    free(h_g_out_id);
    free(h_flow_vector_array);
    free(h_random_numbers);

    //printf("Test Passed\n");
    return 0;
}