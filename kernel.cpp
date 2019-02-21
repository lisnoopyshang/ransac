#include "kernel.h"
#include "partitioner.h"

#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>

#include "stdlib.h"
#include "stdio.h"
// Function to generate model parameters for first order flow (xc, yc, D and R)
int gen_model_param2(int x1, int y1, int vx1, int vy1, int x2, int y2, int vx2, int vy2, float *model_param) {   
   float temp;
    // xc -> model_param[0], yc -> model_param[1], D -> model_param[2], R -> model_param[3]
    temp = (float)((vx1 * (vx1 - (2 * vx2))) + (vx2 * vx2) + (vy1 * vy1) - (vy2 * ((2 * vy1) - vy2)));
    if(temp == 0) { // Check to prevent division by zero
        return (0);
    }
    model_param[0] = (((vx1 * ((-vx2 * x1) + (vx1 * x2) - (vx2 * x2) + (vy2 * y1) - (vy2 * y2))) +
                          (vy1 * ((-vy2 * x1) + (vy1 * x2) - (vy2 * x2) - (vx2 * y1) + (vx2 * y2))) +
                          (x1 * ((vy2 * vy2) + (vx2 * vx2)))) /
                      temp);
    model_param[1] = (((vx2 * ((vy1 * x1) - (vy1 * x2) - (vx1 * y1) + (vx2 * y1) - (vx1 * y2))) +
                          (vy2 * ((-vx1 * x1) + (vx1 * x2) - (vy1 * y1) + (vy2 * y1) - (vy1 * y2))) +
                          (y2 * ((vx1 * vx1) + (vy1 * vy1)))) /
                      temp);

    temp = (float)((x1 * (x1 - (2 * x2))) + (x2 * x2) + (y1 * (y1 - (2 * y2))) + (y2 * y2));
    if(temp == 0) { // Check to prevent division by zero
        return (0);
    }
    model_param[2] = ((((x1 - x2) * (vx1 - vx2)) + ((y1 - y2) * (vy1 - vy2))) / temp);
    model_param[3] = ((((x1 - x2) * (vy1 - vy2)) + ((y2 - y1) * (vx1 - vx2))) / temp);
    return (1); 
}




// CPU threads--------------------------------------------------------------------------------------
void run_cpu_threads(Thread_params *p2) {

    std::vector<std::thread> cpu_threads;
    for(int k = 0; k < p2[0].n_threads; k++) {
        cpu_threads.push_back(std::thread([=]() { // [=] 

            Partitioner p = partitioner_create(p2[0].n_tasks, p2[0].alpha , k, p2[0].n_threads);
            int n_thread = p2[0].n_threads;
            flowvector fv[2];
            int  vx_error, vy_error;
            int  outlier_local_count = 0;

            for(int iter = cpu_first(&p); cpu_more(&p); iter = cpu_next(&p)) {
				float model_param[4] = {0,0,0,0};
                // Select four random flow vectors

				int rand_num = p2[0].random_numbers[iter * 2 + 0];
                fv[0]        = p2[0].flowvectors[rand_num];
                rand_num     = p2[0].random_numbers[iter * 2 + 1];
                fv[1]        = p2[0].flowvectors[rand_num];


                //Initialization   


                int ret = 0;
                int vx1 = fv[0].vx - fv[0].x;
                int vy1 = fv[0].vy - fv[0].y;
                int vx2 = fv[1].vx - fv[1].x;
                int vy2 = fv[1].vy - fv[1].y;
                


                // Function to generate model parameters according to F-o-F (xc, yc, D and R)
                ret = gen_model_param2(fv[0].x, fv[0].y, vx1, vy1,fv[1].x, fv[1].y, vx2, vy2, model_param);
                if(ret == 0)
                    model_param[0] = -2011;
                if(model_param[0] == -2011)
                    continue;                


                // SIMD phase
                // Reset local outlier counter
                outlier_local_count = 0;
                // Compute number of outliers
               for(int i = 0; i < p2[0].flowvector_count; i++) {//5000*20 ,

                    flowvector fvreg = p2[0].flowvectors[i]; // x, y, vx, vy                
                    vx_error         = fvreg.x + ((int)((fvreg.x - model_param[0]) * model_param[2]) -
                                             (int)((fvreg.y - model_param[1]) * model_param[3])) -
                               fvreg.vx;
                    vy_error = fvreg.y + ((int)((fvreg.y - model_param[1]) * model_param[2]) +
                                             (int)((fvreg.x - model_param[0]) * model_param[3])) -
                               fvreg.vy;
                
                    if((fabs(vx_error) >= p2[0].error_threshold) || (fabs(vy_error) >= p2[0].error_threshold)) {
                        outlier_local_count++;
                    }
                }

                // Compare to threshold 
                if(outlier_local_count < p2[0].flowvector_count * p2[0].convergence_threshold  ) {
                        int index                 = p2[0].g_out_id->fetch_add(1);
                        p2[0].model_candidate[index]    = iter;
                        p2[0].outliers_candidate[index] = outlier_local_count; 
                }                     

            }

        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}