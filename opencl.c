#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdbool.h> 
#include <limits.h>
#include <float.h>
#include <math.h>
#include <CL/cl.h>

#define SIZE			(1024)
#define WORKGROUP_SIZE  (256)
#define MAX_SOURCE_SIZE	(16384)
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#define PI 3.14159265
#define N_CONCENTRIC 23
#define N_TRAJECTORIES 3

double realW = 20000;
double realH = 20000;
double SENSOR_DENSITY = 3600;
int    N_SEED_CORRECTIONS = 30*30;
double TOLERANCE = 50 * 50;
double CENTER_TOLERANCE = 10;
double TRAJECTORY_ANGLE_TOLERANCE = PI / 4;
double GPU_SLICE_ANGLE = PI / 6;
double SEED_ANGLE_TOLERANCE = PI / 18;
double MIN_PERC_COVERAGE_FOR_TRAJ = 0.9;
double DETECTION_FAIL_RATE = 0;
bool   WITH_SENSORS = true;

// typedef struct _histogram
// {
//     unsigned int *R;
//     unsigned int *G;
//     unsigned int *B;
// } histogram;

double detections_x [N_CONCENTRIC][N_TRAJECTORIES] = 
{
    { -284.0741502493142, 242.11759521715774, 246.53956396858283 }, 
    { -429.78116605479255, 368.98414312611857, 375.5540833104356 },
    { -577.8911696478044, 500.73877774724724, 508.32368694731247 },
    { -728.3709698824001, 636.7513474701522, 644.7909042608296 },
    { -881.1870113228226, 776.9461041832825, 784.8971494911285 },
    { -1036.305378481565, 919.4043424324232, 928.5827361856593 },
    { -1191.8304905465952, 1067.4924270473593, 1075.7868918954691 },
    { -1351.2381973339072, 1219.5325399591984, 1226.4477731159407 },
    { -1512.8491008695964, 1375.4426102278676, 1380.5024804688749 },
    { -1927.6730167078151, 1778.5325770449492, 1778.5325770449494 },
    { -2352.7056376322275, 2206.5183635844014, 2195.8457367258497 },
    { -2793.38508166904, 2655.4608873064235, 2635.463030605256 },
    { -3242.212153413043, 3123.9029490082917, 3093.1487442313764 },
    { -3706.3421648724243, 3610.3358934423727, 3562.906003379987 },
    { -4176.639651925584, 4113.202590104299, 4052.765191766302 },
    { -4661.719323282076, 4636.07233304047, 4557.020979923081 },
    { -5156.438469008716, 5172.43045846175, 5079.947196829491 },
    { -5654.516685776853, 5720.411438892358, 5609.207100223201 },
    { -6166.235167437699, 6283.503201602818, 6154.632242860129 },
    { -6685.517747382228, 6854.250408193168, 6709.0847987851885 },
    { -7211.640936897965, 7430.476422975394, 7276.792290491385 },
    { -7749.977370405253, 8019.853394239645, 7844.433289857813 },
    { -8287.609321269236, 8609.035808170529, 8427.295971319409 }
};

double detections_y [N_CONCENTRIC][N_TRAJECTORIES] =
{
    { 281.6058897823873, 318.4008010138488, -314.98927505262947 },
    { 418.67427590500813, 473.1286316864411, -467.930689855729 },
    { 553.2104446258155, 623.9075864743054, -617.7434979733017 },
    { 685.1829903263593, 771.0692066831261, -764.3590058228539 },
    { 814.56089463953, 914.5243305644852, -907.7094605217579 },
    { 941.3135304085353, 1055.791482782658, -1047.728066846524 },
    { 1067.4924270473593, 1191.8304905465952, -1184.3490039789313 },
    { 1189.1826327632825, 1323.9110181506408, -1317.5074420355093 },
    { 1308.1619158185483, 1451.9495948446527, -1447.1395583768983 },
    { 1591.8783686753807, 1756.9353637484369, -1756.9353637484364 },
    { 1861.390926875258, 2032.5542332653306, -2044.0796218599 },
    { 2108.7910720383966, 2280.027955088661, -2303.1141123081507 },
    { 2342.660955465122, 2498.2454573514788, -2536.2237373819885 },
    { 2552.0634312040115, 2686.163572182764, -2748.763505847465 },
    { 2748.759941857953, 2842.8092536713193, -2928.3261943314046 },
    { 2918.625181646061, 2959.194708500388, -3079.538892194882 },
    { 3067.758516462313, 3040.717539064944, -3192.825782504301 },
    { 3205.688857370777, 3086.5665341588647, -3284.3257613710653 },
    { 3313.237670293904, 3085.0587539700336, -3334.741662430016 },
    { 3399.0958282221554, 3044.5445212352824, -3352.339655033407 },
    { 3462.980652163313, 2964.4595000786903, -3323.8974053127963 },
    { 3491.1102472145567, 2816.3720519318535, -3273.3570475844103 },
    { 3509.3491900936947, 2623.833541526142, -3159.221836431192 }
};

int N_POINTS = N_CONCENTRIC * N_TRAJECTORIES;
int N_LAYERS = N_CONCENTRIC;
int array_data[N_CONCENTRIC] = { 3,3,3,3,3,
                                    3,3,3,3,3,
                                    3,3,3,3,3,
                                    3,3,3,3,3,
                                    3,3,3 };

int main(int argc, char *argv[]) 
{
    int i;
	cl_int clStatus;

    // Read kernel from file
    FILE *fp;
    char *fileName = "C:\\Users\\Matic\\Documents\\Magistrska\\particle-trajectory-detection\\kernel.cl";
    char *source_str;
    size_t source_size;

    fp = fopen(fileName, "r");
    if (!fp) 
	{
		fprintf(stderr, ":-(#\n");
        exit(1);
    }

    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	source_str[source_size] = '\0';
    fclose( fp );
   
    // Get platforms
    cl_uint num_platforms;
    clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

    //Get platform devices
    cl_uint num_devices;
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    num_devices = 1; // limit to one device
    cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

    // Context
    cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &clStatus);
    
    // Command queue
    // cl_command_queue command_queue = clCreateCommandQueue(context, devices[0], 0, &clStatus);
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, devices[0], 0, &clStatus);

    // Create and build a program
    // Priprava programa
    cl_program program = clCreateProgramWithSource(context,	1, (const char **)&source_str, NULL, &clStatus);
    clStatus = clBuildProgram(program, 1, devices, NULL, NULL, NULL);

    // Log
    size_t build_log_len;
    char *build_log;
    clStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (build_log_len > 2)
    {
        build_log =(char *)malloc(sizeof(char)*(build_log_len+1));
        clStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 
                                        build_log_len, build_log, NULL);
        printf("%s", build_log);
        free(build_log);
        return 1;
    }

    

    // flatten the array of detections
    double* detections_x_flattened;
    double* detections_y_flattened;

    detections_x_flattened = (double*)malloc(N_POINTS * sizeof(double));
    detections_y_flattened = (double*)malloc(N_POINTS * sizeof(double));

    int* trajectory_count;
    double* trajectory_radii;
    double* trajectory_centers_x;
    double* trajectory_centers_y;

    int approximated_trajectories = N_POINTS / N_LAYERS * 2;

    trajectory_count = (int*)malloc(1 * sizeof(int));
    *trajectory_count = 0;
    trajectory_centers_x = (double*)malloc(approximated_trajectories * sizeof(double));
    trajectory_centers_y = (double*)malloc(approximated_trajectories * sizeof(double));
    trajectory_radii = (double*)malloc(approximated_trajectories * sizeof(double));

    int index = 0;

    for (int i = 0; i < N_CONCENTRIC; i++)
    {
        for (int j = 0; j < array_data[i]; j++)
        {
            detections_x_flattened[index] = detections_x[i][j];
            detections_y_flattened[index] = detections_y[i][j];
            index++;
        }
    }

    // Divide work
    size_t global_work_size = 12 * 256;
    size_t local_work_size = 256;

    // allocate memory on device and transfer data from host 
    cl_mem det_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        N_POINTS * sizeof(double), detections_x_flattened, &clStatus);
    cl_mem det_y = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        N_POINTS * sizeof(double), detections_y_flattened, &clStatus);
    cl_mem arr_data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        N_LAYERS * sizeof(int), array_data, &clStatus);

    cl_mem traj_x = clCreateBuffer(context, CL_MEM_READ_WRITE,
        approximated_trajectories * sizeof(double), trajectory_centers_x, &clStatus);
    cl_mem traj_y = clCreateBuffer(context, CL_MEM_READ_WRITE,
        approximated_trajectories * sizeof(double), trajectory_centers_y, &clStatus);
    cl_mem traj_r = clCreateBuffer(context, CL_MEM_READ_WRITE,
        approximated_trajectories * sizeof(double), trajectory_radii, &clStatus);
    cl_mem traj_cnt = clCreateBuffer(context, CL_MEM_READ_WRITE,
        1 * sizeof(int), trajectory_count, &clStatus);
    
    // create kernel and set arguments
    cl_kernel kernel = clCreateKernel(program, "trajectory_calculation", &clStatus);
    clStatus =  clSetKernelArg(kernel, 0, sizeof(cl_mem),             (void*)&det_x);
    clStatus |= clSetKernelArg(kernel, 1, sizeof(cl_mem),             (void*)&det_y);
    clStatus |= clSetKernelArg(kernel, 2, sizeof(cl_mem),             (void*)&arr_data);
    clStatus |= clSetKernelArg(kernel, 3, N_POINTS * sizeof(double),  NULL);
    clStatus |= clSetKernelArg(kernel, 4, N_POINTS * sizeof(double),  NULL);
    clStatus |= clSetKernelArg(kernel, 5, N_LAYERS * sizeof(cl_int),  NULL);
    clStatus |= clSetKernelArg(kernel, 6, sizeof(cl_int),             (void*)&N_POINTS);
    clStatus |= clSetKernelArg(kernel, 7, sizeof(cl_int),             (void*)&N_LAYERS);
    clStatus |= clSetKernelArg(kernel, 8, sizeof(cl_int),             (void*)&GPU_SLICE_ANGLE);

    clStatus |= clSetKernelArg(kernel, 9,  sizeof(cl_mem), (void*)&traj_x);
    clStatus |= clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&traj_y);
    clStatus |= clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*)&traj_r);
    clStatus |= clSetKernelArg(kernel, 12, sizeof(cl_mem), (void*)&traj_cnt);


    // Execute kernel
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

    clFinish(command_queue);
                                                                                            
    // Copy results back to host
    // clStatus = clEnqueueReadBuffer(command_queue, R_gpu, CL_TRUE, 0, 256*sizeof(unsigned int), H.R, 0, NULL, NULL);
    // clStatus = clEnqueueReadBuffer(command_queue, G_gpu, CL_TRUE, 0, 256*sizeof(unsigned int), H.G, 0, NULL, NULL);
    // clStatus = clEnqueueReadBuffer(command_queue, B_gpu, CL_TRUE, 0, 256*sizeof(unsigned int), H.B, 0, NULL, NULL);

    // release & free
    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    // clStatus = clReleaseMemObject(R_gpu);
    // clStatus = clReleaseMemObject(G_gpu);
    // clStatus = clReleaseMemObject(B_gpu);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    free(devices);
    free(platforms);
    //free(image_in);

    return 0;
}
