/*
* This program uses the device CURAND API to calculate what
* proportion of pseudo-random ints have low bit set.
*/
#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <math_constants.h>
#include <device_launch_parameters.h>
#include "math.cuh"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void random_init(curandState_t* states, unsigned long long seedRad, unsigned long long seedPhi) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    curand_init(seedRad, idx, 0, &states[idx]);
    curand_init(seedPhi, idx + 1, 0, &states[idx + 1]);
};

__global__ void generate_random(curandState_t* states, float3* results, const int samples) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t stateRad = states[idx * 2];
    curandState_t statePhi = states[idx * 2 + 1];
    while (idx < samples) {
        float r = curand_uniform(&stateRad);
        float phi = curand_uniform(&statePhi) * 2 * CUDART_PI_F;
        float rad = sqrtf(r);
        results[idx] = make_float3(rad * cosf(phi), rad * sinf(phi), sqrtf(1 - r));
        idx += blockDim.x * gridDim.x;
    }
};

int main(int argc, char *argv[]) {
    atexit([] { _getch(); });

    int devCount;
    cudaGetDeviceCount(&devCount);
    for (int i = 0; i < devCount; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("%i: %s (compat %i.%i)\n", i, props.name, props.major, props.minor);
    }

    curandState_t* states;
    int BLOCKS = 256;
    int TPB = 512;
    const int samples = BLOCKS * TPB * 256;
    printf("Trials: %i, %i Mb\n", samples, (samples * sizeof(float3) + BLOCKS * TPB * 2 * sizeof(curandState_t)) / 1024 / 1024);
    int TRIALS = 10;
    StopWatchInterface *hTimer;

    cudaSetDevice(0);
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    checkCudaErrors(cudaMalloc((void **)&states, BLOCKS * TPB * sizeof(curandState_t) * 2));

    printf("Initializing random\n");
    random_init << <BLOCKS, TPB >> >(states, 0l, 1234l);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaPeekAtLastError());
    printf("Initializing random done\n");

    float3* devRaysDirection;
    checkCudaErrors(cudaMalloc((void **)&devRaysDirection, (size_t)samples * sizeof(float3)));

    sdkCreateTimer(&hTimer);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (int i = 0; i < TRIALS; ++i) {
        generate_random << <BLOCKS, TPB >> >(states, devRaysDirection, samples);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaPeekAtLastError());
        printf("Done trial %i\n", i);
    }
    sdkStopTimer(&hTimer);

    printf("%f Grays/s\n", (float)TRIALS * samples * 1e-9 * 1e+3 / sdkGetTimerValue(&hTimer));

    //float3* rays = new float3[samples];
    //CUDA_CALL(cudaMemcpy(rays, devRaysDirection, sizeof(float3) * samples, cudaMemcpyDeviceToHost));
    //delete rays;
    
    cudaFree(devRaysDirection);

    cudaFree(states);

    return EXIT_SUCCESS;
}