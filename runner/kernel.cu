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
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
    curand_init(seedRad, idx, 0, &states[idx]);
    curand_init(seedPhi, idx + 1, 0, &states[idx + 1]);
};

__global__ void generate_random(curandState_t* states, float3* results, const int samples) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stateIdx = idx * 2;
    while (idx < samples) {
        float r = curand_uniform(&states[stateIdx]);
        float phi = curand_uniform(&states[stateIdx + 1]) * 2 * CUDART_PI_F;
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
    const int samples = 134217728;// 33554432; // 32M rays
    int TRIALS = 1000;
    StopWatchInterface *hTimer;

    cudaSetDevice(0);
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    CUDA_CALL(cudaMalloc((void **)&states, sizeof(BLOCKS * TPB * sizeof(curandState_t) * 2)));

    random_init << <BLOCKS, TPB >> >(states, 0l, 1234l);
    checkCudaErrors(cudaPeekAtLastError());

    float3* devRaysDirection;
    CUDA_CALL(cudaMalloc((void **)&devRaysDirection, (size_t)samples * sizeof(float3)));

    sdkCreateTimer(&hTimer);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (int i = 0; i < TRIALS; ++i) {
        generate_random << <BLOCKS, TPB >> >(states, devRaysDirection, samples);
    }
    sdkStopTimer(&hTimer);
    checkCudaErrors(cudaPeekAtLastError());
    //checkCudaErrors(cudaDeviceSynchronize());

    printf("%f Gigarays/s\n", (float)TRIALS * samples * 1e-9 / sdkGetTimerValue(&hTimer));

    //float3* rays = new float3[samples];
    //CUDA_CALL(cudaMemcpy(rays, devRaysDirection, sizeof(float3) * samples, cudaMemcpyDeviceToHost));
    //delete rays;
    
    cudaFree(devRaysDirection);

    cudaFree(states);

    return EXIT_SUCCESS;
}