/* Matrix normalization.
 * Compile with "gcc matrixNorm.c"
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

/* Program Parameters */
#define N 9000  /* Matrix size */

/* Matrices */
volatile float A[N][N], B[N][N];

// Flattened array A
float flattenA[N * N], flattenB[N * N];

/* Initialize A and B*/
void initialize_inputs() {
    int row, col;

    srand(22);
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            A[row][col] = (float)rand() / 32768.0;
            B[row][col] = 0.0;
        }
    }
}

// Printing array func
//  This will help us in both seeing inputs/results before and after normalization
void print_arrays() {
    int row, col;

    if (N < 10) {
        printf("\nA =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }
        printf("\nB =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%5.2f%s", B[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }
        printf("\n");
    }
}

// Even though CUDA API has cudaMemcpy2D, I found it much easier
//  to convert a 2D array into a 1D with mapping scheme like this
void flattenArray() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            flattenA[i * N + j] = A[i][j];
            flattenB[i * N + j] = B[i][j];
        }
    }
}

// This function basically prints out a 1D array to console
void checkFlatten(float targetArray[]) {
    if (N < 10) {
        printf("---- Checking ----\n");
        for (int i = 0; i < (N * N); i++) {
            if (i % N == 0 && i != 0) {
                printf("\n");
            }
            printf("%5.2f ", targetArray[i]);
        }
        printf("\n");
    }
}

// CUDA device information
void getCudaDevices(int nDevices) {
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}

// GPU parallel matrix normalization kernel
__global__ void gpuMatrixNorm(float *flattenA, float *flattenB, int arraySize) {
    float mu, sigma;

    // Index when inside GPU i.e. threadID
    //  After flattenning, we can access a pseudo 2D array in the same way
    //  we flatten it i.e. A[i][j] == flattenA[i * arraySize + idx]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // If clause here is to prevent faulty computation tasks
    //  where gpu processes an index that is beyond the scope
    //  of the vector A
    if (idx < arraySize) {
        // Mean
        mu = 0.0;
        for (int row = 0; row < arraySize; row++) {
            mu += flattenA[row * arraySize + idx];
        }
        mu /= (float) arraySize;

        // Wait here until every mean computations have arrived
        //  Once arrived, then continue to Standard deviation
        //  syncthreads == barrier
        __syncthreads();

        // Standard deviation
        sigma = 0.0;
        for (int row = 0; row < arraySize; row++) {
            sigma += powf((flattenA[row * arraySize + idx] - mu), 2.0);
        }
        sigma /= (float) arraySize;

        // Wait here until every Standard deviation computations have arrived
        //  Once arrived, then continue to compute the final result for B
        //  syncthreads == barrier
        __syncthreads();
        sigma = sqrt(sigma);

        for (int row = 0; row < arraySize; row++) {
            if (sigma == 0.0) {
                flattenB[row * arraySize + idx] = 0.0;
            }
            else {
                flattenB[row * arraySize + idx] = (flattenA[row * arraySize + idx] - mu) / sigma;
            }
        }
    }
}

int main(int argc, char **argv) {
    // Variables for CUDA
    float *device_A, *device_B;
    int nDevices;
    int cudaRunTimeVersion;
    int cudaDriverVersion;

    /* Timing variables */
    struct timeval startGPU, stopGPU;  /* Elapsed times using gettimeofday() */
    struct timezone tzdummy;
    unsigned long long runtimeGPU;

    /* Initialize A and B */
    initialize_inputs();

    // Sanity check after inputs initialization
    printf("---- Initialized inputs ----\n");
    print_arrays();

    // Flatten 2D array A & B
    flattenArray();

    // Sanity check after flattening
    //  Usually commented this out ... I only un-conmment it to validate the flattening process went ok
    checkFlatten(flattenA);

    // After flattening, size of array flattenA will be N * N
    int arraySize = sizeof(float) * N * N;

    // Cuda device info
    cudaGetDeviceCount(&nDevices);
    getCudaDevices(nDevices);
    cudaRuntimeGetVersion(&cudaRunTimeVersion);
    cudaDriverGetVersion(&cudaDriverVersion);

    printf("Cuda Runtime Version: %i\n", cudaRunTimeVersion);
    printf("Cuda Driver Version: %i\n", cudaDriverVersion);

    // Start Clock GPU
    printf("---------------------------------------------\n");
    printf("Matrix size N = %d", N);
    printf("\nStarting clock for GPU.\n\n");
    gettimeofday(&startGPU, &tzdummy);

    // Allocating space for GPU device
    cudaMalloc((void**)&device_A, arraySize);
    cudaMalloc((void**)&device_B, arraySize);

    // Copying array A from HOST to GPU
    cudaMemcpy(device_A, flattenA, arraySize, cudaMemcpyHostToDevice);

    // Launch GPU kernel gpuMatrixNorm
    gpuMatrixNorm<<<N, N>>>(device_A, device_B, N);

    // Copying array B from GPU to HOST
    //  Initially I had cudaDeviceSynchronize() before copying B from device to host
    //  However, by reading CUDA's doc further, cudaMemcpy is a blocking method
    cudaMemcpy(flattenB, device_B, arraySize, cudaMemcpyDeviceToHost);

    /* Stop Clock */
    gettimeofday(&stopGPU, &tzdummy);
    runtimeGPU = (unsigned long long)(stopGPU.tv_sec - startGPU.tv_sec) * 1000000 + (stopGPU.tv_usec - startGPU.tv_usec);

    /* Display timing results */
    printf("GPU Runtime = %g ms.\n", (float)runtimeGPU/(float)1000);
    printf("\nStopped clock for GPU.");
    printf("\n---------------------------------------------\n");
    printf("---- Results ----\n");

    // Sanity check the result after computes by GPU and deliver back to host machine
    //  Usually commented this out ... I only un-comment it to validate the computed result went ok
    checkFlatten(flattenB);

    // Freeing memory in GPU device
    cudaFree(device_A);
    cudaFree(device_B);
    exit(0);
}