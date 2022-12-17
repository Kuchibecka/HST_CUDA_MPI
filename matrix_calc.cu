#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <strings.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); } //ошибки
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void matrixCalc(int *a, int *b, int *res, int N) {//cчитываем колонку и столбец

    int col = blockIdx.x * blockDim.x + threadIdx.x;//при инициализации создаются сами,номер колонки номер столбца
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < N && row < N / 2) {
        res[row * N + col] = a[row * N + col] * b[row];
    }
}

extern "C" void * launch_counter(int res_buf, int * matrix, int * arr, int * res_arr) {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int *a, *b, *res;

    size_t bytes = res_buf * res_buf * sizeof(int) / 2;
    size_t bytes_res = res_buf * sizeof(int) / 2;

    printf("INFO: Start counting cuda\n");

    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&a), bytes));//выделение памяти на видеокарте
    gpuErrchk(cudaMemcpyAsync(a, matrix, bytes, cudaMemcpyHostToDevice));// а это матрица,копирование массива из оперативки в память видеокарты
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&b), bytes_res));
    gpuErrchk(cudaMemcpyAsync(b, arr, bytes_res, cudaMemcpyHostToDevice));// маленький массив

    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&res), bytes));// выделение памяти для рез. матрицы

    int block_size = 16;
    int grid_size = (res_buf + block_size) / block_size;// непросто выделить память. количестов блоков

    dim3 DimGrid(grid_size, grid_size,1);//создаем переменные грид и блок
    dim3 DimBlock(block_size, block_size,1);

    matrixCalc<<<DimGrid,DimBlock>>>(a, b, res, res_buf);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    printf("INFO: Stop counting cuda\n");

    gpuErrchk(cudaMemcpyAsync(res_arr, res, bytes, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree( res ));
    gpuErrchk(cudaFree( a ));
    gpuErrchk(cudaFree( b ));
}

// nvcc -arch=sm_35 -c matrix_calc.cu -o matrix_calc.o