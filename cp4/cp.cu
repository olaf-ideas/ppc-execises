/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

static inline int divup(int a, int b) {
    return (a + b - 1) / b;
}

static inline void check(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << context << ": "
      << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(x) check(x, #x)

__global__ void kernel(int ny, int nx, const float *d, float* r) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i >= ny || j >= ny) {
    return;
  }

  float v = 0;
  for (int k = 0; k < nx; k++) {
    float x = d[nx*i + k];
    float y = d[nx*j + k];
    v += x * y;
  }

  r[ny*i + j] = v;
}

void correlate(int ny, int nx, const float *data, float *result) {
  float* _data = (float*) malloc(sizeof(float) * ny * nx);

  for (int y = 0; y < ny; y++) {
    float sum = 0;
    for (int x = 0; x < nx; x++) {
      sum += data[nx*y + x];
    }
    float avg = sum / nx, std = 0;
    for (int x = 0; x < nx; x++) {
      float d = data[nx*y + x];
      std += (d - avg) * (d - avg);
    }
    std = sqrt(std / (nx - 1));

    for (int x = 0; x < nx; x++) {
      _data[nx*y + x] = (data[nx*y + x] - avg) / std;
    }
  }

  float* dataGPU = NULL;
  CHECK(cudaMalloc((void**)&dataGPU, nx * ny * sizeof(float)));
  CHECK(cudaMemcpy(dataGPU, _data, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

  float* resultGPU = NULL;
  CHECK(cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float)));

  dim3 dimBlock(16, 16);
  dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
  kernel<<<dimGrid, dimBlock>>>(ny, nx, dataGPU, resultGPU);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(dataGPU));
  CHECK(cudaFree(resultGPU));

  free(_data);

  for (int j = 0; j < ny; j++) {
    for (int i = j; i < ny; i++) {
      result[j*ny + i] /= (nx - 1);
    }
  }
}
