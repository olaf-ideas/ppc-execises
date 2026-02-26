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
#include <cassert>

static inline int divup(int a, int b) {
    return (a + b - 1) / b;
}

static inline int roundup(int a, int b) {
  return divup(a, b) * b;
}

static inline void check(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << context << ": "
      << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(x) check(x, #x)

__global__ void matmul_kernel(const int my, const int mx, const int ny, const int nx, const float *d, float* r) {
  const int it = threadIdx.x;
  const int jt = threadIdx.y;
  const int ib = blockIdx.x;
  const int jb = blockIdx.y;
 
  ///*
  if (ib < jb) {
    return; 
   /* for (int ic = 0; ic < 8; ic++) {
      for (int jc = 0; jc < 8; jc++) {
        const int i = 64*ib + 8*ic + it;
        const int j = 64*jb + 8*jc + jt;

        if (i < ny && j < ny) {
          r[ny*j + i] = 0;
        }
      }
    }
   */ 
    return;
  }//*/
   
  __shared__ float di[2][64];
  __shared__ float dj[2][64];

  float v[8][8];
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      v[i][j] = 0;
    }
  }

  for (int ks = 0; ks < mx; ks += 2) {
    const int ij = 8*jt + it;
    const int i = 64*ib + ij;
    const int j = 64*jb + ij;
    #pragma unroll
    for (int f = 0; f < 2; f++) {
      const int k = ks + f;
      di[f][ij] = d[my*k + i];
      dj[f][ij] = d[my*k + j];
    }

    __syncthreads();

    #pragma unroll
    for (int f = 0; f < 2; f++) {
      float dj_reg[8];
      #pragma unroll
      for (int jc = 0; jc < 8; jc++) {
        dj_reg[jc] = dj[f][8*jc + jt];
      }
      #pragma unroll
      for (int ic = 0; ic < 8; ic++) {
        const float di_reg = di[f][8*ic + it];
        #pragma unroll
        for (int jc = 0; jc < 8; jc++) {
          v[ic][jc] += di_reg * dj_reg[jc];
        }
      }
    }

    __syncthreads();
  }

  for (int ic = 0; ic < 8; ic++) {
    for (int jc = 0; jc < 8; jc++) {
      const int i = 64*ib + 8*ic + it;
      const int j = 64*jb + 8*jc + jt;

      if (i < ny && j < ny) {
        r[ny*j + i] = v[ic][jc] / nx;
      }
    }
  }
}

__global__ void norm_kernel(const int ny, const int nx, const int my, const int mx, const float* d, float* dk) {
  const int y = threadIdx.x + blockIdx.x * blockDim.x;

  if (y >= ny) {
    for (int x = 0; x < mx; x++) {
      dk[my*x + y] = 0;
    }
    return;
  }

  float sum = 0;
  for (int x = 0; x < nx; x++) {
    sum += d[nx*y + x];
  }

  float avg = sum / nx;
  float std = 0;
  for (int x = 0; x < nx; x++) {
    float dd = d[nx*y + x];
    std += (dd - avg) * (dd - avg);
  }

  std = sqrt(std / nx);
  for (int x = 0; x < nx; x++) {
    dk[my*x + y] = (d[nx*y + x] - avg) / std;
  }
  for (int x = nx; x < mx; x++) {
    dk[my*x + y] = 0;
  }
}

void correlate(int ny, int nx, const float *data, float *result) {
  int mx = roundup(nx, 2); 
  int my = roundup(ny, 64);

  float* dataGPU = NULL;
  float* resultGPU = NULL;
  float* dataKernel = NULL;
  
  CHECK(cudaMalloc((void**)&dataGPU, ny * nx * sizeof(float)));
  CHECK(cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float)));
  CHECK(cudaMalloc((void**)&dataKernel, my * mx * sizeof(float)));
  CHECK(cudaMemcpy(dataGPU, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(resultGPU, 0, ny * ny * sizeof(float)));

  {
    norm_kernel<<<64, my / 64>>>(ny, nx, my, mx, dataGPU, dataKernel);
    CHECK(cudaGetLastError());
  }

  {
    dim3 dimBlock(8, 8);
    dim3 dimGrid(my / 64, my / 64);
    matmul_kernel<<<dimGrid, dimBlock>>>(my, mx, ny, nx, dataKernel, resultGPU);
    CHECK(cudaGetLastError());
  }
  
  CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(dataKernel));
  CHECK(cudaFree(resultGPU));
  CHECK(cudaFree(dataGPU));
}
