#include <cstdlib>
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>
#include <cmath>

#define INLINE inline __attribute__ ((always_inline))

using u32 = std::uint32_t;
using f32 = float;
using f32x16 __attribute__ (( vector_size((64)) )) = f32;

static constexpr u32 B1 = 2*112; // 2048;
static constexpr u32 B2 = 4;

template <u32 A> u32 roundup(u32 v) { return ((v + A - 1) / A) * A; }

// data[x + y*nx]
// result[i + j*ny] for 0 <= j <= i < ny
void correlate(int _ny, int _nx, const float *data, float *result) {
  const u32 ny = _ny;
  const u32 nx = _nx;

  if (ny * nx < 100) {
    alignas(64) static f32 _data[100];
    for (u32 y = 0; y < ny; y++) {
      f32 sum = 0;
      for (u32 x = 0; x < nx; x++)
        sum += data[x + y*nx];
      f32 avg = sum / nx;
      f32 std = 0;
      for (u32 x = 0; x < nx; x++) {
        f32 d = data[x + y*nx];
        std += (d - avg) * (d - avg);
      }
      std = sqrt(std / (nx - 1));

      for (u32 x = 0; x < nx; x++)
        _data[x + y*nx] = (data[x + y*nx] - avg) / std;
    }

    for (u32 j = 0; j < ny; j++) {
      for (u32 i = j; i < ny; i++) {
        f32 dot = 0;
        for (u32 x = 0; x < nx; x++) {
          dot += _data[x + i*nx] * _data[x + j*nx];
        }
        result[i + j*ny] = dot / (nx - 1);
      }
    }
    return;
  }

  const u32 my = roundup<8>(roundup<B2>(ny));
  const u32 mx = roundup<8>(roundup<B1>(nx));

  alignas(64) static f32 __data[5100 * 5100];
  alignas(64) static f32 __result[5100 * 5100];

  f32* _data = __data;
  f32* _result = __result;
  
  if (my * std::max(my, mx) > 5100 * 5100) {
    _data = static_cast<f32*>(std::aligned_alloc(64, my * mx * sizeof(f32)));
    _result = static_cast<f32*>(std::aligned_alloc(64, my * my * sizeof(f32)));
  }

  #pragma omp parallel for
  for (u32 y = 0; y < ny; y++) {
    f32 sum = 0;
    for (u32 x = 0; x < nx; x++) {
      f32 d = data[x + y*nx];
      sum += d;
    }

    f32 avg = sum / nx;
    f32 std = 0;
    for (u32 x = 0; x < nx; x++) {
      f32 d = data[x + y*nx];
      std += (d - avg) * (d - avg);
    }

    std = sqrt(std / (nx - 1));

    for (u32 x = 0; x < nx; x++)
      _data[x + y*mx] = (data[x + y*nx] - avg) / std;
    for (u32 x = nx; x < mx; x++)
      _data[x + y*mx] = 0;
  }

  #pragma omp parallel for
  for (u32 j = 0; j < ny; j++) {
    for (u32 i = j; i < ny; i++) {	
      _result[i + j*my] = 0;
    }
  }

  for (u32 x = 0; x < mx; x += B1) {
    for (u32 j = 0; j < my; j += B2) {
      #pragma omp parallel for
      for (u32 i = j; i < my; i += B2) {
        f32x16 result_reg[B2][B2] = {};

        for (u32 k = 0; k < B1; k += 16) {
          f32x16 i_reg[B2], j_reg[B2];

          #pragma GCC unroll 4
          for (u32 l = 0; l < B2; l++) {
            i_reg[l] = *(f32x16*)&_data[x+k + (i+l)*mx];
            j_reg[l] = *(f32x16*)&_data[x+k + (j+l)*mx];
          }

          #pragma GCC unroll 4
          for (u32 jj = 0; jj < B2; jj++) {
            #pragma GCC unroll 4 
            for (u32 ii = 0; ii < B2; ii++) {
              result_reg[jj][ii] += j_reg[jj] * i_reg[ii];
            }
          }
        }

        #pragma GCC unroll 4
        for (u32 jj = 0; jj < B2; jj++) {
          #pragma GCC unroll 4
          for (u32 ii = 0; ii < B2; ii++) {
            _result[i+ii + (j+jj)*my] +=
                _mm512_reduce_add_ps(result_reg[jj][ii]);
          }
        }
      }
    }
  }

  for (u32 j = 0; j < ny; j++) {
    for (u32 i = j; i < ny; i++) {
      result[i + j*ny] = _result[i + j*my] / (nx - 1);
    }
  }

  if (my * std::max(my, mx) > 5100 * 5100) {
    free(_data);
    free(_result);
  }

}
