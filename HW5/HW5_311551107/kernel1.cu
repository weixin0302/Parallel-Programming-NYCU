#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__global__ void mandelKernel(int* cuda_data, float lowerX, float lowerY, float stepX, float stepY, int width, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;

    int index = thisX + thisY * width;

    int value = mandel(x, y, maxIterations);

    cuda_data[index] = value;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *host_data = (int*) malloc(resX * resY * sizeof(int));
    int *cuda_data;
    cudaMalloc(&cuda_data, resX * resY * sizeof(int));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlock(resX / threadsPerBlock.x, resY / threadsPerBlock.y);

    mandelKernel<<<numBlock, threadsPerBlock>>>(cuda_data, lowerX, lowerY, stepX, stepY, resX, maxIterations);

    cudaMemcpy(host_data, cuda_data, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);
    memcpy(img, host_data, resX * resY * sizeof(int));
    cudaFree(cuda_data);
    free(host_data);
}
