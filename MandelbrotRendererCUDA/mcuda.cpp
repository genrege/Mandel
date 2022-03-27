#include <stdio.h>
#include "mcuda.h"
#include "MandelbrotSet.cuh"

void render_mbrot(double x0, double x1, double y0, double y1, int wx, int wy, int max_iter, int* r)
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus == cudaSuccess)
    {
        int* dev_r;
        cudaMalloc((void**)&dev_r, sizeof(int) * wx * wy);
        kernel_mbrot<<<1,1>>>(x0, x1, y0, y1, wx, wy, max_iter, dev_r);
        cudaMemcpy(r, dev_r, 255 * 255 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(dev_r);
    }
}
