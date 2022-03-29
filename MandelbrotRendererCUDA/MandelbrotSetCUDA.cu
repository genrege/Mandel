#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "MandelbrotSetCUDA.h"
#include <stdio.h>
#include <Windows.h>

__global__ void kernel_mbrot(double x0, double x1, double y0, double y1, int wx, int wy, double w, double h, int max_iter, unsigned int* r)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix > wx)
        return;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (iy > wy)
        return;

    const double c_re = x0 + w * ix;
    const double c_im = y1 - h * iy;

    int iter = 0;

    double z_re = 0.0;
    double z_im = 0.0;

    double zr2 = z_re * z_re;
    double zi2 = z_im * z_im;
    
    while (iter < max_iter && (zr2 + zi2) < 4.0)
    {
        z_im = (z_re + z_re) * z_im + c_im;
        z_re = zr2 - zi2 + c_re;

        zr2 = z_re * z_re;
        zi2 = z_im * z_im;

        ++iter;
    }
    const auto idx = ix + wx * iy;
    r[idx] = iter;
}

__global__ void kernel_julia(double x0, double x1, double y0, double y1, double kr, double ki, double w, double h, int wx, int wy, int max_iter, unsigned int* r)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix > wx)
        return;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (iy > wy)
        return;

    const double c_re = x0 + w * ix;
    const double c_im = y1 - h * iy;

    double z_re = c_re;
    double z_im = c_im;

    double zr2 = z_re * z_re;
    double zi2 = z_im * z_im;

    int iter = 0;
    while (iter < max_iter && (zr2 + zi2) < 4.0)
    {
        z_im = 2.0 * z_re * z_im + ki;
        z_re = zr2 - zi2 + kr;
        ++iter;

        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
    }
    const int idx = ix + wx * iy;
    r[idx] = iter;
}


mbrot_cuda::mbrot_cuda() : m_dev_r(nullptr), m_csize(-1)
{
    cudaSetDevice(0);
}

mbrot_cuda::~mbrot_cuda()
{
    cudaFree(m_dev_r);
}

unsigned int* mbrot_cuda::alloc_cuda(int size)
{
    if (size != m_csize)
    {
        cudaFree(m_dev_r);
        cudaMalloc((void**)&m_dev_r, size);
        m_csize = size;
    }
    return m_dev_r;
}

void mbrot_cuda::render_mbrot(double x0, double x1, double y0, double y1, int wx, int wy, int max_iter, unsigned int* r)
{
    const double w = (x1 - x0) / double(wx);
    const double h = (y1 - y0) / double(wy);

    const int gs = 96;
    dim3 blocks(gs, gs);
    dim3 threads(wx / gs + 1, wy / gs + 1);

    auto * dev_r = alloc_cuda(sizeof(unsigned int) * wx * wy);
    kernel_mbrot << <blocks, threads>> > (x0, x1, y0, y1, wx, wy, w, h, max_iter, dev_r);
    cudaMemcpy(r, dev_r, sizeof(unsigned int) * wx * wy, cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

void mbrot_cuda::render_julia(double x0, double x1, double y0, double y1, double kr, double ki, int wx, int wy, int max_iter, unsigned int* r)
{
    const double w = (x1 - x0) / double(wx);
    const double h = (y1 - y0) / double(wy);

    const int gs = 96;
    dim3 grid(gs, gs);
    dim3 block(wx / gs + 1, wy / gs + 1);

    auto* dev_r = alloc_cuda(sizeof(unsigned int) * wx * wy);
    kernel_julia << <grid, block >> > (x0, x1, y0, y1, kr, ki, w, h, wx, wy, max_iter, dev_r);
    cudaMemcpy(r, dev_r, sizeof(unsigned int) * wx * wy, cudaMemcpyKind::cudaMemcpyDeviceToHost);
}
