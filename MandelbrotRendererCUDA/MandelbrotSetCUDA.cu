#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "MandelbrotSetCUDA.h"

__device__ unsigned int calculate_point(double re, double im, unsigned int max_iter)
{
    const double cr = re;
    const double ci = im;

    double zr = 0.0;
    double zi = 0.0;

    double zr2 = zr * zr;
    double zi2 = zi * zi;

    int iter = 0;
    while (iter < max_iter && (zr2 + zi2) <= 4.0)
    {
        zi = (zr + zr) * zi + ci;
        zr = zr2 - zi2 + cr;

        zr2 = zr * zr;
        zi2 = zi * zi;

        ++iter;
    }
    return iter;
}

__global__ void kernel_mbrot(double x0, double x1, double y0, double y1, int wx, int wy, double w, double h, int max_iter, unsigned int* r, unsigned* p, unsigned palette_index)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= wx)
        return;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (iy >= wy)
        return;

    const double cr = x0 + w * ix;
    const double ci = y1 - h * iy;

    double zr = 0.0;
    double zi = 0.0;

    double zr2 = zr * zr;
    double zi2 = zi * zi;
    
    int iter = 0;
    while (iter < max_iter && (zr2 + zi2) <= 4.0)
    {
        zi = (zr + zr) * zi + ci;
        zr = zr2 - zi2 + cr;

        zr2 = zr * zr;
        zi2 = zi * zi;

        ++iter;
    }
    const auto idx = ix + wx * iy;
    if (p == nullptr)
        r[idx] = iter;
    else
        r[idx] = iter >= max_iter ? 0 : p[(iter + palette_index) % max_iter];
}

__global__ void kernel_julia(double x0, double x1, double y0, double y1, double kr, double ki, double w, double h, int wx, int wy, int max_iter, unsigned* r, unsigned* p, unsigned palette_index)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= wx)
        return;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (iy >= wy)
        return;

    const double cr = x0 + w * ix;
    const double ci = y1 - h * iy;

    double zr = cr;
    double zi = ci;

    double zr2 = zr * zr;
    double zi2 = zi * zi;

    int iter = 0;
    while (iter < max_iter && (zr2 + zi2) < 4.0)
    {
        zi = (zr + zr) * zi + ki;
        zr = zr2 - zi2 + kr;

        zr2 = zr * zr;
        zi2 = zi * zi;

        ++iter;
    }
    const int idx = ix + wx * iy;
    if (p == nullptr)
        r[idx] = iter;
    else
        r[idx] = iter >= max_iter ? 0 : p[(iter + palette_index) % max_iter];
}

__global__ void kernel_buddha(bool anti_buddha, double x0, double x1, double y0, double y1, int wx, int wy, double w, double h, int max_iter, unsigned int* r)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= wx)
        return;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (iy >= wy)
        return;

    const double cr = x0 + w * ix;
    const double ci = y1 - h * iy;

    const auto escaped = anti_buddha ? false : calculate_point(cr, ci, max_iter) >= max_iter;
    if (escaped)
        return;

    double zr = 0.0;
    double zi = 0.0;

    double zr2 = zr * zr;
    double zi2 = zi * zi;

    const auto set_width = x1 - x0;
    const auto set_height = y1 - y0;

    const auto set_step_x = set_width / double(wx);
    const auto set_step_y = set_height / double(wy);

    const double r_set_step_x = 1.0 / set_step_x;
    const double r_set_step_y = 1.0 / set_step_y;

    int iter = 0;
    while (iter < max_iter && (zr2 + zi2) <= 4.0)
    {
        zi = (zr + zr) * zi + ci;
        zr = zr2 - zi2 + cr;

        zr2 = zr * zr;
        zi2 = zi * zi;

        const auto rx = int((zr - x0) * r_set_step_x + 0.5);
        const auto ry = int(wy - 1 - ((zi - y0) * r_set_step_y) + 0.5);

        if (rx >= 0 && ry >= 0 && rx < wx && ry < wy)
        {
            const auto idx = rx + wx * ry;
            r[idx] += 1;
        }

        ++iter;
    }
}


mbrot_cuda::mbrot_cuda(int gpu_index) : m_dev_r(nullptr), m_csize(0), last_cuda_error_()
{
    int device_count = 0;
    const auto error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess)
    {
        last_cuda_error_ = error;
    }
    else if (gpu_index < device_count)
        cudaSetDevice(gpu_index);
    else
        cudaSetDevice(0);
}

mbrot_cuda::~mbrot_cuda()
{
    cudaFree(m_dev_r);
}

size_t mbrot_cuda::device_count() 
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess)
        return 0;
    return (size_t)deviceCount;
}

std::string mbrot_cuda::device_name(size_t index) 
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess)
        return {};

    cudaSetDevice((int)index);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, (int)index);

    return deviceProp.name;
}

unsigned int* mbrot_cuda::alloc_cuda(int size)
{
    if (size != m_csize)
    {
        cudaFree(m_dev_r);
        cudaMalloc(&m_dev_r, size);
        m_csize = size;
    }
    return m_dev_r;
}

unsigned int* mbrot_cuda::alloc_palette(int size)
{
    if (size != m_psize)
    {
        cudaFree(m_dev_p);
        cudaMalloc(&m_dev_p, size);
        m_psize = size;
    }
    return m_dev_p;
}

void mbrot_cuda::render_mbrot(int wx, int wy, double x0, double x1, double y0, double y1, int max_iter, unsigned int* r, unsigned* palette, unsigned palette_index)
{
    const double w = (x1 - x0) / double(wx);
    const double h = (y1 - y0) / double(wy);

    const int gs = 32;
    int extra = (wx % gs == 0) ? 0 : 1;
    dim3 threads(gs, gs);
    dim3 blocks(wx / gs + extra, wy / gs + extra);

    auto* dev_r = alloc_cuda(sizeof(unsigned int) * wx * wy);
    unsigned* dev_p = nullptr;
    if (palette != nullptr)
    {
        dev_p = alloc_palette(sizeof(unsigned) * (1 + max_iter));
        cudaMemcpy(dev_p, palette, sizeof(unsigned) * (1 + max_iter), cudaMemcpyKind::cudaMemcpyHostToDevice);
    }

    kernel_mbrot << <blocks, threads>> > (x0, x1, y0, y1, wx, wy, w, h, max_iter, dev_r, dev_p, palette_index);
    const auto error = cudaMemcpy(r, dev_r, sizeof(unsigned int) * wx * wy, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    (void)error;
}

void mbrot_cuda::render_julia(int wx, int wy, double x0, double x1, double y0, double y1, double kr, double ki, int max_iter, unsigned int* r, unsigned* palette, unsigned palette_index)
{
    const double w = (x1 - x0) / double(wx);
    const double h = (y1 - y0) / double(wy);

    const int gs = 32;
    int extra = (wx % gs == 0) ? 0 : 1;
    dim3 threads(gs, gs);
    dim3 blocks(wx / gs + extra, wy / gs + extra);

    auto* dev_r = alloc_cuda(sizeof(unsigned int) * wx * wy);
    unsigned* dev_p = nullptr;
    if (palette != nullptr)
    {
        dev_p = alloc_palette(sizeof(unsigned) * (1 + max_iter));
        cudaMemcpy(dev_p, palette, sizeof(unsigned) * (1 + max_iter), cudaMemcpyKind::cudaMemcpyHostToDevice);
    }

    kernel_julia << <blocks, threads>> > (x0, x1, y0, y1, kr, ki, w, h, wx, wy, max_iter, dev_r, dev_p, palette_index);
    cudaMemcpy(r, dev_r, sizeof(unsigned int) * wx * wy, cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

void mbrot_cuda::render_buddha(bool anti_buddha, int wx, int wy, double x0, double x1, double y0, double y1, int max_iter, unsigned int* r)
{
    const double w = (x1 - x0) / double(wx);
    const double h = (y1 - y0) / double(wy);

    const int gs = 32;
    int extra = (wx % gs == 0) ? 0 : 1;
    dim3 threads(gs, gs);
    dim3 blocks(wx / gs + extra, wy / gs + extra);

    auto* dev_r = alloc_cuda(sizeof(unsigned int) * wx * wy);
    cudaMemset(r, 0, sizeof(unsigned int) * wx * wy);

    kernel_buddha << <blocks, threads >> > (anti_buddha, x0, x1, y0, y1, wx, wy, w, h, max_iter, dev_r);
    cudaMemcpy(r, dev_r, sizeof(unsigned int) * wx * wy, cudaMemcpyKind::cudaMemcpyDeviceToHost);
}