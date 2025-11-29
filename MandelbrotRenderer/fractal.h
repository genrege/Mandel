#pragma once

#include <omp.h>

#include "cached_memory.h"
#include "Complex.h"
#include "kernel_cuda.h"
#include "kernel_cpu.h"

#include "../MandelbrotRendererCUDA/MandelbrotSetCUDA.h"

namespace fractals
{

class fractal 
{
public:
    fractal() : m_x0(0), m_x1(0), m_y0(0), m_y1(0), m_wx(0), m_wy(0)
    {
    }

    void set_scale(double x0, double x1, double y0, double y1, unsigned wx, unsigned wy)
    {
        m_x0 = x0;
        m_x1 = x1;
        m_y0 = y0;
        m_y1 = y1;
        if (m_wx != wx || m_wy != wy)
        {
            m_wx = wx;
            m_wy = wy;
            m_data.allocate(m_wx * m_wy);
        }
    }

    const double& x0() const { return m_x0; }
    const double& x1() const { return m_x1; }
    const double& y0() const { return m_y0; }
    const double& y1() const { return m_y1; }
    unsigned wx() const { return m_wx; }
    unsigned wy() const { return m_wy; }

    cache_memory<unsigned>&  data() { return m_data; };

    void allocate_data() { m_data.allocate(m_wx * m_wy); }
private:
    //Virtual dimensions
    double m_x0;
    double m_x1;
    double m_y0;
    double m_y1;

    //Display dimensions
    unsigned m_wx;
    unsigned m_wy;

    //Calculation result across virtual and display dimensions
    cache_memory<unsigned>  m_data;
};


class mandelbrot_set : public fractal
{
public:
    mandelbrot_set() = default;

    void calculate_set_cpu(const unsigned max_iterations) 
    {
        allocate_data();
        kernel_cpu::mandelbrot_kernel(wx(), wy(), x0(), x1(), y0(), y1(), max_iterations, data());
    }

    void calculate_set_cuda(int gpu_index, const unsigned max_iterations, unsigned* palette = nullptr, unsigned palette_index = 0)
    {
        kernel_cuda::mandelbrot_kernel(gpu_index, wx(), wy(), x0(), x1(), y0(), y1(), max_iterations, data(), palette, palette_index);
    }
};

class julia_set : public fractal
{
public:
    julia_set() = default;

    void calculate_set_cpu(const complex& k, const unsigned max_iterations)
    {
        allocate_data();
        kernel_cpu::julia_kernel(wx(), wy(), x0(), x1(), y0(), y1(), k.Re(), k.Im(), max_iterations, data());
    }

    void calculate_set_cuda(int gpu_index, const complex& k, const unsigned max_iterations, unsigned* palette = nullptr, unsigned palette_offset = 0)
    {
        kernel_cuda::julia_kernel(gpu_index, wx(), wy(), x0(), x1(), y0(), y1(), k.Re(), k.Im(), max_iterations, data(), palette, palette_offset);
    }
};

class buddha_set : public fractal
{
public:
    buddha_set() = default;

    void calculate_set_cpu()
    {
        allocate_data();
    }

    void calculate_set_cuda(int gpu_index, bool anti_buddha, const unsigned max_iterations)
    {
        kernel_cuda::buddha_kernel(gpu_index, anti_buddha, wx(), wy(), x0(), x1(), y0(), y1(), max_iterations, data());
    }
};
}   //  MathsEx
