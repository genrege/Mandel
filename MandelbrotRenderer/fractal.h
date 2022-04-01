#pragma once

//TODO C++ AMP headers to move out of here
#include <amp.h>
#include <amp_math.h>
#include <math.h>
#include <omp.h>

#include "cached_memory.h"

#include "../MandelbrotRendererCUDA/MandelbrotSetCUDA.h"

//TODO C++ AMP to move out of here
using namespace Concurrency;
using namespace precise_math;

namespace MathsEx
{
struct rgb
{
    rgb() : b(0), g(0), r(0), pad(0xFF) {}
    unsigned char b;
    unsigned char g;
    unsigned char r;
    unsigned char pad;
};


class iteration_palette
{
public:
    iteration_palette() = default;

    void apply(unsigned max_iterations, const cache_memory<unsigned>& data, cache_memory<rgb>& bmp) 
    {
        if (palette_.reserve(max_iterations + 1))
        {
            const float scale = 6.3f / max_iterations;
            #pragma omp         parallel for
            for (int i = 0; i < static_cast<int>(max_iterations); ++i)
            {
                palette_[i].r = static_cast<unsigned char>(sin(scale * i + 3) * 127 + 128);
                palette_[i].g = static_cast<unsigned char>(sin(scale * i + 5) * 127 + 128);
                palette_[i].b = static_cast<unsigned char>(sin(scale * i + 1) * 127 + 128);
            }

            palette_[max_iterations].r = 0;
            palette_[max_iterations].g = 0;
            palette_[max_iterations].b = 0;
        }

        
        #pragma omp         parallel for
        for (int i = 0; i < static_cast<int>(data.size()); ++i)
        {
            bmp[i] = palette_[data[i]];
        }
    }

private:
    cache_memory<rgb> palette_;
};

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

            m_data.reserve(m_wx * m_wy);
        }
    }

    const double& x0() const { return m_x0; }
    const double& x1() const { return m_x1; }
    const double& y0() const { return m_y0; }
    const double& y1() const { return m_y1; }
    unsigned wx() const { return m_wx; }
    unsigned wy() const { return m_wy; }

    cache_memory<unsigned>&  data() { return m_data; };

    void allocate_data() { m_data.reserve(m_wx * m_wy); }
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
        mandelbrot_kernel_cpu(wx(), wy(), x0(), x1(), y0(), y1(), max_iterations, data());
    }

    void calculate_set_amp(const accelerator_view& v, const unsigned max_iterations)
    {
        allocate_data();
        mandelbrot_kernel_amp(v, wx(), wy(), x0(), x1(), y0(), y1(), max_iterations, data());
    }

    void calculate_set_cuda(const unsigned max_iterations)
    {
        m_cuda.render_mbrot(wx(), wy(), x0(), x1(), y0(), y1(), max_iterations, data());
    }

    inline static unsigned calculate_point(const Complex<double>& c, unsigned max_iter) restrict(amp, cpu)
    {
        const double cr = c.Re();
        const double ci = c.Im();

        double zr = 0.0;
        double zi = 0.0;

        double zr2 = zr * zr;
        double zi2 = zi * zi;

        unsigned iter = 0;
        while (iter < max_iter && (zr2 + zi2) < 4.0)
        {
            zi = (zr + zr) * zi + ci;
            zr = zr2 - zi2 + cr;

            zr2 = zr * zr;
            zi2 = zi * zi;

            ++iter;
        }
        return iter;
    }

    static void mandelbrot_kernel_cpu(unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
    {
        const int num_points = display_w * display_h;

        const auto set_width = x1 - x0;
        const auto set_height = y1 - y0;

        const auto set_step_x = set_width / double(display_w);
        const auto set_step_y = set_height / double(display_h);

        #pragma omp parallel for
        for (int i = 0; i < num_points; ++i)
        {
            const auto array_x = i % display_w;
            const auto array_y = display_h - i / display_w;

            const auto re = x0 + array_x * set_step_x;
            const auto im = y0 + array_y * set_step_y;
            const Complex<double> c(re, im);

            const auto point_value = calculate_point(c, max_iter);
            iters[i] = point_value;
        }
    }

    static void mandelbrot_kernel_amp(const accelerator_view& v, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
    {
        const auto num_points = display_w * display_h;

        const auto set_width = x1 - x0;
        const auto set_height = y1 - y0;

        const auto set_step_x = set_width / double(display_w);
        const auto set_step_y = set_height / double(display_h);

        concurrency::extent<1> e(num_points);
        concurrency::array_view<unsigned, 1> mandelbrotResult(e, iters);

        concurrency::parallel_for_each(v, mandelbrotResult.extent,
            [display_w, display_h, x0, y0, set_step_x, set_step_y, max_iter, mandelbrotResult](index<1> idx) restrict(amp, cpu)
            {
                const auto array_x = idx[0] % display_w;
                const auto array_y = display_h - idx[0] / display_w;

                const auto re = x0 + array_x * set_step_x;
                const auto im = y0 + array_y * set_step_y;
                const Complex<double> c(re, im);

                const auto point_value = calculate_point(c, max_iter);
                mandelbrotResult[idx] = point_value;
            });
        mandelbrotResult.synchronize();
        mandelbrotResult.discard_data();
    }

    static void mandelbrot_kernel_cuda(unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
    {
        mbrot_cuda cuda;
        cuda.render_mbrot(display_w, display_h, x0, x1, y0, y1, max_iter, iters);
    }

private:
    mbrot_cuda m_cuda;

};

}   //  MathsEx
