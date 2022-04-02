#include "kernel_cpu.h"
#include "Complex.h"

namespace MathsEx
{
    namespace kernel_cpu
    {
        unsigned calculate_point(double cr, double ci, unsigned max_iter)
        {
            double zr = 0.0;
            double zi = 0.0;

            double zr2 = zr * zr;
            double zi2 = zi * zi;

            unsigned iter = 0;
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

        void mandelbrot_kernel(unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
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

                iters[i] = calculate_point(re, im, max_iter);
            }
        }

        unsigned calculate_julia_point(double cr, double ci, double kr, double ki, unsigned maxIters)
        {
            Complex k(kr, ki);
            Complex z(cr, ci);

            unsigned iters = 0;
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z.squared() + k;
                ++iters;
            }

            return iters;
        }

        void julia_kernel(unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, double kr, double ki, unsigned max_iter, unsigned* iters)
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

                const auto point_value = calculate_julia_point(re, im, kr, ki, max_iter);
                iters[i] = point_value;
            }
        }

    }
}