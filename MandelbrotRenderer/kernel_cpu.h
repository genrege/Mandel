#pragma once

#include "Complex.h"

namespace MathsEx
{
    namespace kernel_cpu
    {
        inline static unsigned calculate_point(const Complex<double>& c, unsigned max_iter)
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

        static void mandelbrot_kernel(unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
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

                iters[i] = calculate_point(c, max_iter);
            }
        }

    }
}