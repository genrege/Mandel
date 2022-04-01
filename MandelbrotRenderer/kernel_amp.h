#pragma once

#include <amp.h>
#include <amp_math.h>

#include "Complex.h"

using namespace Concurrency;
using namespace precise_math;

namespace MathsEx
{
    namespace kernel_amp
    {
        unsigned calculate_point(const Complex<double>& c, unsigned max_iter) restrict(amp)
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

        void mandelbrot_kernel(const accelerator_view& v, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
        {
            const auto num_points = display_w * display_h;

            const auto set_width = x1 - x0;
            const auto set_height = y1 - y0;

            const auto set_step_x = set_width / double(display_w);
            const auto set_step_y = set_height / double(display_h);

            concurrency::extent<1> e(num_points);
            concurrency::array_view<unsigned, 1> mandelbrotResult(e, iters);

            concurrency::parallel_for_each(v, mandelbrotResult.extent,
                [display_w, display_h, x0, y0, set_step_x, set_step_y, max_iter, mandelbrotResult](index<1> idx) restrict(amp)
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
    }
}