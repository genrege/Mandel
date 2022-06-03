#include "kernel_amp.h"
#include "Complex.h"

namespace fractals
{
    namespace kernel_amp
    {

        void mandelbrot_kernel(const accelerator_view& v, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters, unsigned* , unsigned )
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

                    const auto point_value = calculate_point(re, im, max_iter);
                    mandelbrotResult[idx] = point_value;
                });
            mandelbrotResult.synchronize();
            mandelbrotResult.discard_data();
        }

        unsigned calculate_julia_point(const complex& c, const complex& k, unsigned maxIter) restrict(amp)
        {
            unsigned iters = 0;

            complex z = c;
            while (iters < maxIter && SumSquares(z) <= 4.0)
            {
                z = z.squared() + k;
                ++iters;
            }

            return iters;
        }

        void julia_kernel(const accelerator_view& v, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, double kr, double ki, unsigned max_iter, unsigned* iters, unsigned*, unsigned)
        {
            const auto num_points = display_w * display_h;

            const complex k(kr, ki);

            const auto set_width = x1 - x0;
            const auto set_height = y1 - y0;

            const auto set_step_x = set_width / double(display_w);
            const auto set_step_y = set_height / double(display_h);

            concurrency::extent<1> e(num_points);
            concurrency::array_view<unsigned, 1> mandelbrotResult(e, iters);

            concurrency::parallel_for_each(v, mandelbrotResult.extent,
                [k, display_w, display_h, x0, y0, set_step_x, set_step_y, max_iter, mandelbrotResult](index<1> idx) restrict(amp)
                {
                    const auto array_x = idx[0] % display_w;
                    const auto array_y = display_h - idx[0] / display_w;

                    const auto re = x0 + array_x * set_step_x;
                    const auto im = y0 + array_y * set_step_y;
                    const complex c(re, im);

                    const auto point_value = calculate_julia_point(c, k, max_iter);
                    mandelbrotResult[idx] = point_value;
                });

            mandelbrotResult.synchronize();
            mandelbrotResult.discard_data();
        }

    }
}