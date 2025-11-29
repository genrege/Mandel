/*
    DEPRECATED
*/
/*
#pragma once

#include <amp.h>
#include <amp_math.h>

#include "Complex.h"

using namespace Concurrency;
using namespace precise_math;

namespace fractals
{
    namespace kernel_amp
    {
        inline unsigned calculate_point(double cr, double ci, unsigned max_iter) restrict(amp)
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
        void mandelbrot_kernel(const accelerator_view& v, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters, unsigned* palette = nullptr, unsigned palette_offset = 0);
        unsigned calculate_julia_point(const complex& c, const complex& k, unsigned maxIters) restrict(amp);
        void julia_kernel(const accelerator_view& v, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, double kr, double ki, unsigned max_iter, unsigned* iters, unsigned* palette = nullptr, unsigned palette_offset = 0);
    }
}
*/