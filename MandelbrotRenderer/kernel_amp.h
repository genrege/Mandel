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
        unsigned calculate_point(double cr, double ci, unsigned max_iter) restrict(amp);
        void mandelbrot_kernel(const accelerator_view& v, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters);
        unsigned calculate_julia_point(const Complex& c, const Complex& k, unsigned maxIters) restrict(amp);
        void julia_kernel(const accelerator_view& v, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, double kr, double ki, unsigned max_iter, unsigned* iters);
    }
}