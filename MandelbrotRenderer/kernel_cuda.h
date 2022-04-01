#pragma once

#include "../MandelbrotRendererCUDA/MandelbrotSetCUDA.h"

namespace MathsEx
{
    namespace kernel_cuda
    {
        static void mandelbrot_kernel(unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
        {
            mbrot_cuda().render_mbrot(display_w, display_h, x0, x1, y0, y1, max_iter, iters);
        }
    }
}