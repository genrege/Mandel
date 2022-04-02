#pragma once

namespace MathsEx
{
    namespace kernel_cpu
    {
        extern "C" unsigned calculate_point(double cr, double ci, unsigned max_iter);
        extern "C" unsigned calculate_julia_point(double cr, double ci, double kr, double ki, unsigned maxIters);
        extern "C" void mandelbrot_kernel(unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters);
        extern "C" void julia_kernel(unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, double kr, double ki, unsigned max_iter, unsigned* iters);
   }
}