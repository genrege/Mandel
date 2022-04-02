#pragma once

namespace MathsEx
{
    namespace kernel_cpu
    {
        unsigned calculate_point(double cr, double ci, unsigned max_iter);
        unsigned calculate_julia_point(double cr, double ci, double kr, double ki, unsigned maxIters);
        void mandelbrot_kernel(unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters);
        void julia_kernel(unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, double kr, double ki, unsigned max_iter, unsigned* iters);
   }
}