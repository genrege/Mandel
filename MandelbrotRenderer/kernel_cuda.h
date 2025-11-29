#pragma once

#include "../MandelbrotRendererCUDA/MandelbrotSetCUDA.h"

namespace fractals
{
    namespace kernel_cuda
    {
        static void mandelbrot_kernel(int gpu_index, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters, unsigned* palette = nullptr, unsigned palette_index = 0)
        {
            mbrot_cuda m(gpu_index);
            m.render_mbrot(display_w, display_h, x0, x1, y0, y1, max_iter, iters, palette, palette_index);
        }

        static void julia_kernel(int gpu_index, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, double kr, double ki, unsigned max_iter, unsigned* iters, unsigned* palette = nullptr, unsigned palette_index = 0)
        {
            mbrot_cuda m(gpu_index);
            m.render_julia(display_w, display_h, x0, x1, y0, y1, kr, ki, max_iter, iters, palette, palette_index);
        }

        static void buddha_kernel(int gpu_index, bool anti_buddha, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
        {
            mbrot_cuda m(gpu_index);
            m.render_buddha(anti_buddha, display_w, display_h, x0, x1, y0, y1, max_iter, iters);
        }
    }
}