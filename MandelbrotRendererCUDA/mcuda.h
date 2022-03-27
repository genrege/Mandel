#pragma once

#if defined(MANDELBROT_RENDERER_CUDA_EXPORT)
#define MANDELBROT_RENDERER_CUDA_API __declspec(dllexport)
#else
#define MANDELBROT_RENDERER_CUDA_API  __declspec(dllimport)
#endif

MANDELBROT_RENDERER_CUDA_API void render_mbrot(double x0, double x1, double y0, double y1, int wx, int wy, int max_iter, int* r);
