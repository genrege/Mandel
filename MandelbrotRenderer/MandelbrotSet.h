#pragma once

#include <amp.h>
#include <amp_math.h>
#include <math.h>
#include <omp.h>

#include "Complex.h"
#include "cached_memory.h"
#include "fractal.h"
#include "palette.h"

#include "../MandelbrotRendererCUDA/MandelbrotSetCUDA.h"

#pragma warning (disable: 4996)

using namespace Concurrency;
using namespace precise_math;


namespace fractals
{
    class MandelbrotSet
    {
    public:

        const unsigned* bmp() const { return (unsigned*)m_bmp.access(); }
        const unsigned data_size() const { return m_wx * m_wy; }

        MandelbrotSet() : m_x0(0), m_x1(0), m_y0(0), m_y1(0), m_wx(0), m_wy(0)
        {
        }

        ~MandelbrotSet()
        {
        }

        void SetScale(double x0, double x1, double y0, double y1, unsigned wx, unsigned wy) restrict(cpu)
        {
            m_x0 = x0;
            m_x1 = x1;

            m_y0 = y0;
            m_y1 = y1;

            //Allocate AMP input/output structures
            if (m_wx != wx || m_wy != wy)
            {

                m_wx = wx;
                m_wy = wy;

                m_arr.allocate(m_wx * m_wy);
                m_bmp.allocate(m_wx * m_wy);
                m_density.allocate(m_wx * m_wy);
            }
        }


        void CalculateBuddha(const accelerator_view& v, bool anti_buddha, const unsigned maxIters) restrict(cpu)
        {
            gpuCalculationDensity(v, anti_buddha, m_wx, m_wy, m_x0, m_x1, m_y0, m_y1, maxIters, m_density);

            setPaletteBuddha(1 + maxIters, m_density, m_wx * m_wy);
            gpuPaletteKernel(v, m_wx * m_wy, m_density, m_bmp, maxIters, m_palette_buddha);
        }

        void setPaletteBuddha(size_t size, unsigned* density, unsigned size_density)
        {
            if (m_palette_buddha.allocate(size))
            {
                unsigned max_density = 0;
                for (size_t i = 0; i < size_density; ++i)
                    if (density[i] > max_density) max_density = density[i];

                const double palette_scale = 15.0 / double(max_density);

                for (size_t i = 0; i < size; ++i)
                {
                    ZeroMemory(&m_palette_buddha[i], sizeof(rgb));

                    const double s1 = min(1.0, double(i) * palette_scale);
                    const double s2 = min(1.0, double(i) * palette_scale);
                    const double s3 = min(1.0, double(i) * palette_scale);

                    const double f = min(1.0, tanh(s2));
                    const double g = min(1.0, (1 - pow(s1 - 1, 2)));
                    const double h = min(1.0, (1 - pow(s3 - 1, 2)));

                    m_palette_buddha[i].r = char(255 * f);
                    if (i > 16)
                        m_palette_buddha[i].g = char(255 * g);
                    if (i > 24)
                        m_palette_buddha[i].b = char(255 * h);
                }
            }
        }

        static void cpuSpecialKernel(int func, const complex& k, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* mandelbrotResult)
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
                const complex c(re, im);

                switch(func){
                case  0: mandelbrotResult[i] = CalculateSpecial_0(c, k, max_iter); break;
                case  1: mandelbrotResult[i] = CalculateSpecial_1(c, k, max_iter); break;
                case  2: mandelbrotResult[i] = CalculateSpecial_2(c, k, max_iter); break;
                case  3: mandelbrotResult[i] = CalculateSpecial_3(c, k, max_iter); break;
                case  4: mandelbrotResult[i] = CalculateSpecial_4(c, k, max_iter); break;
                case  5: mandelbrotResult[i] = CalculateSpecial_5(c, k, max_iter); break;
                case  6: mandelbrotResult[i] = CalculateSpecial_6(c, k, max_iter); break;
                case  7: mandelbrotResult[i] = CalculateSpecial_7(c, k, max_iter); break;
                case  8: mandelbrotResult[i] = CalculateSpecial_8(c, k, max_iter); break;
                case  9: mandelbrotResult[i] = CalculateSpecial_9(c, k, max_iter); break;
                case 10: mandelbrotResult[i] = CalculateSpecial_10(c, k, max_iter); break;
                case 11: mandelbrotResult[i] = CalculateSpecial_11(c, k, max_iter); break;
                case 12: mandelbrotResult[i] = CalculateSpecial_12(c, k, max_iter); break;
                case 13: mandelbrotResult[i] = CalculateSpecial_13(c, k, max_iter); break;
                case 14: mandelbrotResult[i] = CalculateSpecial_14(c, k, max_iter); break;
                }
            }
        }


        static void gpuSpecialKernel(const accelerator_view& v, int func, const complex& k, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
        {
            const auto num_points = display_w * display_h;

            const auto set_width  = x1 - x0;
            const auto set_height = y1 - y0;

            const auto set_step_x = set_width / double(display_w);
            const auto set_step_y = set_height / double(display_h);

            concurrency::extent<1> e(num_points);
            concurrency::array_view<unsigned, 1> mandelbrotResult(e, iters);

            concurrency::parallel_for_each(v, mandelbrotResult.extent,
                [func, k, display_w, display_h, x0, y0, set_step_x, set_step_y, max_iter, mandelbrotResult](index<1> idx) restrict(amp, cpu)
                {
                    const auto array_x = idx[0] % display_w;
                    const auto array_y = display_h - idx[0] / display_w;

                    const auto re = x0 + array_x * set_step_x;
                    const auto im = y0 + array_y * set_step_y;
                    const complex c(re, im);

                    switch(func){
                    case  0: mandelbrotResult[idx] = CalculateSpecial_0(c, k, max_iter); break;
                    case  1: mandelbrotResult[idx] = CalculateSpecial_1(c, k, max_iter); break;
                    case  2: mandelbrotResult[idx] = CalculateSpecial_2(c, k, max_iter); break;
                    case  3: mandelbrotResult[idx] = CalculateSpecial_3(c, k, max_iter); break;
                    case  4: mandelbrotResult[idx] = CalculateSpecial_4(c, k, max_iter); break;
                    case  5: mandelbrotResult[idx] = CalculateSpecial_5(c, k, max_iter); break;
                    case  6: mandelbrotResult[idx] = CalculateSpecial_6(c, k, max_iter); break;
                    case  7: mandelbrotResult[idx] = CalculateSpecial_7(c, k, max_iter); break;
                    case  8: mandelbrotResult[idx] = CalculateSpecial_8(c, k, max_iter); break;
                    case  9: mandelbrotResult[idx] = CalculateSpecial_9(c, k, max_iter); break;
                    case 10: mandelbrotResult[idx] = CalculateSpecial_10(c, k, max_iter); break;
                    case 11: mandelbrotResult[idx] = CalculateSpecial_11(c, k, max_iter); break;
                    case 12: mandelbrotResult[idx] = CalculateSpecial_12(c, k, max_iter); break;
                    case 13: mandelbrotResult[idx] = CalculateSpecial_13(c, k, max_iter); break;
                    case 14: mandelbrotResult[idx] = CalculateSpecial_14(c, k, max_iter); break;
                    }
                });
            mandelbrotResult.synchronize();
            mandelbrotResult.discard_data();
        }

        //This function's a bit crap because gpu threads write concurrently to the map without any synchronisation, 
        //so rendering the same image twice will probably give slightly different maps each time.
        //Also, unless you fix the bounds of the set (ie real/imaginary bounds) and the calculation step between 
        //points, you get "mathematical shadows" as you move around and zoom in/out as the intermediate calculation 
        //points change  whenever the view changes.
        //And this computation is relatively slow, probably because of the extra floating point calculations here.
        //Interesting to see the Mandelbrot "buddha" though!
        static void gpuCalculationDensity(const accelerator_view& v, bool anti_buddha, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* dmap)
        {
            const auto num_points = display_w * display_h;

            const auto set_width = x1 - x0;
            const auto set_height = y1 - y0;

            const auto set_step_x = set_width / double(display_w);
            const auto set_step_y = set_height / double(display_h);

            memset(dmap, 0, sizeof(unsigned) * display_w * display_h);

            concurrency::extent<1> e(num_points);
            concurrency::array_view<unsigned, 1> edmap(e, dmap);

            const double r_set_step_x = 1.0 / set_step_x;
            const double r_set_step_y = 1.0 / set_step_y;

            concurrency::parallel_for_each(v, e,
                [anti_buddha, display_w, display_h, x0, y0, set_step_x, set_step_y, r_set_step_x, r_set_step_y, max_iter, edmap](index<1> idx) restrict(amp)
                {
                    const auto array_x = idx[0] % display_w;
                    const auto array_y = display_h - idx[0] / display_w;

                    const auto re = x0 + array_x * set_step_x;
                    const auto im = y0 + array_y * set_step_y;
                    const complex c(re, im);

                    const bool escaped = anti_buddha ? false : kernel_amp::calculate_point(re, im,  max_iter) >= max_iter;
                    if (!escaped)
                    {
                        unsigned iters = 0;

                        complex z(0.0);
                        while (iters < max_iter && SumSquares(z) <= 4.0)
                        {
                            z = z * z + c;

                            const int rx = int((z.Re() - x0) * r_set_step_x + 0.5);
                            const int ry = int(display_h - 1 - ((z.Im() - y0) * r_set_step_y) + 0.5);

                            //This if is needed in case some point flies off outside the display boundaries
                            if (rx >= 0 && ry >= 0 && rx < int(display_w) && ry < int(display_h))
                            {
                                const auto index = rx + display_w * ry;
                                atomic_fetch_inc(&edmap[index]);
                            }

                            ++iters;
                        }
                    }
                });
            edmap.synchronize();
            edmap.discard_data();
        }

        //Use this function to map iters->palette with the result going into bmp
        //No bounds checking is done, so size_palette must be at least max(iters[first..last]) in size
        //iters and bmp must have the same dimension (size)
        static void gpuPaletteKernel(const accelerator_view& v, unsigned size, unsigned* iters, unsigned* bmp, unsigned size_palette, rgb* palette)
        {
            concurrency::extent<1> e(size);
            concurrency::array_view<unsigned, 1> av_iters(e, iters);
            concurrency::array_view<unsigned, 1> av_bmp(e, bmp);

            concurrency::extent<1> ep(size_palette);
            concurrency::array_view<unsigned, 1> av_palette(ep, (unsigned*)palette);

            concurrency::parallel_for_each(v, e, [av_iters, av_bmp, av_palette](index<1> idx) restrict(amp, cpu)
                {
                    av_bmp[idx] = av_palette[av_iters[idx]];
                });
        }

        inline static unsigned CalculateSpecial_0(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                const auto& z2 = z * z;
                const auto& z3 = z * z2;

                z = z - (z2 - k) / (z2 + k) + (z3 - k) / (z3 + k);

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_1(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                const auto& z2 = z * z;
                z = (z2 - k) * (z + k) / z2;

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_2(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(k);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + c;

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_3(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z * z * z * z + k;

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_4(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z * z * z * z * z + k;

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_5(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z * z * z * z * z * z + k;

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_6(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = (z + z * z * z * z * z * z + z * z * z * z * z) / (z * z * z) + k;

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_7(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z * z + Sin(z) + k;

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_8(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Cos(z * k);

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_9(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Cos(Sin(z + k));

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_10(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z * (z -k) * (z + k);

                ++iters;
            }

            return iters;
        }
        inline static unsigned CalculateSpecial_11(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Tan(z + k);

                ++iters;
            }

            return iters;
        }
        inline static unsigned CalculateSpecial_12(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z * z + Sqrt(k + z);

                ++iters;
            }

            return iters;
        }
        inline static unsigned CalculateSpecial_13(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(k);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z * z + c;

                ++iters;
            }

            return iters;
        }
        inline static unsigned CalculateSpecial_14(const complex& c, const complex& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            complex z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = Sin(z) * Cos(z) + k;

                ++iters;
            }

            return iters;
        }

        inline unsigned ValueAt(size_t ix, size_t iy) 
        {
            const auto index = ix + m_wy * iy;
            if (index < m_wx * m_wy)
            {
                m_bmp[index] = 0x00FFFFFF;
                return m_arr[index];
            }
            return 0;
        }

    private:
        //Virtual dimensions
        double m_x0;
        double m_x1;
        double m_y0;
        double m_y1;

        //Display dimensions
        unsigned m_wx;
        unsigned m_wy;

        //Runtime storage
        cache_memory<rgb>       m_palette_buddha;
        cache_memory<unsigned>  m_arr;
        cache_memory<unsigned>  m_bmp;
        cache_memory<unsigned>  m_density;
    };
}

