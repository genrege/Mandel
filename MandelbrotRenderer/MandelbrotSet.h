#pragma once

#include <amp.h>
#include <amp_math.h>
#include <math.h>

#include "Complex.h"

#pragma warning (disable: 4996)

using namespace Concurrency;
using namespace precise_math;


namespace MathsEx
{
    template <class RealType> class MandelbrotSet
    {
    public:
        struct rgb
        {
            rgb() : b(0), g(0), r(0), pad(0xFF) {}
            unsigned char b;
            unsigned char g;
            unsigned char r;
            unsigned char pad;
        };

        typedef RealType FloatingPointType;

        const unsigned* data() const { return m_arr; }
        const unsigned* bmp() const { return m_bmp; }
        const unsigned data_size() const { return m_wx * m_wy; }

        MandelbrotSet() : m_x0(0), m_x1(0), m_xstep(0), m_y0(0), m_y1(0), m_ystep(0), m_wx(0), m_wy(0), m_arr(nullptr), m_bmp(nullptr), m_density(nullptr)
        {
        }

        ~MandelbrotSet()
        {
            delete[] m_arr;
            delete[] m_bmp;
            delete[] m_density;
        }

        void SetScale(RealType x0, RealType x1, RealType y0, RealType y1, unsigned wx, unsigned wy) restrict(cpu)
        {
            m_x0 = x0;
            m_x1 = x1;
            m_xstep = (x1 - x0) / (wx - 1);

            m_y0 = y0;
            m_y1 = y1;
            m_ystep = (y1 - y0) / (wy - 1);

            //Allocate AMP input/output structures
            if (m_wx != wx || m_wy != wy || m_arr == nullptr || m_bmp == nullptr || m_density == nullptr)
            {
                delete[] m_arr;
                delete[] m_bmp;
                delete[] m_density;

                m_wx = wx;
                m_wy = wy;

                m_arr = new unsigned[m_wx * m_wy];
                m_bmp = new unsigned[m_wx * m_wy];
                m_density = new unsigned[m_wx * m_wy];
            }
        }

        void CalculateSetCPU(const unsigned maxIters) restrict(cpu)
        {
            cpuMandelbrotKernel(m_wx, m_wy, m_x0, m_x1, m_y0, m_y1, maxIters, m_arr);

            rgb* palette = new rgb[maxIters];
            setPalette(maxIters, palette);
            cpuPaletteKernel(m_wx * m_wy, m_arr, m_bmp, maxIters, palette);
            delete[] palette;
        }

        //Uses AMP to calculate the entire set over the space (m_x0, m_x1)-(m_x1, m_y1) in steps (m_xstep, m_ystep)
        void CalculateSet(const unsigned maxIters) restrict(cpu)
        {
            gpuMandelbrotKernel(m_wx, m_wy, m_x0, m_x1, m_y0, m_y1, maxIters, m_arr);

            rgb* palette = new rgb[maxIters];
            setPalette(maxIters, palette);
            gpuPaletteKernel(m_wx * m_wy, m_arr, m_bmp, maxIters, palette);
            delete[] palette;
        }

        void CalculateBuddha(bool anti_buddha, const unsigned maxIters) restrict(cpu)
        {
            gpuCalculationDensity(anti_buddha, m_wx, m_wy, m_x0, m_x1, m_y0, m_y1, maxIters, m_density);

            rgb* palette = new rgb[maxIters];
            setPaletteBuddha(maxIters, palette, m_density, m_wx * m_wy);
            gpuPaletteKernel(m_wx * m_wy, m_density, m_bmp, maxIters, palette);
            delete[] palette;
        }

        void CalculateJulia(const Complex<RealType>& k, const unsigned maxIters) restrict(cpu)
        {
            gpuJuliaKernel(k, m_wx, m_wy, m_x0, m_x1, m_y0, m_y1, maxIters, m_arr);

            rgb* palette = new rgb[maxIters];
            setPaletteJulia(maxIters, palette);
            gpuPaletteKernel(m_wx * m_wy, m_arr, m_bmp, maxIters, palette);
            delete[] palette;
        }

        void CalculateJuliaCPU(const Complex<RealType>& k, const unsigned maxIters) restrict(cpu)
        {
            cpuJuliaKernel(k, m_wx, m_wy, m_x0, m_x1, m_y0, m_y1, maxIters, m_arr);

            rgb* palette = new rgb[maxIters];
            setPaletteJulia(maxIters, palette);
            gpuPaletteKernel(m_wx * m_wy, m_arr, m_bmp, maxIters, palette);
            delete[] palette;
        }

        static void setPalette(size_t size, rgb* palette)
        {
            for (size_t i = 0; i < size; ++i)
            {
                const double s1 = double(i * 3) / size;
                const double s2 = double(i) / size;
                const double s3 = double(i * 5 / 2) / size;

                const double f = min(1.0, (1 - pow(s1 - 1, 8)));
                const double g = min(1.0, (1 - pow(s2 - 1, 4)));
                const double h = min(1.0, (1 - pow(s3 - 1, 2)));

                palette[i].r = char(255 * f);
                palette[i].g = char(255 * g);
                palette[i].b = char(255 * h);
            }
        }

        static void setPaletteJulia(size_t size, rgb* palette)
        {
            for (size_t i = 0; i < size; ++i)
            {
                const double s1 = double(i * 1) / size;
                const double s2 = double(i / 2) / size;
                const double s3 = double(i * 5 / 2) / size;

                const double f = min(1.0, (1 - pow(s1 - 1, 2)));
                const double g = min(1.0, (1 - pow(s2 - 1, 4)));
                const double h = min(1.0, (1 - pow(s3 - 1, 6)));

                palette[i].r = char(255 * f);
                palette[i].g = char(255 * g);
                palette[i].b = char(255 * h);
            }
        }

        static void setPaletteBuddha(size_t palette_size, rgb* palette, unsigned* density, unsigned size_density)
        {
            unsigned max_density = 0;
            for (size_t i = 0; i < size_density; ++i)
                if (density[i] > max_density) max_density = density[i];

            const double palette_scale = 15.0 / double(max_density);

            for (size_t i = 0; i < palette_size; ++i)
            {
                ZeroMemory(&palette[i], sizeof(rgb));
                
                const double s1 = min(1.0, double(i) * palette_scale);
                const double s2 = min(1.0, double(i) * palette_scale);
                const double s3 = min(1.0, double(i) * palette_scale);

                const double f = min(1.0, tanh(s1));
                const double g = min(1.0, (1 - pow(s1 - 1, 2)));
                const double h = min(1.0, (1 - pow(s3 - 1, 2)));

                palette[i].r = char(255 * f);
                if (i > 16)
                    palette[i].g = char(255 * g);
                if (i > 24)
                    palette[i].b = char(255 * h);
            }
        }

        static void cpuMandelbrotKernel(unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
        {
            const int num_points = display_w * display_h;

            const int wx = static_cast<int>(display_w);
            const int wy = static_cast<int>(display_h);

            const auto set_width  = x1 - x0;
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
                const Complex<RealType> c(re, im);

                const auto point_value = CalculatePoint(c, max_iter);
                iters[i] = point_value;
            }
        }

        static void gpuMandelbrotKernel(unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
        {
            const auto num_points = display_w * display_h;

            const auto set_width  = x1 - x0;
            const auto set_height = y1 - y0;

            const auto set_step_x = set_width / double(display_w);
            const auto set_step_y = set_height / double(display_h);

            concurrency::extent<1> e(num_points);
            concurrency::array_view<unsigned, 1> mandelbrotResult(e, iters);

            concurrency::parallel_for_each(mandelbrotResult.extent,
                [display_w, display_h, x0, y0, set_step_x, set_step_y, max_iter, mandelbrotResult](index<1> idx) restrict(amp, cpu)
                {
                    const auto array_x = idx[0] % display_w;
                    const auto array_y = display_h - idx[0] / display_w;

                    const auto re = x0 + array_x * set_step_x;
                    const auto im = y0 + array_y * set_step_y;
                    const Complex<RealType> c(re, im);

                    const auto point_value = CalculatePoint(c, max_iter);
                    mandelbrotResult[idx] = point_value;
                });
            mandelbrotResult.synchronize();
            mandelbrotResult.discard_data();
        }

        static void cpuJuliaKernel(const Complex<RealType>& k, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
        {
            const int num_points = display_w * display_h;

            const int wx = static_cast<int>(display_w);
            const int wy = static_cast<int>(display_h);

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
                const Complex<RealType> c(re, im);

                const auto point_value = CalculateJulia(c, k, max_iter);
                iters[i] = point_value;
            }
        }

        static void cpuSpecialKernel(int func, const Complex<RealType>& k, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
        {
            const int num_points = display_w * display_h;

            const int wx = static_cast<int>(display_w);
            const int wy = static_cast<int>(display_h);

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
                const Complex<RealType> c(re, im);

                const auto point_value = func == 0 ? CalculateSpecial_0(c, k, max_iter) : CalculateSpecial_1(c, k, max_iter);
                iters[i] = point_value;
            }
        }


        static void gpuSpecialKernel(int func, const Complex<RealType>& k, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
        {
            const auto num_points = display_w * display_h;

            const auto set_width  = x1 - x0;
            const auto set_height = y1 - y0;

            const auto set_step_x = set_width / double(display_w);
            const auto set_step_y = set_height / double(display_h);

            concurrency::extent<1> e(num_points);
            concurrency::array_view<unsigned, 1> mandelbrotResult(e, iters);

            concurrency::parallel_for_each(mandelbrotResult.extent,
                [func, k, display_w, display_h, x0, y0, set_step_x, set_step_y, max_iter, mandelbrotResult](index<1> idx) restrict(amp, cpu)
                {
                    const auto array_x = idx[0] % display_w;
                    const auto array_y = display_h - idx[0] / display_w;

                    const auto re = x0 + array_x * set_step_x;
                    const auto im = y0 + array_y * set_step_y;
                    const Complex<RealType> c(re, im);

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
                    case 15: mandelbrotResult[idx] = CalculateSpecial_15(c, k, max_iter); break;
                    case 16: mandelbrotResult[idx] = CalculateSpecial_16(c, k, max_iter); break;
                    case 17: mandelbrotResult[idx] = CalculateSpecial_17(c, k, max_iter); break;
                    case 18: mandelbrotResult[idx] = CalculateSpecial_18(c, k, max_iter); break;
                    case 19: mandelbrotResult[idx] = CalculateSpecial_19(c, k, max_iter); break;
                    }
                });
            mandelbrotResult.synchronize();
            mandelbrotResult.discard_data();
        }

        static void gpuJuliaKernel(const Complex<RealType>& k, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* iters)
        {
            const auto num_points = display_w * display_h;

            const auto set_width = x1 - x0;
            const auto set_height = y1 - y0;

            const auto set_step_x = set_width / double(display_w);
            const auto set_step_y = set_height / double(display_h);

            concurrency::extent<1> e(num_points);
            concurrency::array_view<unsigned, 1> mandelbrotResult(e, iters);


            concurrency::parallel_for_each(mandelbrotResult.extent,
                [k, display_w, display_h, x0, y0, set_step_x, set_step_y, max_iter, mandelbrotResult](index<1> idx) restrict(amp, cpu)
                {
                    const auto array_x = idx[0] % display_w;
                    const auto array_y = display_h - idx[0] / display_w;

                    const auto re = x0 + array_x * set_step_x;
                    const auto im = y0 + array_y * set_step_y;
                    const Complex<RealType> c(re, im);

                    const auto point_value = CalculateJulia(c, k, max_iter);
                    mandelbrotResult[idx] = point_value;
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
        static void gpuCalculationDensity(bool anti_buddha, unsigned display_w, unsigned display_h, double x0, double x1, double y0, double y1, unsigned max_iter, unsigned* dmap)
        {
            const auto num_points = display_w * display_h;

            const auto set_width = x1 - x0;
            const auto set_height = y1 - y0;

            const auto set_step_x = set_width / double(display_w);
            const auto set_step_y = set_height / double(display_h);

            concurrency::extent<1> e(num_points);
            concurrency::array_view<unsigned, 1> edmap(e, dmap);

            const double r_set_step_x = 1.0 / set_step_x;
            const double r_set_step_y = 1.0 / set_step_y;

            concurrency::parallel_for_each(e,
                [anti_buddha, display_w, display_h, x0, y0, set_step_x, set_step_y, r_set_step_x, r_set_step_y, max_iter, edmap](index<1> idx) restrict(amp, cpu)
                {
                    const auto array_x = idx[0] % display_w;
                    const auto array_y = display_h - idx[0] / display_w;

                    const auto re = x0 + array_x * set_step_x;
                    const auto im = y0 + array_y * set_step_y;
                    const Complex<RealType> c(re, im);

                    const bool escaped = anti_buddha ? false : CalculatePoint(c,  max_iter) >= max_iter;
                    if (!escaped)
                    {
                        unsigned iters = 0;

                        Complex<RealType> z(0.0);
                        while (iters < max_iter && SumSquares(z) <= 4.0)
                        {
                            z = z * z + c;

                            const int rx = int((z.Re() - x0) * r_set_step_x + 0.5);
                            const int ry = int(display_h - 1 - ((z.Im() - y0) * r_set_step_y) + 0.5);

                            //This if is needed in case some point flies off outside the display boundaries
                            if (rx >= 0 && ry >= 0 && rx < int(display_w) && ry < int(display_h))
                            {
                                if (edmap[rx + display_w * ry] < max_iter)
                                    edmap[rx + display_w * ry] += 1;
                            }

                            ++iters;
                        }
                    }
                });
            edmap.synchronize();
            edmap.discard_data();
        }

        static void gpuPaletteKernel(unsigned size, unsigned* iters, unsigned* bmp, unsigned size_palette, rgb* palette)
        {
            concurrency::extent<1> e(size);
            concurrency::array_view<unsigned, 1> av_iters(e, iters);
            concurrency::array_view<unsigned, 1> av_bmp(e, bmp);

            concurrency::extent<1> ep(size_palette);
            concurrency::array_view<unsigned, 1> av_palette(ep, (unsigned*)palette);

            concurrency::parallel_for_each(e, [av_iters, av_bmp, av_palette](index<1> idx) restrict(amp, cpu)
                {
                    av_bmp[idx] = av_palette[av_iters[idx]];
                });
        }

        static void cpuPaletteKernel(unsigned size, unsigned* iters, unsigned* bmp, unsigned size_palette, rgb* palette)
        {
            const auto* ipalette = (unsigned*)palette;
            const auto isize = int(size);
            #pragma omp parallel for
            for (int i = 0; i < isize; ++i)
            {
                bmp[i] = ipalette[iters[i]];
            }
        }

        //Calculation Mandelbrot set iterations
        inline static unsigned CalculatePoint(const Complex<RealType>& c, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<RealType> z(0.0, 0.0);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z * z + c;
                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateJulia(const Complex<RealType>& c, const Complex<RealType>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<RealType> z = c;
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z * z + k;
                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_0(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                const auto& z2 = z * z;
                const auto& z3 = z * z2;

                z = z - (z2 - k) / (z2 + k) + (z3 - k) / (z3 + k);

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_1(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                const auto& z2 = z * z;
                z = (z2 - k) * (z + k) / z2;

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_2(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(k);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + c;

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_3(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z * z * z * z + k;

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_4(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z * z * z * z * z + k;

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_5(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z * z * z * z * z * z + k;

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_6(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = (z + z * z * z * z * z * z + z * z * z * z * z) / (z * z * z) + k;

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_7(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z * z + Sin(z) + k;

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_8(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Cos(z * k);

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_9(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Cos(Sin(z + k));

                ++iters;
            }

            return iters;
        }

        inline static unsigned CalculateSpecial_10(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Tan(z + k);

                ++iters;
            }

            return iters;
        }
        inline static unsigned CalculateSpecial_11(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Tan(z + k);

                ++iters;
            }

            return iters;
        }
        inline static unsigned CalculateSpecial_12(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Tan(z + k);

                ++iters;
            }

            return iters;
        }
        inline static unsigned CalculateSpecial_13(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Tan(z + k);

                ++iters;
            }

            return iters;
        }
        inline static unsigned CalculateSpecial_14(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Tan(z + k);

                ++iters;
            }

            return iters;
        }
        inline static unsigned CalculateSpecial_15(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Tan(z + k);

                ++iters;
            }

            return iters;
        }
        inline static unsigned CalculateSpecial_16(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Tan(z + k);

                ++iters;
            }

            return iters;
        }
        inline static unsigned CalculateSpecial_17(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Tan(z + k);

                ++iters;
            }

            return iters;
        }
        inline static unsigned CalculateSpecial_18(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Tan(z + k);

                ++iters;
            }

            return iters;
        }
        inline static unsigned CalculateSpecial_19(const Complex<double>& c, const Complex<double>& k, unsigned maxIters) restrict(amp, cpu)
        {
            unsigned iters = 0;

            Complex<double> z(c);
            while (iters < maxIters && SumSquares(z) <= 4.0)
            {
                z = z + Tan(z + k);

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
        RealType m_x0;
        RealType m_x1;
        RealType m_xstep;

        RealType m_y0;
        RealType m_y1;
        RealType m_ystep;

        unsigned m_wx;
        unsigned m_wy;

        unsigned* m_arr;
        unsigned* m_bmp;
        unsigned* m_density;
    };
}

