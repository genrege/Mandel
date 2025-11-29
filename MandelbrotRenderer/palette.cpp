#include <algorithm>
#include <cmath>
#include <cstring> 
#include "palette.h"

namespace fractals
{
void iteration_palette::update(unsigned max_iterations)
{
    if (palette_.allocate(max_iterations + 1))
    {
        const float scale = 6.3f / max_iterations;

#pragma omp         parallel for
        for (int i = 0; i < static_cast<int>(max_iterations); ++i)
        {
            palette_[i].r = static_cast<unsigned char>(sin(scale * i + 3) * 127 + 128);
            palette_[i].g = static_cast<unsigned char>(sin(scale * i + 5) * 127 + 128);
            palette_[i].b = static_cast<unsigned char>(sin(scale * i + 1) * 127 + 128);
        }

        palette_[max_iterations].r = 0;
        palette_[max_iterations].g = 0;
        palette_[max_iterations].b = 0;
    }
}

void iteration_palette::update_for_buddha(unsigned* density, unsigned size_density)
{
    unsigned max_density = 0;
    for (size_t i = 0; i < size_density; ++i)
        if (density[i] > max_density) max_density = density[i];

    if (palette_.allocate(max_density))
    {
        const double palette_scale = 1.0 / sqrt(max_density);

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(max_density); ++i)
        {
            if (i < 12)
            {
                palette_[i].r = (unsigned char)std::min(255.0, 255 * sqrt(0.5 * i) * palette_scale);
                palette_[i].g = (unsigned char)std::min(255.0, 255 * sqrt(0.2 * i) * palette_scale);
                palette_[i].b = (unsigned char)std::min(255.0, 255 * sqrt(0.8 * i) * palette_scale);
            }
            else
            {
                palette_[i].r = (unsigned char)std::min(255.0, 255 * sqrt(0.6 * i) * palette_scale);
                palette_[i].g = (unsigned char)std::min(255.0, 255 * sqrt(0.5 * i) * palette_scale);
                palette_[i].b = (unsigned char)std::min(255.0, 255 * sqrt(1.0 * i) * palette_scale);
            }
        }
    }
}

void iteration_palette::apply(unsigned max_iterations, const cache_memory<unsigned>& data, cache_memory<rgb>& bmp, unsigned offset) 
{
    update(max_iterations);

    //TODO - This is too slow on the CPU
#pragma omp         parallel for
    for (int i = 0; i < static_cast<int>(data.size()); ++i)
    {
        unsigned index = data[i] != max_iterations ? (data[i] + offset) % ( max_iterations) : data[i];
        bmp[i] = palette_[index];
    }
}
}