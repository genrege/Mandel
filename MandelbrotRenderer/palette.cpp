#include "palette.h"

namespace MathsEx
{
void iteration_palette::apply(unsigned max_iterations, const cache_memory<unsigned>& data, cache_memory<rgb>& bmp, unsigned offset)
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


    #pragma omp         parallel for
    for (int i = 0; i < static_cast<int>(data.size()); ++i)
    {
        unsigned index = data[i] != max_iterations ? (data[i] + offset) % ( max_iterations) : data[i];
        bmp[i] = palette_[index];
    }
}
}