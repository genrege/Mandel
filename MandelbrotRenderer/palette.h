#pragma once

#include <math.h>

#include "cached_memory.h"

namespace fractals
{
struct rgb
{
    unsigned char b;
    unsigned char g;
    unsigned char r;
    unsigned char pad;
};

class iteration_palette
{
public:
    iteration_palette() = default;

    void update(unsigned max_iterations);
    void update_for_buddha(unsigned* density, unsigned size_density);
    void apply(unsigned max_iterations, const cache_memory<unsigned>& data, cache_memory<rgb>& bmp, unsigned offset = 0);
    unsigned* data()
    {
        return (unsigned*)palette_.access();
    }

private:
    cache_memory<rgb> palette_;
};
}