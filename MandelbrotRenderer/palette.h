#pragma once

#include <math.h>

#include "cached_memory.h"

namespace MathsEx
{
struct rgb
{
    rgb() : b(0), g(0), r(0), pad(0xFF) {}
    unsigned char b;
    unsigned char g;
    unsigned char r;
    unsigned char pad;
};


class iteration_palette
{
public:
    iteration_palette() = default;

    void apply(unsigned max_iterations, const cache_memory<unsigned>& data, cache_memory<rgb>& bmp, unsigned offset = 0);

private:
    cache_memory<rgb> palette_;
};
}