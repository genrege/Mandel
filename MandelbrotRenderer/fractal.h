#pragma once
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

template <class RealType> 
class fractal 
{
public:
    fractal() : m_x0(0), m_x1(0), m_y0(0), m_y1(0), m_wx(0), m_wy(0)
    {
    }

private:
    //Virtual dimensions
    RealType m_x0;
    RealType m_x1;
    RealType m_y0;
    RealType m_y1;

    //Display dimensions
    unsigned m_wx;
    unsigned m_wy;

    cache_memory<rgb>       m_palette;
    cache_memory<unsigned>  m_arr;


};

}