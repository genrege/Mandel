#pragma once                                 

#include <cmath>
#include <amp.h>
#include <amp_math.h>

namespace fractals
{
    class complex
    {
    public:
        complex() restrict(amp, cpu) : m_re(0), m_im(0)
        {
        }

        complex(double re, double im) restrict(amp, cpu) : m_re(re), m_im(im)
        {
        }

        complex(double re) restrict(amp, cpu) : m_re(re), m_im(0)
        {
        }

        complex(const complex& rhs) restrict(amp, cpu) : m_re(rhs.m_re), m_im(rhs.m_im)
        {
        }

        inline complex& operator=(const complex& rhs) restrict(amp, cpu)
        {
            if (&rhs != this)
            {
                m_re = rhs.m_re;
                m_im = rhs.m_im;
            }

            return *this;
        }

        double Re() const restrict(amp, cpu)
        {
            return m_re;
        }

        double Im() const restrict(amp, cpu)
        {
            return m_im;
        }

        double& Re() restrict(amp, cpu)
        {
            return m_re;
        }

        double& Im() restrict(amp, cpu)
        {
            return m_im;
        }



        inline complex Conjugate() const restrict(amp, cpu)
        {
            return complex{ m_re, -m_im };
        }

        inline complex operator+(double x) const restrict(amp, cpu)
        {
            return { m_re + x, m_im };
        }

        inline complex operator-(double x) const restrict(amp, cpu)
        {
            return { m_re - x, m_im };
        }

        inline complex operator+(const complex& rhs) const restrict(amp, cpu)
        {
            return { m_re + rhs.m_re, m_im + rhs.m_im };
        }

        inline complex operator-(const complex& rhs) const restrict(amp, cpu)
        {
            return { m_re - rhs.m_re, m_im - rhs.m_im };
        }

        inline complex operator*(const complex& rhs) const restrict(amp, cpu)
        {
            return { m_re * rhs.m_re - m_im * rhs.m_im, m_im * rhs.m_re + m_re * rhs.m_im };
        }

        inline complex squared()  const restrict(amp, cpu)
        {
            return { m_re * m_re - m_im * m_im, 2 * m_re * m_im };
        }

        inline complex operator/(const complex& rhs) const restrict(amp, cpu)
        {
            const double a = m_re;
            const double b = m_im;
            const double c = rhs.m_re;
            const double d = rhs.m_im;
            const double denominator = (c * c + d * d);

            return { (a * c + b * d) / denominator, (b * c - a * d) / denominator };
        }

    private:
        double m_re;
        double m_im;

        friend double Re(const complex& z) restrict(amp, cpu);
        friend double Im(const complex& z) restrict(amp, cpu);
        friend double SumSquares(const complex& z) restrict(amp, cpu);
        friend double Mod(const complex& z) restrict (amp, cpu);
        friend complex Reciprocal(const complex& z) restrict(amp, cpu);

        friend complex operator*(double re, const complex& z) restrict(amp, cpu);
        friend complex operator/(double re, const complex& z) restrict(amp, cpu);
    };

    inline double SumSquares(const complex& z) restrict(amp, cpu)
    {
        return z.m_re * z.m_re + z.m_im * z.m_im;
    }

    inline double Mod(const complex& z) restrict(amp, cpu)
    {
        return concurrency::precise_math::sqrt(z.Re() * z.Re() + z.Im() * z.Im());
    }

    inline complex Sin(const complex& z) restrict(amp, cpu)
    {
        using namespace concurrency::precise_math;
        return complex(sin(z.Re()) * cosh(z.Im()), cos(z.Im()) * sinh(z.Re()));
    }

    inline complex Cos(const complex& z) restrict(amp, cpu)
    {
        using namespace concurrency::precise_math;
        return complex(cos(z.Re()) * cosh(z.Im()),  -sin(z.Re()) * sinh(z.Im()));
    }

    inline complex Tan(const complex& z) restrict(amp, cpu)
    {
        using namespace concurrency::precise_math;
        return Sin(z) / Cos(z);
    }

    inline complex Reciprocal(const complex& z) restrict(amp, cpu)
    {
        return complex(1, 0) / z;
    }

    inline complex operator*(double re, const complex& z) restrict(amp, cpu)
    {
        return complex(re * z.m_re, re * z.m_im);
    }

    inline complex operator/(double re, const complex& z) restrict(amp, cpu)
    {
        return complex(re) / z;
    }

    inline complex Sqrt(const complex& z) restrict(amp, cpu)
    {
        using namespace concurrency::precise_math;

        const auto x = z.Re();
        const auto y = z.Im();

        const auto xy2 = x * x + y * y;
        const auto rxy2 = sqrt(xy2);

        const auto re = sqrt((rxy2 + x) * 0.5);
        const auto im = sqrt((rxy2 - x) * 0.5);
        return { re, im };
    }

}   //namespace MathsEx

