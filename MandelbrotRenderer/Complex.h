#pragma once                                 

#include <cmath>
#include <amp.h>
#include <amp_math.h>

namespace MathsEx
{
    template <class RealType> class Complex
    {
    public:
        Complex() restrict(amp, cpu) : m_re(0), m_im(0)
        {
        }

        Complex(RealType re, RealType im) restrict(amp, cpu) : m_re(re), m_im(im)
        {
        }

        Complex(RealType re) restrict(amp, cpu) : m_re(re), m_im(0)
        {
        }

        Complex(const Complex& rhs) restrict(amp, cpu) : m_re(rhs.m_re), m_im(rhs.m_im)
        {
        }

        inline Complex& operator=(const Complex& rhs) restrict(amp, cpu)
        {
            if (&rhs != this)
            {
                m_re = rhs.m_re;
                m_im = rhs.m_im;
            }

            return *this;
        }

        RealType Re() const restrict(amp, cpu)
        {
            return m_re;
        }

        RealType Im() const restrict(amp, cpu)
        {
            return m_im;
        }

        RealType& Re() restrict(amp, cpu)
        {
            return m_re;
        }

        RealType& Im() restrict(amp, cpu)
        {
            return m_im;
        }

        inline Complex<RealType> Conjugate() const restrict(amp, cpu)
        {
            return Complex{ m_re, -m_im };
        }

        inline Complex<RealType> operator+(RealType x) const restrict(amp, cpu)
        {
            return { m_re + x, m_im };
        }

        inline Complex<RealType> operator-(RealType x) const restrict(amp, cpu)
        {
            return { m_re - x, m_im };
        }

        inline Complex<RealType> operator+(const Complex& rhs) const restrict(amp, cpu)
        {
            return { m_re + rhs.m_re, m_im + rhs.m_im };
        }

        inline Complex<RealType> operator-(const Complex<RealType>& rhs) const restrict(amp, cpu)
        {
            return { m_re - rhs.m_re, m_im - rhs.m_im };
        }

        inline Complex<RealType> operator*(const Complex<RealType>& rhs) const restrict(amp, cpu)
        {
            return { m_re * rhs.m_re - m_im * rhs.m_im, m_im * rhs.m_re + m_re * rhs.m_im };
        }

        inline Complex<RealType> operator/(const Complex<RealType>& rhs) const restrict(amp, cpu)
        {
            const RealType a = m_re;
            const RealType b = m_im;
            const RealType c = rhs.m_re;
            const RealType d = rhs.m_im;
            const RealType denominator = (c * c + d * d);

            return { (a * c + b * d) / denominator, (b * c - a * d) / denominator };
        }

    private:
        RealType m_re;
        RealType m_im;

        friend RealType Re(const Complex<RealType>& z) restrict(amp, cpu);
        friend RealType Im(const Complex<RealType>& z) restrict(amp, cpu);
        friend RealType SumSquares(const Complex<RealType>& z) restrict(amp, cpu);
        friend RealType Mod(const Complex<RealType>& z) restrict (amp, cpu);
        friend Complex<RealType> Reciprocal(const Complex<RealType>& z) restrict(amp, cpu);

        friend Complex<RealType> operator*(RealType re, const Complex<RealType>& z) restrict(amp, cpu);
        friend Complex<RealType> operator/(RealType re, const Complex<RealType>& z) restrict(amp, cpu);
    };

    template <class RealType> RealType Re(const Complex<RealType>& z) restrict(amp, cpu)
    {
        return z.m_re;
    }

    template <class RealType> RealType Im(const Complex<RealType>& z) restrict(amp, cpu)
    {
        return z.m_im;
    }

    template <class RealType> RealType SumSquares(const Complex<RealType>& z) restrict(amp, cpu)
    {
        return z.m_re * z.m_re + z.m_im * z.m_im;
    }

    float SumSquares(const Complex<float>& z) restrict(amp, cpu)
    {
        return z.m_re * z.m_re + z.m_im * z.m_im;
    }

    double SumSquares(const Complex<double>& z) restrict(amp, cpu)
    {
        return z.m_re * z.m_re + z.m_im * z.m_im;
    }

    template <class RealType> RealType Mod(const Complex<RealType>& z) restrict(amp, cpu)
    {
        return concurrency::precise_math::sqrt(z.Re() * z.Re() + z.Im() * z.Im());
    }

    double Mod(const Complex<double>& z) restrict(amp, cpu)
    {
        return concurrency::precise_math::sqrt(z.Re() * z.Re() + z.Im() * z.Im());
    }

    Complex<double> Sin(const Complex<double>& z) restrict(amp, cpu)
    {   
        using namespace concurrency::precise_math;
        return Complex<double>(sin(z.Re()) * cos(z.Im()) + cos(z.Im()) * sin(z.Re()));
    }

    template <class RealType> Complex<RealType> Reciprocal(const Complex<RealType>& z) restrict(amp, cpu)
    {
        return Complex<RealType>(1, 0) / z;
    }

    template <class RealType> Complex<RealType> operator*(RealType re, const Complex<RealType>& z) restrict(amp, cpu)
    {
        return Complex<RealType>(re * z.m_re, re * z.m_im);
    }

    Complex<double> operator*(double re, const Complex<double>& z) restrict(amp, cpu)
    {
        return Complex<double>(re * z.m_re, re * z.m_im);
    }

    template <class RealType> Complex<RealType> operator/(RealType re, const Complex<RealType>& z) restrict(amp, cpu)
    {
        return Complex<RealType>(re) / z;
    }

}   //namespace MathsEx

