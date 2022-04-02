#pragma once                                 

#include <cmath>
#include <amp.h>
#include <amp_math.h>

namespace MathsEx
{
    class Complex
    {
    public:
        Complex() restrict(amp, cpu) : m_re(0), m_im(0)
        {
        }

        Complex(double re, double im) restrict(amp, cpu) : m_re(re), m_im(im)
        {
        }

        Complex(double re) restrict(amp, cpu) : m_re(re), m_im(0)
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



        inline Complex Conjugate() const restrict(amp, cpu)
        {
            return Complex{ m_re, -m_im };
        }

        inline Complex operator+(double x) const restrict(amp, cpu)
        {
            return { m_re + x, m_im };
        }

        inline Complex operator-(double x) const restrict(amp, cpu)
        {
            return { m_re - x, m_im };
        }

        inline Complex operator+(const Complex& rhs) const restrict(amp, cpu)
        {
            return { m_re + rhs.m_re, m_im + rhs.m_im };
        }

        inline Complex operator-(const Complex& rhs) const restrict(amp, cpu)
        {
            return { m_re - rhs.m_re, m_im - rhs.m_im };
        }

        inline Complex operator*(const Complex& rhs) const restrict(amp, cpu)
        {
            return { m_re * rhs.m_re - m_im * rhs.m_im, m_im * rhs.m_re + m_re * rhs.m_im };
        }

        inline Complex squared()  const restrict(amp, cpu)
        {
            return { m_re * m_re - m_im * m_im, 2 * m_re * m_im };
        }

        inline Complex operator/(const Complex& rhs) const restrict(amp, cpu)
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

        friend double Re(const Complex& z) restrict(amp, cpu);
        friend double Im(const Complex& z) restrict(amp, cpu);
        friend double SumSquares(const Complex& z) restrict(amp, cpu);
        friend double Mod(const Complex& z) restrict (amp, cpu);
        friend Complex Reciprocal(const Complex& z) restrict(amp, cpu);

        friend Complex operator*(double re, const Complex& z) restrict(amp, cpu);
        friend Complex operator/(double re, const Complex& z) restrict(amp, cpu);
    };

    double Re(const Complex& z) restrict(amp, cpu);
    double Im(const Complex& z) restrict(amp, cpu);
    double SumSquares(const Complex& z) restrict(amp, cpu);
    double Mod(const Complex& z) restrict(amp, cpu);
    Complex Sin(const Complex& z) restrict(amp, cpu);
    Complex Cos(const Complex& z) restrict(amp, cpu);
    Complex Tan(const Complex& z) restrict(amp, cpu);
    Complex Reciprocal(const Complex& z) restrict(amp, cpu);
    Complex operator*(double re, const Complex& z) restrict(amp, cpu);
    Complex operator/(double re, const Complex& z) restrict(amp, cpu);
    Complex Sqrt(const Complex& z) restrict(amp, cpu);

}   //namespace MathsEx

