#include "Complex.h"

namespace MathsEx
{
    double Re(const Complex& z) restrict(amp, cpu)
    {
        return z.m_re;
    }

    double Im(const Complex& z) restrict(amp, cpu)
    {
        return z.m_im;
    }

    double SumSquares(const Complex& z) restrict(amp, cpu)
    {
        return z.m_re * z.m_re + z.m_im * z.m_im;
    }

    double Mod(const Complex& z) restrict(amp, cpu)
    {
        return concurrency::precise_math::sqrt(z.Re() * z.Re() + z.Im() * z.Im());
    }

    Complex Sin(const Complex& z) restrict(amp, cpu)
    {
        using namespace concurrency::precise_math;
        return Complex(sin(z.Re()) * cos(z.Im()) + cos(z.Im()) * sin(z.Re()));
    }

    Complex Cos(const Complex& z) restrict(amp, cpu)
    {
        using namespace concurrency::precise_math;
        return Complex(cos(z.Re()) * cosh(z.Im()) - sin(z.Im()) * sinh(z.Re()));
    }

    Complex Tan(const Complex& z) restrict(amp, cpu)
    {
        using namespace concurrency::precise_math;
        return Sin(z) / Cos(z);
    }

    Complex Reciprocal(const Complex& z) restrict(amp, cpu)
    {
        return Complex(1, 0) / z;
    }

    Complex operator*(double re, const Complex& z) restrict(amp, cpu)
    {
        return Complex(re * z.m_re, re * z.m_im);
    }

    Complex operator/(double re, const Complex& z) restrict(amp, cpu)
    {
        return Complex(re) / z;
    }

    Complex Sqrt(const Complex& z) restrict(amp, cpu)
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
}
