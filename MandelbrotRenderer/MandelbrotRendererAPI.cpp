
#include "MandelbrotRendererAPI.h"

#include "Complex.h"
#include "MandelbrotSet.h"
#include "MandelbrotSetCL.h"
#include "fractal.h"
#include "hdc_utils.h"
#include "../MandelbrotRendererCUDA/MandelbrotSetCUDA.h"

MathsEx::MandelbrotSet mset;
MathsEx::mandelbrot_set mandelbrotset;
MathsEx::julia_set juliaset;

MathsEx::iteration_palette default_palette;
cache_memory<MathsEx::rgb> display_bitmap;

namespace
{
    auto gpu_accelerator(int gpuIndex)
    {
        const auto& accls = accelerator::get_all();
        return accelerator(accls[gpuIndex]).default_view;
    }
}

int countGPU()
{
    std::vector<accelerator> accls = accelerator::get_all();

    int count = 0;
    for (size_t i = 0; i < accls.size(); ++i)
    {
        const auto& accl = accls[i];

        const auto& desc = std::to_wstring(i) + L";" + accl.description;
        
        if (desc.find(L"NVIDIA") != std::string::npos) //NVIDIA has a CUDA option, so add an extra one
            ++count;
        ++count;
    }

    return static_cast<int>(accls.size());
}

extern "C" void GPU(SAFEARRAY** ppsa)
{
    const auto& accls = accelerator::get_all();

    size_t count = accls.size();;

    SAFEARRAYBOUND rgsa;
    rgsa.lLbound = 0;
    rgsa.cElements = static_cast<ULONG>(count);
    *ppsa = SafeArrayCreate(VT_BSTR, 1, &rgsa);
    
    unsigned* result;
    SafeArrayLock(*ppsa);
    SafeArrayAccessData(*ppsa, (void HUGEP**)&result);

    size_t index = 0;
    for (size_t i = 0; i < accls.size(); ++i)
    {
        const auto& accl = accls[i];
        const auto& desc = std::to_wstring(i) + L";" + accl.description;
        (void)SafeArrayPutElement(*ppsa, (LONG*)&index, SysAllocString(desc.c_str()));

        if (desc.find(L"NVIDIA") != std::string::npos)
        {
            ++index;
            (void)SafeArrayPutElement(*ppsa, (LONG*)&index, SysAllocString((desc + L" (CUDA)").c_str()));
        }

        ++index;
    }

    SafeArrayUnaccessData(*ppsa);
    SafeArrayUnlock(*ppsa);
}

extern "C" void renderMandelbrot(int gpuIndex, HDC hdc, bool gpu, bool cuda, int max_iterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset)
{
    mandelbrotset.set_scale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    if (gpu)
    {
        if (cuda)
        {
            mandelbrotset.calculate_set_cuda(max_iterations);
        }
        else
        {
            mandelbrotset.calculate_set_amp(gpu_accelerator(gpuIndex), max_iterations);
        }
    }
    else
        mandelbrotset.calculate_set_cpu(max_iterations); 

    display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
    default_palette.apply(max_iterations, mandelbrotset.data(), display_bitmap, palette_offset);

    sendToDisplay(hdc, screenWidth, screenHeight, display_bitmap.access_as<unsigned int>());
}

extern "C" void renderJulia(int gpuIndex, HDC hdc, bool gpu, bool cuda, int max_iterations, double re, double im, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset)
{
    juliaset.set_scale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    if (gpu)
    {
        if (cuda)
        {
            juliaset.calculate_set_cuda(MathsEx::Complex(re, im), max_iterations);
        }
        else
        {
            juliaset.calculate_set_amp(gpu_accelerator(gpuIndex), MathsEx::Complex(re, im), max_iterations);
        }
    }
    else
        juliaset.calculate_set_cpu(MathsEx::Complex(re, im), max_iterations);

    display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
    default_palette.apply(max_iterations, juliaset.data(), display_bitmap, palette_offset);

    sendToDisplay(hdc, screenWidth, screenHeight, display_bitmap.access_as<unsigned int>());
}

extern "C" void renderBuddha(int gpuIndex, HDC hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax)
{
    mset.SetScale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    mset.CalculateBuddha(gpu_accelerator(gpuIndex), antiBuddha, maxIterations);

    sendToDisplay(hdc, screenWidth, screenHeight, mset.bmp());
}

extern "C" void saveMandelbrotBitmap(int gpuIndex, HDC hdc, int max_iterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename)
{
    mandelbrotset.set_scale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    mandelbrotset.calculate_set_amp(gpu_accelerator(gpuIndex), max_iterations);
    display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
    default_palette.apply(max_iterations, mandelbrotset.data(), display_bitmap, palette_offset);
    sendToBitmap(hdc, screenWidth, screenHeight, display_bitmap.access_as<unsigned int>(), filename);

}

extern "C" void saveJuliaBitmap(int gpuIndex, double re, double im, HDC hdc, int max_iterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename)
{
    juliaset.set_scale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    juliaset.calculate_set_amp(gpu_accelerator(gpuIndex), MathsEx::Complex(re, im), max_iterations);
    display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
    default_palette.apply(max_iterations, juliaset.data(), display_bitmap, palette_offset);
    sendToBitmap(hdc, screenWidth, screenHeight, display_bitmap.access_as<unsigned int>(), filename);
}

extern "C" void saveBuddhaBitmap(int gpuIndex, HDC hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, const char* filename)
{
    mset.SetScale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    mset.CalculateBuddha(gpu_accelerator(gpuIndex), antiBuddha, maxIterations);
    sendToBitmap(hdc, screenWidth, screenHeight, mset.bmp(), filename);
}

extern "C" void saveMandelbrotJPG(int gpuIndex, HDC hdc, int max_iterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename)
{
    mandelbrotset.set_scale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    mandelbrotset.calculate_set_amp(gpu_accelerator(gpuIndex), max_iterations);
    display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
    default_palette.apply(max_iterations, mandelbrotset.data(), display_bitmap, palette_offset);
    sendToJPEG(hdc, screenWidth, screenHeight, display_bitmap.access_as<unsigned>(), filename);
}

extern "C" void saveJuliaJPG(int gpuIndex, double re, double im, HDC hdc, int max_iterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename)
{
    juliaset.set_scale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    juliaset.calculate_set_amp(gpu_accelerator(gpuIndex), MathsEx::Complex(re, im), max_iterations);
    display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
    default_palette.apply(max_iterations, juliaset.data(), display_bitmap, palette_offset);
    sendToJPEG(hdc, screenWidth, screenHeight, mset.bmp(), filename);
}

extern "C" void saveBuddhaJPG(int gpuIndex, HDC hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, const char* filename)
{
    mset.SetScale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    mset.CalculateBuddha(gpu_accelerator(gpuIndex), antiBuddha, maxIterations);
    sendToJPEG(hdc, screenWidth, screenHeight, mset.bmp(), filename);
}

//  Managed client API to calculate the set data only
extern "C" DLL_API void calculateMandelbrot(int gpuIndex, bool gpu, bool cuda, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY** ppsa)
{
    using namespace MathsEx;

    const unsigned array_size = width * height;
    
    SAFEARRAYBOUND rgsa;
    rgsa.lLbound = 0;
    rgsa.cElements = array_size;
    *ppsa = SafeArrayCreate(VT_I4, 1, &rgsa);
    
    unsigned* result;
    SafeArrayLock(*ppsa);
    SafeArrayAccessData(*ppsa, (void HUGEP**)&result);

    if (gpu)
    {
        if (cuda)
            kernel_cuda::mandelbrot_kernel(width, height, xMin, xMax, yMin, yMax, maxIterations, result);
        else
            kernel_amp::mandelbrot_kernel(gpu_accelerator(gpuIndex), width, height, xMin, xMax, yMin, yMax, maxIterations, result);
    }
    else
        kernel_cpu::mandelbrot_kernel(width, height, xMin, xMax, yMin, yMax, maxIterations, result);

    SafeArrayUnaccessData(*ppsa);
    SafeArrayUnlock(*ppsa);
}

//  Managed client API to calculate the Julia set data only
extern "C" DLL_API void calculateJulia(int gpuIndex, double re, double im, bool gpu, bool cuda, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY * *ppsa)
{
    using namespace MathsEx;

    const unsigned array_size = width * height;

    SAFEARRAYBOUND rgsa;
    rgsa.lLbound = 0;
    rgsa.cElements = array_size;
    *ppsa = SafeArrayCreate(VT_I4, 1, &rgsa);

    unsigned* result;
    SafeArrayLock(*ppsa);
    SafeArrayAccessData(*ppsa, (void HUGEP**) & result);

    if (gpu)
    {
        if (cuda)
            kernel_cuda::julia_kernel(width, height, xMin, xMax, yMin, yMax, re, im, maxIterations, result);
        else
            kernel_amp::julia_kernel(gpu_accelerator(gpuIndex), width, height, xMin, xMax, yMin, yMax, re, im, maxIterations, result);
    }
    else
        kernel_cpu::julia_kernel(width, height, xMin, xMax, yMin, yMax, re, im, maxIterations, result);

    SafeArrayUnaccessData(*ppsa);
    SafeArrayUnlock(*ppsa);
}

//  Managed client API to calculate the Julia set data only
extern "C" DLL_API void calculateJulia2(int gpuIndex, double re, double im, bool gpu, bool cuda, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY * *ppsa)
{
    using namespace MathsEx;

    unsigned* result;
    SafeArrayLock(*ppsa);
    SafeArrayAccessData(*ppsa, (void HUGEP**) & result);

    if (gpu)
    {
        if (cuda)
            kernel_cuda::julia_kernel(width, height, xMin, xMax, yMin, yMax, re, im, maxIterations, result);
        else
            kernel_amp::julia_kernel(gpu_accelerator(gpuIndex), width, height, xMin, xMax, yMin, yMax, re, im, maxIterations, result);
    }
    else
        kernel_cpu::julia_kernel(width, height, xMin, xMax, yMin, yMax, re, im, maxIterations, result);

    SafeArrayUnaccessData(*ppsa);
    SafeArrayUnlock(*ppsa);
}

extern "C" DLL_API void calculateSpecial(int gpuIndex, int func, double re, double im, bool gpu, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY * *ppsa)
{
    const unsigned array_size = width * height;

    SAFEARRAYBOUND rgsa;
    rgsa.lLbound = 0;
    rgsa.cElements = array_size;
    *ppsa = SafeArrayCreate(VT_I4, 1, &rgsa);

    unsigned* result;
    SafeArrayLock(*ppsa);
    SafeArrayAccessData(*ppsa, (void HUGEP**) & result);

    if (gpu)
        MathsEx::MandelbrotSet::gpuSpecialKernel(gpu_accelerator(gpuIndex), func, MathsEx::Complex(re, im), width, height, xMin, xMax, yMin, yMax, maxIterations, result);
    else
        MathsEx::MandelbrotSet::cpuSpecialKernel(func, MathsEx::Complex(re, im), width, height, xMin, xMax, yMin, yMax, maxIterations, result);

    SafeArrayUnaccessData(*ppsa);
    SafeArrayUnlock(*ppsa);
}

extern "C" DLL_API void calculateBuddha(int gpuIndex, bool antiBuddha, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY * *ppsa)
{
    const unsigned array_size = width * height;

    SAFEARRAYBOUND rgsa;
    rgsa.lLbound = 0;
    rgsa.cElements = array_size;
    *ppsa = SafeArrayCreate(VT_I4, 1, &rgsa);

    unsigned* result;
    SafeArrayLock(*ppsa);
    SafeArrayAccessData(*ppsa, (void HUGEP**) & result);

    MathsEx::MandelbrotSet::gpuCalculationDensity(gpu_accelerator(gpuIndex), antiBuddha, width, height, xMin, xMax, yMin, yMax, maxIterations, result);

    SafeArrayUnaccessData(*ppsa);
    SafeArrayUnlock(*ppsa);
}

//  Managed client API to transform data according to an input palette.  
//  The operation is ppsaResult[i] = palette[input[i]].
extern "C" void paletteTransform(int gpuIndex, SAFEARRAY* input, SAFEARRAY* palette, SAFEARRAY** ppsaResult)
{
    SafeArrayLock(input);
    unsigned array_size = input->rgsabound->cElements;
    unsigned* input_data;
    SafeArrayAccessData(input, (void HUGEP**)&input_data);

    SafeArrayLock(palette);
    unsigned palette_size = palette->rgsabound->cElements;
    unsigned* palette_data;
    SafeArrayAccessData(palette, (void HUGEP**)&palette_data);

    SAFEARRAYBOUND rgsa;
    rgsa.lLbound = 0;
    rgsa.cElements = array_size;
    *ppsaResult = SafeArrayCreate(VT_I4, 1, &rgsa);

    unsigned* result;
    SafeArrayLock(*ppsaResult);
    SafeArrayAccessData(*ppsaResult, (void HUGEP**) &result);

    MathsEx::MandelbrotSet::gpuPaletteKernel(gpu_accelerator(gpuIndex), array_size, input_data, result, palette_size, (MathsEx::rgb*)palette_data);

    SafeArrayUnaccessData(*ppsaResult);
    SafeArrayUnlock(*ppsaResult);

    SafeArrayUnaccessData(palette);
    SafeArrayUnlock(palette);

    SafeArrayUnaccessData(input);
    SafeArrayUnlock(input);
}

//  Managed client API to transform data according to an input palette.  
//  The operation is ppsaResult[i] = palette[input[i]].
extern "C" void paletteTransform2(int gpuIndex, SAFEARRAY* input, SAFEARRAY* palette, SAFEARRAY** ppsaResult)
{
    SafeArrayLock(input);
    unsigned array_size = input->rgsabound->cElements;
    unsigned* input_data;
    SafeArrayAccessData(input, (void HUGEP**)&input_data);

    SafeArrayLock(palette);
    unsigned palette_size = palette->rgsabound->cElements;
    unsigned* palette_data;
    SafeArrayAccessData(palette, (void HUGEP**)&palette_data);

    unsigned* result;
    SafeArrayLock(*ppsaResult);
    SafeArrayAccessData(*ppsaResult, (void HUGEP**) &result);

    MathsEx::MandelbrotSet::gpuPaletteKernel(gpu_accelerator(gpuIndex), array_size, input_data, result, palette_size, (MathsEx::rgb*)palette_data);

    SafeArrayUnaccessData(*ppsaResult);
    SafeArrayUnlock(*ppsaResult);

    SafeArrayUnaccessData(palette);
    SafeArrayUnlock(palette);

    SafeArrayUnaccessData(input);
    SafeArrayUnlock(input);
}

template <class F>
void renderArrayToDevice(SAFEARRAY* input, F f)
{
    SafeArrayLock(input);
    unsigned* input_data;
    SafeArrayAccessData(input, (void HUGEP**) & input_data);

    f(input_data);

    SafeArrayUnaccessData(input);
    SafeArrayUnlock(input);
}

//  Managed client API to render an array of data to the given device context
void renderArrayToDisplay(HDC hdc, int width, int height, SAFEARRAY* input)
{
    renderArrayToDevice(input, [&](auto* input_data) {sendToDisplay(hdc, width, height, input_data); });
}

//  Managed client API to render to a bitmap
void renderArrayToBitmap(HDC hdc, int width, int height, SAFEARRAY* input, const char* filename)
{
    renderArrayToDevice(input, [&](auto* input_data) {sendToBitmap(hdc, width, height, input_data, filename); });
}

//  Managed client API to render to a JPEG
void renderArrayToJPEG(HDC hdc, int width, int height, SAFEARRAY* input, const char* filename)
{
    renderArrayToDevice(input, [&](auto* input_data) {sendToJPEG(hdc, width, height, input_data, filename); });
    
}

