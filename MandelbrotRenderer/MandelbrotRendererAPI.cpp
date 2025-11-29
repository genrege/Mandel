//This API is for C# clients.  C++ clients can use the raw functions in MandelbrotSet.h for max calculation performance.  Rendering can still be done
//with the functions here for convenience without any loss of performance.

#include "MandelbrotRendererAPI.h"

#include "Complex.h"
#include "MandelbrotSet.h"
#include "fractal.h"
#include "hdc_utils.h"
#include "../MandelbrotRendererCUDA/MandelbrotSetCUDA.h"

//This is the "legacy" set
fractals::MandelbrotSet mset;

//New set
fractals::mandelbrot_set mandelbrotset;
fractals::julia_set juliaset;
fractals::buddha_set buddhaset;

fractals::iteration_palette default_palette;
fractals::iteration_palette buddha_palette;
cache_memory<fractals::rgb> display_bitmap;

extern "C" void GPU(SAFEARRAY** ppsa)
{
    const auto& accls = accelerator::get_all();

    const auto count = accls.size() - 1;
    const auto cuda_device_count = mbrot_cuda::device_count();

    SAFEARRAYBOUND rgsa;
    rgsa.lLbound = 0;
    rgsa.cElements = static_cast<ULONG>(count + cuda_device_count);
    *ppsa = SafeArrayCreate(VT_BSTR, 1, &rgsa);
    
    unsigned* result;
    SafeArrayLock(*ppsa);
    SafeArrayAccessData(*ppsa, (void HUGEP**)&result);

    size_t index = 0;
    for (size_t i = 0; i < cuda_device_count; ++i)
    {
        auto device_name = std::to_string(i) + ";CUDA;" + mbrot_cuda::device_name(i);
        OLECHAR dv[256];
        mbstowcs(dv, device_name.c_str(), 256);
        std::wstring wdv = dv;
        auto hr = SafeArrayPutElement(*ppsa, (LONG*)&index, SysAllocString(wdv.c_str()));
        if (SUCCEEDED(hr))
            ++index;
    }
    for (size_t i = 0; i < accls.size(); ++i)
    {
        const auto& accl = accls[i];
        if (accl.description.find(L"CPU accelerator") != std::wstring::npos)
            continue;

        const auto& desc = std::to_wstring(i) + L";C++ AMP;" + accl.description;
        auto hr = SafeArrayPutElement(*ppsa, (LONG*)&index, SysAllocString(desc.c_str()));
        if (SUCCEEDED(hr))
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
            default_palette.update(max_iterations);
            mandelbrotset.calculate_set_cuda(gpuIndex, max_iterations, default_palette.data(), palette_offset);
            sendToDisplay(hdc, screenWidth, screenHeight, mandelbrotset.data());
            return;
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
            default_palette.update(max_iterations);
            juliaset.calculate_set_cuda(gpuIndex, fractals::complex(re, im), max_iterations, default_palette.data(), palette_offset);
            sendToDisplay(hdc, screenWidth, screenHeight, juliaset.data());
            return;
        }
    }
    else
        juliaset.calculate_set_cpu(fractals::complex(re, im), max_iterations);

    display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
    default_palette.apply(max_iterations, juliaset.data(), display_bitmap, palette_offset);

    sendToDisplay(hdc, screenWidth, screenHeight, display_bitmap.access_as<unsigned int>());
}


extern "C" void renderBuddha(int gpuIndex, HDC hdc, bool cuda, bool anti_buddha, int max_iterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset)
{
    if (cuda)
    {
        buddhaset.set_scale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
        display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
        display_bitmap.zero_memory();
        buddhaset.calculate_set_cuda(gpuIndex, anti_buddha, max_iterations);

        buddha_palette.update_for_buddha(buddhaset.data(), buddhaset.wx() * buddhaset.wy());
        buddha_palette.apply(max_iterations, mandelbrotset.data(), display_bitmap, palette_offset);

        buddha_palette.apply(max_iterations, buddhaset.data(), display_bitmap, palette_offset);
        sendToDisplay(hdc, screenWidth, screenHeight, display_bitmap.access_as<unsigned int>());
    }
}

extern "C" void saveMandelbrotBitmap(int gpuIndex, HDC hdc, int max_iterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename)
{
    mandelbrotset.set_scale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    mandelbrotset.calculate_set_cuda(gpuIndex, max_iterations);
    display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
    default_palette.apply(max_iterations, mandelbrotset.data(), display_bitmap, palette_offset);
    sendToBitmap(hdc, screenWidth, screenHeight, display_bitmap.access_as<unsigned int>(), filename);

}

extern "C" void saveJuliaBitmap(int gpuIndex, double re, double im, HDC hdc, int max_iterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename)
{
    juliaset.set_scale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    juliaset.calculate_set_cuda(gpuIndex, fractals::complex(re, im), max_iterations);
    display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
    default_palette.apply(max_iterations, juliaset.data(), display_bitmap, palette_offset);
    sendToBitmap(hdc, screenWidth, screenHeight, display_bitmap.access_as<unsigned int>(), filename);
}

extern "C" void saveMandelbrotJPG(int gpuIndex, HDC hdc, int max_iterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename)
{
    mandelbrotset.set_scale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    mandelbrotset.calculate_set_cuda(gpuIndex, max_iterations);
    display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
    default_palette.apply(max_iterations, mandelbrotset.data(), display_bitmap, palette_offset);
    sendToJPEG(hdc, screenWidth, screenHeight, display_bitmap.access_as<unsigned>(), filename);
}

extern "C" void saveJuliaJPG(int gpuIndex, double re, double im, HDC hdc, int max_iterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename)
{
    juliaset.set_scale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    juliaset.calculate_set_cuda(gpuIndex, fractals::complex(re, im), max_iterations);
    display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
    default_palette.apply(max_iterations, juliaset.data(), display_bitmap, palette_offset);
    sendToJPEG(hdc, screenWidth, screenHeight, display_bitmap.access_as<unsigned>(), filename);
}

//  Managed client API to calculate the set data only
extern "C" DLL_API void calculateMandelbrot(int gpuIndex, bool gpu, bool cuda, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY** ppsa)
{
    using namespace fractals;

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
            kernel_cuda::mandelbrot_kernel(gpuIndex, width, height, xMin, xMax, yMin, yMax, maxIterations, result);
    }
    else
        kernel_cpu::mandelbrot_kernel(width, height, xMin, xMax, yMin, yMax, maxIterations, result);

    SafeArrayUnaccessData(*ppsa);
    SafeArrayUnlock(*ppsa);
}

//  Managed client API to calculate the Julia set data only
extern "C" DLL_API void calculateJulia(int gpuIndex, double re, double im, bool gpu, bool cuda, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY * *ppsa)
{
    using namespace fractals;

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
            kernel_cuda::julia_kernel(gpuIndex, width, height, xMin, xMax, yMin, yMax, re, im, maxIterations, result);
    }
    else
        kernel_cpu::julia_kernel(width, height, xMin, xMax, yMin, yMax, re, im, maxIterations, result);

    SafeArrayUnaccessData(*ppsa);
    SafeArrayUnlock(*ppsa);
}

//  Managed client API to calculate the Julia set data only
extern "C" DLL_API void calculateJulia2(int gpuIndex, double re, double im, bool gpu, bool cuda, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY * *ppsa)
{
    using namespace fractals;

    unsigned* result;
    SafeArrayLock(*ppsa);
    SafeArrayAccessData(*ppsa, (void HUGEP**) & result);

    if (gpu)
    {
        if (cuda)
            kernel_cuda::julia_kernel(gpuIndex, width, height, xMin, xMax, yMin, yMax, re, im, maxIterations, result);
    }
    else
        kernel_cpu::julia_kernel(width, height, xMin, xMax, yMin, yMax, re, im, maxIterations, result);

    SafeArrayUnaccessData(*ppsa);
    SafeArrayUnlock(*ppsa);
}

extern "C" DLL_API void calculateSpecial(int gpuIndex, int func, double re, double im, bool gpu, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY * *ppsa)
{
    (void)gpu;
    (void)gpuIndex;
    const unsigned array_size = width * height;

    SAFEARRAYBOUND rgsa;
    rgsa.lLbound = 0;
    rgsa.cElements = array_size;
    *ppsa = SafeArrayCreate(VT_I4, 1, &rgsa);

    unsigned* result;
    SafeArrayLock(*ppsa);
    SafeArrayAccessData(*ppsa, (void HUGEP**) & result);

    fractals::MandelbrotSet::cpuSpecialKernel(func, fractals::complex(re, im), width, height, xMin, xMax, yMin, yMax, maxIterations, result);

    SafeArrayUnaccessData(*ppsa);
    SafeArrayUnlock(*ppsa);
}

/*
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

    fractals::MandelbrotSet::gpuCalculationDensity(gpu_accelerator(gpuIndex), antiBuddha, width, height, xMin, xMax, yMin, yMax, maxIterations, result);

    SafeArrayUnaccessData(*ppsa);
    SafeArrayUnlock(*ppsa);
}
*/

//  Managed client API to transform data according to an input palette.  
//  The operation is ppsaResult[i] = palette[input[i]].
extern "C" void paletteTransform(int gpuIndex, SAFEARRAY* input, SAFEARRAY* palette, SAFEARRAY** ppsaResult)
{
    (void)gpuIndex;

    SafeArrayLock(input);
    unsigned array_size = input->rgsabound->cElements;
    unsigned* input_data;
    SafeArrayAccessData(input, (void HUGEP**)&input_data);

    SafeArrayLock(palette);
    //unsigned palette_size = palette->rgsabound->cElements;
    unsigned* palette_data;
    SafeArrayAccessData(palette, (void HUGEP**)&palette_data);

    SAFEARRAYBOUND rgsa;
    rgsa.lLbound = 0;
    rgsa.cElements = array_size;
    *ppsaResult = SafeArrayCreate(VT_I4, 1, &rgsa);

    unsigned* result;
    SafeArrayLock(*ppsaResult);
    SafeArrayAccessData(*ppsaResult, (void HUGEP**) &result);

    //fractals::MandelbrotSet::gpuPaletteKernel(gpu_accelerator(gpuIndex), array_size, input_data, result, palette_size, (fractals::rgb*)palette_data);

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
    (void)gpuIndex;
    SafeArrayLock(input);
    //unsigned array_size = input->rgsabound->cElements;
    unsigned* input_data;
    SafeArrayAccessData(input, (void HUGEP**)&input_data);

    SafeArrayLock(palette);
    //unsigned palette_size = palette->rgsabound->cElements;
    unsigned* palette_data;
    SafeArrayAccessData(palette, (void HUGEP**)&palette_data);

    unsigned* result;
    SafeArrayLock(*ppsaResult);
    SafeArrayAccessData(*ppsaResult, (void HUGEP**) &result);

    //fractals::MandelbrotSet::gpuPaletteKernel(gpu_accelerator(gpuIndex), array_size, input_data, result, palette_size, (fractals::rgb*)palette_data);

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

