
#include "MandelbrotRendererAPI.h"

#include <atlsafe.h>
#include <atlimage.h>
#include "Complex.h"
#include "MandelbrotSet.h"
#include "MandelbrotSetCL.h"
#include "fractal.h"
#include "../MandelbrotRendererCUDA/MandelbrotSetCUDA.h"

MathsEx::MandelbrotSet<double> mset;
MathsEx::mandelbrot_set mandelbrotset;
MathsEx::julia_set juliaset;

MathsEx::iteration_palette default_palette;
cache_memory<MathsEx::rgb> display_bitmap;

namespace
{
    //This fn from stackoverflow
    PBITMAPINFO CreateBitmapInfoStruct(HWND /*hwnd*/, HBITMAP hBmp)
    {
        BITMAP bmp;
        PBITMAPINFO pbmi;

        // Retrieve the bitmap color format, width, and height.  
        GetObject(hBmp, sizeof(BITMAP), (LPSTR)&bmp);

        // Convert the color format to a count of bits.  
        auto cClrBits = bmp.bmPlanes * bmp.bmBitsPixel;
        if (cClrBits == 1)
            cClrBits = 1;
        else if (cClrBits <= 4)
            cClrBits = 4;
        else if (cClrBits <= 8)
            cClrBits = 8;
        else if (cClrBits <= 16)
            cClrBits = 16;
        else if (cClrBits <= 24)
            cClrBits = 24;
        else cClrBits = 32;

        // Allocate memory for the BITMAPINFO structure. (This structure  
        // contains a BITMAPINFOHEADER structure and an array of RGBQUAD  
        // data structures.)  

        const auto mag = 1 << cClrBits;
        if (cClrBits < 24)
            pbmi = (PBITMAPINFO)LocalAlloc(LPTR, sizeof(BITMAPINFOHEADER) + sizeof(RGBQUAD) * mag);
        else
            pbmi = (PBITMAPINFO)LocalAlloc(LPTR, sizeof(BITMAPINFOHEADER));

        // Initialize the fields in the BITMAPINFO structure.  

        pbmi->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        pbmi->bmiHeader.biWidth = bmp.bmWidth;
        pbmi->bmiHeader.biHeight = bmp.bmHeight;
        pbmi->bmiHeader.biPlanes = bmp.bmPlanes;
        pbmi->bmiHeader.biBitCount = bmp.bmBitsPixel;
        if (cClrBits < 24)
            pbmi->bmiHeader.biClrUsed = (1 << cClrBits);

        // If the bitmap is not compressed, set the BI_RGB flag.  
        pbmi->bmiHeader.biCompression = BI_RGB;

        // Compute the number of bytes in the array of color  
        // indices and store the result in biSizeImage.  
        // The width must be DWORD aligned unless the bitmap is RLE 
        // compressed. 
        pbmi->bmiHeader.biSizeImage = ((pbmi->bmiHeader.biWidth * cClrBits + 31) & ~31) / 8 * pbmi->bmiHeader.biHeight;
        
        // Set biClrImportant to 0, indicating that all of the  
        // device colors are important.  
        pbmi->bmiHeader.biClrImportant = 0;
        return pbmi;
    }

    //This fn from stackoverflow
    void CreateBMPFile(const char* pszFile, PBITMAPINFO pbi, HBITMAP hBMP, HDC hDC)
    {
        HANDLE hf;                 // file handle  
        BITMAPFILEHEADER hdr;       // bitmap file-header  
        PBITMAPINFOHEADER pbih;     // bitmap info-header  
        LPBYTE lpBits;              // memory pointer  
        DWORD dwTotal;              // total count of bytes  
        DWORD cb;                   // incremental count of bytes  
        BYTE* hp;                   // byte pointer  
        DWORD dwTmp;

        pbih = (PBITMAPINFOHEADER)pbi;
        lpBits = (LPBYTE)GlobalAlloc(GMEM_FIXED, pbih->biSizeImage);

        // Retrieve the color table (RGBQUAD array) and the bits  
        // (array of palette indices) from the DIB.  
        GetDIBits(hDC, hBMP, 0, (WORD)pbih->biHeight, lpBits, pbi, DIB_RGB_COLORS);

        // Create the .BMP file.  
        hf = CreateFileA(pszFile,
            GENERIC_READ | GENERIC_WRITE,
            (DWORD)0,
            NULL,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            (HANDLE)NULL);
        hdr.bfType = 0x4d42;        // 0x42 = "B" 0x4d = "M"  
        // Compute the size of the entire file.  
        hdr.bfSize = (DWORD)(sizeof(BITMAPFILEHEADER) +
            pbih->biSize + pbih->biClrUsed
            * sizeof(RGBQUAD) + pbih->biSizeImage);
        hdr.bfReserved1 = 0;
        hdr.bfReserved2 = 0;

        // Compute the offset to the array of color indices.  
        hdr.bfOffBits = (DWORD)sizeof(BITMAPFILEHEADER) +
            pbih->biSize + pbih->biClrUsed
            * sizeof(RGBQUAD);

        // Copy the BITMAPFILEHEADER into the .BMP file.  
        WriteFile(hf, (LPVOID)&hdr, sizeof(BITMAPFILEHEADER), (LPDWORD)&dwTmp, NULL);

        // Copy the BITMAPINFOHEADER and RGBQUAD array into the file.  
        WriteFile(hf, (LPVOID)pbih, sizeof(BITMAPINFOHEADER) + pbih->biClrUsed * sizeof(RGBQUAD), (LPDWORD)&dwTmp, (NULL));

        // Copy the array of color indices into the .BMP file.  
        dwTotal = cb = pbih->biSizeImage;
        hp = lpBits;
        WriteFile(hf, (LPSTR)hp, (int)cb, (LPDWORD)&dwTmp, NULL);

        // Close the .BMP file.  
        CloseHandle(hf);

        // Free memory.  
        GlobalFree((HGLOBAL)lpBits);
    }

    //This fn from stackoverflow
    void CreateJPEGFile(const char* filename, HBITMAP bmp)
    {
        std::vector<BYTE> buf;
        IStream* stream = NULL;
        CreateStreamOnHGlobal(0, TRUE, &stream);
        CImage image;
        ULARGE_INTEGER liSize;

        image.Attach(bmp);
        image.Save(stream, Gdiplus::ImageFormatJPEG);
        IStream_Size(stream, &liSize);
        DWORD len = liSize.LowPart;
        IStream_Reset(stream);
        buf.resize(len);
        IStream_Read(stream, &buf[0], len);
        stream->Release();

        FILE* fp = fopen(filename, "wb");
        if (fp)
        {
            fwrite(buf.data(), buf.size(), 1, fp);
            fclose(fp);
        }
    }

    BITMAPINFOHEADER createBitmapInfoHeader(unsigned width, unsigned height, WORD /*bit_count*/)
    {
        BITMAPINFOHEADER   bi;

        bi.biSize = sizeof(BITMAPINFOHEADER);
        bi.biWidth = width;
        bi.biHeight = height;
        bi.biPlanes = 1;
        bi.biBitCount = 32;
        bi.biCompression = BI_RGB;
        bi.biSizeImage = 0;
        bi.biXPelsPerMeter = 0;
        bi.biYPelsPerMeter = 0;
        bi.biClrUsed = 0;
        bi.biClrImportant = 0;

        return bi;
    }

    void saveToBitmap(HDC hdc, int screenWidth, int screenHeight, const unsigned* data, const char* filename)
    {
        HDC hmemDC = CreateCompatibleDC(hdc);
        HBITMAP bmp = CreateCompatibleBitmap(hdc, screenWidth, screenHeight);
        SelectObject(hmemDC, bmp);

        BITMAP bmpBuffer;
        GetObject(bmp, sizeof(BITMAP), &bmpBuffer);

        const auto& bi = createBitmapInfoHeader(bmpBuffer.bmWidth, bmpBuffer.bmHeight, 32);
        SetDIBits(hmemDC, bmp, 0, screenHeight, data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

        BITMAPINFO binf;
        memcpy(&binf.bmiHeader, &bi, sizeof(bi));

        const auto& bitmapInfo = CreateBitmapInfoStruct(0, bmp);
        CreateBMPFile(filename, bitmapInfo, bmp, hmemDC);

        DeleteObject(bmp);
        DeleteDC(hmemDC);
    }

    void saveToJPEG(HDC hdc, int screenWidth, int screenHeight, const unsigned* data, const char* filename)
    {
        HDC hmemDC = CreateCompatibleDC(hdc);
        HBITMAP bmp = CreateCompatibleBitmap(hdc, screenWidth, screenHeight);
        SelectObject(hmemDC, bmp);

        BITMAP bmpBuffer;
        GetObject(bmp, sizeof(BITMAP), &bmpBuffer);
        const auto& bi = createBitmapInfoHeader(bmpBuffer.bmWidth, bmpBuffer.bmHeight, 32);
        SetDIBits(hmemDC, bmp, 0, screenHeight, data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);
        CreateJPEGFile(filename, bmp);
        DeleteObject(bmp);
        DeleteDC(hmemDC);
    }

    void sendToDisplay(HDC hdc, int screenWidth, int screenHeight, const unsigned* bitmapData)
    {
        HDC hmemDC = CreateCompatibleDC(hdc);
        HBITMAP bmp = CreateCompatibleBitmap(hdc, screenWidth, screenHeight);
        SelectObject(hmemDC, bmp);

        BITMAP bmpBuffer;
        GetObject(bmp, sizeof(BITMAP), &bmpBuffer);

        const auto& bi = createBitmapInfoHeader(bmpBuffer.bmWidth, bmpBuffer.bmHeight, 32);
        SetDIBits(hmemDC, bmp, 0, screenHeight, bitmapData, (BITMAPINFO*)&bi, DIB_RGB_COLORS);
        BitBlt(hdc, 0, 0, screenWidth, screenHeight, hmemDC, 0, 0, SRCCOPY);

        DeleteObject(bmp);
        DeleteDC(hmemDC);
    }

    auto gpu_accelerator(int gpuIndex)
    {
        const auto& accls = accelerator::get_all();
        return accelerator(accls[gpuIndex]).default_view;
    }
}

extern "C" int countGPU()
{
    std::vector<accelerator> accls = accelerator::get_all();

    int count = 0;
    for (size_t i = 0; i < accls.size(); ++i)
    {
        const auto& accl = accls[i];

        const auto& desc = std::to_wstring(i) + L";" + accl.description;
        if (desc.find(L"NVIDIA") != std::string::npos)
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
    saveToBitmap(hdc, screenWidth, screenHeight, display_bitmap.access_as<unsigned int>(), filename);

}

extern "C" void saveJuliaBitmap(int gpuIndex, double re, double im, HDC hdc, int max_iterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename)
{
    juliaset.set_scale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    juliaset.calculate_set_amp(gpu_accelerator(gpuIndex), MathsEx::Complex<double>(re, im), max_iterations);
    display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
    default_palette.apply(max_iterations, juliaset.data(), display_bitmap, palette_offset);
    saveToBitmap(hdc, screenWidth, screenHeight, display_bitmap.access_as<unsigned int>(), filename);
}

extern "C" void saveBuddhaBitmap(int gpuIndex, HDC hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, const char* filename)
{
    mset.SetScale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    mset.CalculateBuddha(gpu_accelerator(gpuIndex), antiBuddha, maxIterations);
    saveToBitmap(hdc, screenWidth, screenHeight, mset.bmp(), filename);
}

extern "C" void saveMandelbrotJPG(int gpuIndex, HDC hdc, int max_iterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename)
{
    mandelbrotset.set_scale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    mandelbrotset.calculate_set_amp(gpu_accelerator(gpuIndex), max_iterations);
    display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
    default_palette.apply(max_iterations, mandelbrotset.data(), display_bitmap, palette_offset);
    saveToJPEG(hdc, screenWidth, screenHeight, display_bitmap.access_as<unsigned>(), filename);
}

extern "C" void saveJuliaJPG(int gpuIndex, double re, double im, HDC hdc, int max_iterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename)
{
    juliaset.set_scale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    juliaset.calculate_set_amp(gpu_accelerator(gpuIndex), MathsEx::Complex<double>(re, im), max_iterations);
    display_bitmap.allocate(sizeof(unsigned) * screenWidth * screenHeight);
    default_palette.apply(max_iterations, juliaset.data(), display_bitmap, palette_offset);
    saveToJPEG(hdc, screenWidth, screenHeight, mset.bmp(), filename);
}

extern "C" void saveBuddhaJPG(int gpuIndex, HDC hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, const char* filename)
{
    mset.SetScale(xMin, xMax, yMin, yMax, screenWidth, screenHeight);
    mset.CalculateBuddha(gpu_accelerator(gpuIndex), antiBuddha, maxIterations);
    saveToJPEG(hdc, screenWidth, screenHeight, mset.bmp(), filename);
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
        MathsEx::MandelbrotSet<double>::gpuSpecialKernel(gpu_accelerator(gpuIndex), func, MathsEx::Complex<double>(re, im), width, height, xMin, xMax, yMin, yMax, maxIterations, result);
    else
        MathsEx::MandelbrotSet<double>::cpuSpecialKernel(func, MathsEx::Complex<double>(re, im), width, height, xMin, xMax, yMin, yMax, maxIterations, result);

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

    MathsEx::MandelbrotSet<double>::gpuCalculationDensity(gpu_accelerator(gpuIndex), antiBuddha, width, height, xMin, xMax, yMin, yMax, maxIterations, result);

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

    MathsEx::MandelbrotSet<double>::gpuPaletteKernel(gpu_accelerator(gpuIndex), array_size, input_data, result, palette_size, (MathsEx::rgb*)palette_data);

    SafeArrayUnaccessData(*ppsaResult);
    SafeArrayUnlock(*ppsaResult);

    SafeArrayUnaccessData(palette);
    SafeArrayUnlock(palette);

    SafeArrayUnaccessData(input);
    SafeArrayUnlock(input);
}

//  Managed client API to render an array of data to the given device context
extern "C" void renderArrayToDevice(HDC hdc, int width, int height, SAFEARRAY* input)
{
    SafeArrayLock(input);
    unsigned* input_data;
    SafeArrayAccessData(input, (void HUGEP**) & input_data);

    sendToDisplay(hdc, width, height, input_data);

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

    MathsEx::MandelbrotSet<double>::gpuPaletteKernel(gpu_accelerator(gpuIndex), array_size, input_data, result, palette_size, (MathsEx::rgb*)palette_data);

    SafeArrayUnaccessData(*ppsaResult);
    SafeArrayUnlock(*ppsaResult);

    SafeArrayUnaccessData(palette);
    SafeArrayUnlock(palette);

    SafeArrayUnaccessData(input);
    SafeArrayUnlock(input);
}

//  Managed client API to render to a bitmap
extern "C" void renderArrayToBitmap(HDC hdc, int width, int height, SAFEARRAY * input, const char* filename)
{
    SafeArrayLock(input);
    unsigned* input_data;
    SafeArrayAccessData(input, (void HUGEP**) & input_data);

    saveToBitmap(hdc, width, height, input_data, filename);

    SafeArrayUnaccessData(input);
    SafeArrayUnlock(input);
}

//  Managed client API to render to a JPEG
extern "C" void renderArrayToJPEG(HDC hdc, int width, int height, SAFEARRAY * input, const char* filename)
{
    SafeArrayLock(input);
    unsigned* input_data;
    SafeArrayAccessData(input, (void HUGEP**) & input_data);

    saveToJPEG(hdc, width, height, input_data, filename);

    SafeArrayUnaccessData(input);
    SafeArrayUnlock(input);
}
