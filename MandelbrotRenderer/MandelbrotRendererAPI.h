//This API is for C# clients.  C++ clients can use the raw functions in MandelbrotSet.h for max calculation performance.  Rendering can still be done
//with the functions here for convenience without any loss of performance.

#pragma once
#include <Windows.h>

#ifdef MANDELBROTRENDERER_EXPORTS
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif

extern "C" struct palette
{
    SAFEARRAY* palette_;
};

//These functions can be used by any client.
extern "C" DLL_API void GPU(SAFEARRAY** ppsa);
extern "C" DLL_API void renderMandelbrot(int gpuIndex, HDC hdc, bool gpu, bool cuda, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset);
extern "C" DLL_API void renderJulia(int gpuIndex, HDC hdc, bool gpu, bool cuda, int maxIterations, double re, double im, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset);
extern "C" DLL_API void renderBuddha(int gpuIndex, HDC hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax);
extern "C" DLL_API void saveMandelbrotBitmap(int gpuIndex, HDC hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename);
extern "C" DLL_API void saveJuliaBitmap(int gpuIndex, double re, double im, HDC hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename);
extern "C" DLL_API void saveBuddhaBitmap(int gpuIndex, HDC hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, const char* filename);
extern "C" DLL_API void saveMandelbrotJPG(int gpuIndex, HDC hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename);
extern "C" DLL_API void saveJuliaJPG(int gpuIndex, double re, double im, HDC hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, unsigned palette_offset, const char* filename);
extern "C" DLL_API void saveBuddhaJPG(int gpuIndex, HDC hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, const char* filename);

//These functions are strictly for managed clients or those who want to use them but don't worry too much about performance and the horrible safearrays.
extern "C" DLL_API void calculateMandelbrot(int gpuIndex, bool gpu, bool cuda, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY * *ppsa);
extern "C" DLL_API void calculateJulia(int gpuIndex, double re, double im, bool gpu, bool cuda, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY * *ppsa);
extern "C" DLL_API void calculateJulia2(int gpuIndex, double re, double im, bool gpu, bool cuda, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY * *ppsa);
extern "C" DLL_API void calculateSpecial(int gpuIndex, int func, double re, double im, bool gpu, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY * *ppsa);
extern "C" DLL_API void calculateBuddha(int gpuIndex, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, SAFEARRAY * *ppsa);
extern "C" DLL_API void paletteTransform(int gpuIndex, SAFEARRAY * input, SAFEARRAY * palette, SAFEARRAY * *ppsa);
extern "C" DLL_API void paletteTransform2(int gpuIndex, SAFEARRAY * input, SAFEARRAY * palette, SAFEARRAY * *ppsa);
extern "C" DLL_API void renderArrayToDisplay(HDC hdc, int width, int height, SAFEARRAY* input);
extern "C" DLL_API void renderArrayToBitmap(HDC hdc, int width, int height, SAFEARRAY * input, const char* filename);
extern "C" DLL_API void renderArrayToJPEG(HDC hdc, int width, int height, SAFEARRAY * input, const char* filename);
