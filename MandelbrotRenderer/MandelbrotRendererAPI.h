//This API is for C# clients.  C++ clients can use the raw functions in MandelbrotSet.h for max calculation performance.  Rendering can still be done
//with the functions here for convenience without any loss of performance.

#pragma once
#include <Windows.h>

#ifdef MANDELBROTRENDERER_EXPORTS
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif

//These functions can be used by any client.
extern "C" DLL_API void render(HDC hdc, bool gpu, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax);
extern "C" DLL_API void renderJulia(HDC hdc, bool gpu, int maxIterations, double re, double im, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax);
extern "C" DLL_API void renderBuddha(HDC hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax);
extern "C" DLL_API void saveMandelbrotBitmap(HDC hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, const char* filename);
extern "C" DLL_API void saveJuliaBitmap(double re, double im, HDC hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, const char* filename);
extern "C" DLL_API void saveBuddhaBitmap(HDC hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, const char* filename);
extern "C" DLL_API void saveMandelbrotJPG(HDC hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, const char* filename);
extern "C" DLL_API void saveJuliaJPG(double re, double im, HDC hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, const char* filename);
extern "C" DLL_API void saveBuddhaJPG(HDC hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, const char* filename);

//These functions are strictly for managed clients or those who want to use them but don't worry too much about performance and the horrible safearrays.
extern "C" DLL_API void calculateMandelbrot(bool gpu, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY * *ppsa);
extern "C" DLL_API void calculateJulia(double re, double im, bool gpu, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, SAFEARRAY * *ppsa);
extern "C" DLL_API void paletteTransform(SAFEARRAY * input, SAFEARRAY * palette, SAFEARRAY * *ppsa);
extern "C" DLL_API void renderArrayToDevice(HDC hdc, int width, int height, SAFEARRAY* input);
extern "C" DLL_API void renderArrayToBitmap(HDC hdc, int width, int height, SAFEARRAY * input, const char* filename);
extern "C" DLL_API void renderArrayToJPEG(HDC hdc, int width, int height, SAFEARRAY * input, const char* filename);
