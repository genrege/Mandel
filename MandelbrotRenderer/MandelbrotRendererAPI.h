#pragma once
#include <Windows.h>

#ifdef MANDELBROTRENDERER_EXPORTS
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif

extern "C" DLL_API void render(bool gpu, HDC hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax);
extern "C" DLL_API void renderJulia(double re, double im, HDC hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax);
extern "C" DLL_API void renderBuddha(HDC hdc, bool antiBuddha, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax);
extern "C" DLL_API void saveMandelbrotBitmap(HDC hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, const char* filename);
extern "C" DLL_API void saveJuliaBitmap(double re, double im, HDC hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, const char* filename);
extern "C" DLL_API void saveBuddhaBitmap(HDC hdc, bool antiBuddha, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, const char* filename);
extern "C" DLL_API void saveMandelbrotJPG(HDC hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, const char* filename);
extern "C" DLL_API void saveJuliaJPG(double re, double im, HDC hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, const char* filename);
extern "C" DLL_API void saveBuddhaJPG(HDC hdc, bool antiBuddha, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, const char* filename);
