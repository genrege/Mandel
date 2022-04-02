#pragma once

#include <atlsafe.h>
#include <atlimage.h>
#include <vector>

//This fn from stackoverflow
PBITMAPINFO CreateBitmapInfoStruct(HWND /*hwnd*/, HBITMAP hBmp);
void CreateBMPFile(const char* pszFile, PBITMAPINFO pbi, HBITMAP hBMP, HDC hDC);
void CreateJPEGFile(const char* filename, HBITMAP bmp);
BITMAPINFOHEADER createBitmapInfoHeader(unsigned width, unsigned height, WORD /*bit_count*/);
void sendToBitmap(HDC hdc, int screenWidth, int screenHeight, const unsigned* data, const char* filename);
void sendToJPEG(HDC hdc, int screenWidth, int screenHeight, const unsigned* data, const char* filename);
void sendToDisplay(HDC hdc, int screenWidth, int screenHeight, const unsigned* data);

