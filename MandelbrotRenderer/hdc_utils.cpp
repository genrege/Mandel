#define _CRT_SECURE_NO_WARNINGS

#include "hdc_utils.h"

template <class F>
void sendImage(HDC hdc, int screenWidth, int screenHeight, const unsigned* bitmapData, F f)
{
    HDC hmemDC = CreateCompatibleDC(hdc);
    HBITMAP bmp = CreateCompatibleBitmap(hdc, screenWidth, screenHeight);
    SelectObject(hmemDC, bmp);

    BITMAP bmpBuffer;
    GetObject(bmp, sizeof(BITMAP), &bmpBuffer);

    const auto& bi = createBitmapInfoHeader(bmpBuffer.bmWidth, bmpBuffer.bmHeight, 32);
    SetDIBits(hmemDC, bmp, 0, screenHeight, bitmapData, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    f(hmemDC, bmp, bi);

    DeleteObject(bmp);
    DeleteDC(hmemDC);
}

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

void sendToBitmap(HDC hdc, int screenWidth, int screenHeight, const unsigned* data, const char* filename)
{
    sendImage(hdc, screenWidth, screenHeight, data, [&](auto hmemDC, auto bmp, const auto& bi)
        {
            BITMAPINFO binf;
            memcpy(&binf.bmiHeader, &bi, sizeof(bi));
            const auto& bitmapInfo = CreateBitmapInfoStruct(0, bmp);
            CreateBMPFile(filename, bitmapInfo, bmp, hmemDC);
        });
}

void sendToJPEG(HDC hdc, int screenWidth, int screenHeight, const unsigned* data, const char* filename)
{
    sendImage(hdc, screenWidth, screenHeight, data, [&](auto, auto bmp, const auto&) {CreateJPEGFile(filename, bmp); });
}

void sendToDisplay(HDC hdc, int screenWidth, int screenHeight, const unsigned* data)
{
    sendImage(hdc, screenWidth, screenHeight, data, [&](auto hmemDC, auto, const auto&) {BitBlt(hdc, 0, 0, screenWidth, screenHeight, hmemDC, 0, 0, SRCCOPY); });
}

