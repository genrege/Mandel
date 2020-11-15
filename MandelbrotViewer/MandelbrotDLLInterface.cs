using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace MandelbrotViewer
{
    class MandelbrotDLLInterface
    {

        [DllImport(@"MandelbrotRenderer.dll")] public static extern void render(IntPtr hdc, bool gpu, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void renderJulia(IntPtr hdc, bool gpu, int maxIterations, double re, double im, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void renderBuddha(IntPtr hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void saveMandelbrotBitmap(IntPtr hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void saveMandelbrotJPG(IntPtr hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void saveJuliaBitmap(double re, double im, IntPtr hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void saveJuliaJPG(double re, double im, IntPtr hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void saveBuddhaBitmap(IntPtr hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void saveBuddhaJPG(IntPtr hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);

        [DllImport(@"MandelbrotRenderer.dll")] public static extern void calculateMandelbrot(bool gpu, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void calculateJulia(double re, double im, bool gpu, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void calculateSpecial(int func, double re, double im, bool gpu, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void calculateBuddha(bool antiBuddha, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void paletteTransform([MarshalAs(UnmanagedType.SafeArray)] int[] input, [MarshalAs(UnmanagedType.SafeArray)] int[] palette, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void renderArrayToDevice(IntPtr hdc, int width, int height, [MarshalAs(UnmanagedType.SafeArray)] int[] input);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void renderArrayToBitmap(IntPtr hdc, int width, int height, [MarshalAs(UnmanagedType.SafeArray)] int[] input, [MarshalAs(UnmanagedType.LPStr)] string filename);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void renderArrayToJPEG(IntPtr hdc, int width, int height, [MarshalAs(UnmanagedType.SafeArray)] int[] input, [MarshalAs(UnmanagedType.LPStr)] string filename);
    }
}
