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

        [DllImport(@"MandelbrotRenderer.dll")] public static extern void GPU([MarshalAs(UnmanagedType.SafeArray)] ref string[] gpus);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void renderMandelbrot(int gpuIndex, IntPtr hdc, bool gpu, bool cuda, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, int palette_offset);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void renderJulia(int gpuIndex, IntPtr hdc, bool gpu, bool cuda, int maxIterations, double re, double im, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, int palette_offset);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void renderBuddha(int gpuIndex, IntPtr hdc, bool cuda, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, int palette_offset);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void saveMandelbrotBitmap(int gpuIndex, IntPtr hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, int palette_offset, [MarshalAs(UnmanagedType.LPStr)] string lpString);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void saveMandelbrotJPG(int gpuIndex, IntPtr hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, int palette_offset, [MarshalAs(UnmanagedType.LPStr)] string lpString);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void saveJuliaBitmap(int gpuIndex, double re, double im, IntPtr hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, int palette_offset, [MarshalAs(UnmanagedType.LPStr)] string lpString);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void saveJuliaJPG(int gpuIndex, double re, double im, IntPtr hdc, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, int palette_offset, [MarshalAs(UnmanagedType.LPStr)] string lpString);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void saveBuddhaBitmap(int gpuIndex, IntPtr hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void saveBuddhaJPG(int gpuIndex, IntPtr hdc, bool antiBuddha, int maxIterations, int screenWidth, int screenHeight, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);

        [DllImport(@"MandelbrotRenderer.dll")] public static extern void calculateMandelbrot(int gpuIndex, bool gpu, bool cuda, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void calculateJulia(int gpuIndex, double re, double im, bool gpu, bool cuda, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void calculateJulia2(int gpuIndex, double re, double im, bool gpu, bool cuda, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.SafeArray)] ref int[] result);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void calculateSpecial(int gpuIndex, int func, double re, double im, bool gpu, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void calculateBuddha(int gpuIndex, bool antiBuddha, int maxIterations, int width, int height, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void paletteTransform(int gpuIndex, [MarshalAs(UnmanagedType.SafeArray)] int[] input, [MarshalAs(UnmanagedType.SafeArray)] int[] palette, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void paletteTransform2(int gpuIndex, [MarshalAs(UnmanagedType.SafeArray)] int[] input, [MarshalAs(UnmanagedType.SafeArray)] int[] palette, [MarshalAs(UnmanagedType.SafeArray)] ref int[] result);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void renderArrayToDisplay(IntPtr hdc, int width, int height, [MarshalAs(UnmanagedType.SafeArray)] int[] input);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void renderArrayToBitmap(IntPtr hdc, int width, int height, [MarshalAs(UnmanagedType.SafeArray)] int[] input, [MarshalAs(UnmanagedType.LPStr)] string filename);
        [DllImport(@"MandelbrotRenderer.dll")] public static extern void renderArrayToJPEG(IntPtr hdc, int width, int height, [MarshalAs(UnmanagedType.SafeArray)] int[] input, [MarshalAs(UnmanagedType.LPStr)] string filename);
    }
}
