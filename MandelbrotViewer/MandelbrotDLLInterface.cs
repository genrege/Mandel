﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace MandelbrotViewer
{
    class MandelbrotDLLInterface
    {
        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void render(bool gpu, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void renderJulia(double re, double im, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void renderBuddha(IntPtr hdc, bool antiBuddha, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void saveMandelbrotBitmap(IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void saveMandelbrotJPG(IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void saveJuliaBitmap(double re, double im, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void saveJuliaJPG(double re, double im, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void saveBuddhaBitmap(IntPtr hdc, bool antiBuddha, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void saveBuddhaJPG(IntPtr hdc, bool antiBuddha, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void calculateMandelbrot(bool gpu, int width, int height, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void calculateJulia(double re, double im, int width, int height, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void paletteTransform([MarshalAs(UnmanagedType.SafeArray)] int[] input, [MarshalAs(UnmanagedType.SafeArray)] int[] palette, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void renderArrayToDevice(IntPtr hdc, int width, int height, [MarshalAs(UnmanagedType.SafeArray)] int[] input);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void renderArrayToBitmap(IntPtr hdc, int width, int height, [MarshalAs(UnmanagedType.SafeArray)] int[] input, [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        public static extern void renderArrayToJPEG(IntPtr hdc, int width, int height, [MarshalAs(UnmanagedType.SafeArray)] int[] input, [MarshalAs(UnmanagedType.LPStr)] string filename);
    }
}
