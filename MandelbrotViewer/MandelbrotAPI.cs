using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Drawing;

namespace MandelbrotViewer
{
    public class MandelbrotAPI
    {
        public static void RenderBasic(bool gpu, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax)
        {
            MandelbrotDLLInterface.render(gpu, hdc, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax);
        }
        public static void RenderJulia(double re, double im, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax)
        {
            MandelbrotDLLInterface.renderJulia(re, im, hdc, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax);
        }

        public static void RenderBuddha(IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax)
        {
            MandelbrotDLLInterface.renderBuddha(hdc, false, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax);
        }

        public static void RenderAntiBuddha(IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax)
        {
            MandelbrotDLLInterface.renderBuddha(hdc, true, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax);
        }

        public static void SaveBitmapToFile(IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, string filename)
        {
            MandelbrotDLLInterface.saveMandelbrotBitmap(hdc, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax, filename);
        }

        public static void SaveJPGToFile(IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, string filename)
        {
            MandelbrotDLLInterface.saveMandelbrotJPG(hdc, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax, filename);
        }

        public static void SaveJuliaBitmapToFile(double re, double im, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, string filename)
        {
            MandelbrotDLLInterface.saveJuliaBitmap(re, im, hdc, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax, filename);
        }

        public static void SaveJuliaJPGToFile(double re, double im, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, string filename)
        {
            MandelbrotDLLInterface.saveJuliaJPG(re, im, hdc, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax, filename);
        }

        public static void SaveBuddhaBitmapToFile(IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, string filename)
        {
            MandelbrotDLLInterface.saveBuddhaBitmap(hdc, false, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax, filename);
        }

        public static void SaveAntiBuddhaBitmapToFile(IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, string filename)
        {
            MandelbrotDLLInterface.saveBuddhaBitmap(hdc, true, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax, filename);
        }

        public static void SaveBuddhaJPGToFile(IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, string filename)
        {
            MandelbrotDLLInterface.saveBuddhaJPG(hdc, false, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax, filename);
        }

        public static void SaveAntiBuddhaJPGToFile(IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, string filename)
        {
            MandelbrotDLLInterface.saveBuddhaJPG(hdc, true, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax, filename);
        }

        public static int[] CalculateMandelbrot(bool gpu, int width, int height, int maxIterations, double xMin, double xMax, double yMin, double yMax)
        {
            int[] result;

            MandelbrotDLLInterface.calculateMandelbrot(gpu, width, height, maxIterations, xMin, xMax, yMin, yMax, out result);

            return result;
        }

        public static int[] CalculateJulia(double re, double im, int width, int height, int maxIterations, double xMin, double xMax, double yMin, double yMax)
        {
            int[] result;

            MandelbrotDLLInterface.calculateJulia(re, im, width, height, maxIterations, xMin, xMax, yMin, yMax, out result);

            return result;
        }

        public static int[] PaletteTransform(int[] input, int[] palette)
        {
            int[] result;

            MandelbrotDLLInterface.paletteTransform(input, palette, out result);

            return result;
        }

        public static int[] StandardPalette(int max_iterations)
        {
            int[] palette = new int[max_iterations];

            for (int i = 0; i < max_iterations; ++i)
            {
                double s1 = (double)(i) * 1.0 / (double)max_iterations;
                double s2 = (double)(i) * 2.0 / (double)max_iterations;
                double s3 = (double)(i) * 3.0 / (double)max_iterations;

                double f = Math.Abs(s1 - Math.Pow(s1 - 1, 8));
                double g = Math.Abs(s2 - Math.Pow(s2 - 1, 4));
                double h = Math.Abs(s3 - Math.Pow(s3 - 1, 2));

                palette[i] = ((int)(255.0 * f) << 16) | ((int)(255.0 * g) << 8) | (int)(255.0 * h);
            }

            return palette;
        }

        public static void RenderArrayToDevice(IntPtr hdc, int width, int height, [MarshalAs(UnmanagedType.SafeArray)] int[] input)
        {
            MandelbrotDLLInterface.renderArrayToDevice(hdc, width, height, input);
        }

        public static void RenderArrayToBitmap(IntPtr hdc, int width, int height, [MarshalAs(UnmanagedType.SafeArray)] int[] input, [MarshalAs(UnmanagedType.LPStr)] string filename)
        {
            MandelbrotDLLInterface.renderArrayToBitmap(hdc, width, height, input, filename);
        }

        public static void RenderArrayToJPEG(IntPtr hdc, int width, int height, [MarshalAs(UnmanagedType.SafeArray)] int[] input, [MarshalAs(UnmanagedType.LPStr)] string filename)
        {
            MandelbrotDLLInterface.renderArrayToJPEG(hdc, width, height, input, filename);
        }
    }
}
