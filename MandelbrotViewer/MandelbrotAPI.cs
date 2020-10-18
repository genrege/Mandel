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
        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        private static extern void render(bool gpu, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        private static extern void renderJulia(double re, double im, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        private static extern void renderBuddha(IntPtr hdc, bool antiBuddha, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        private static extern void saveMandelbrotBitmap(IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        private static extern void saveMandelbrotJPG(IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        private static extern void saveJuliaBitmap(double re, double im, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        private static extern void saveJuliaJPG(double re, double im, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        private static extern void saveBuddhaBitmap(IntPtr hdc, bool antiBuddha, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        private static extern void saveBuddhaJPG(IntPtr hdc, bool antiBuddha, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.LPStr)] string lpString);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        private static extern void calculateMandelbrot(bool gpu, int width, int height, int maxIterations, double xMin, double xMax, double yMin, double yMax, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        private static extern void paletteTransform([MarshalAs(UnmanagedType.SafeArray)] int[] input, [MarshalAs(UnmanagedType.SafeArray)] int[] palette, [MarshalAs(UnmanagedType.SafeArray)] out int[] result);

        [DllImport(@"C:\git\Mandel\x64\Release\MandelbrotRenderer.dll")]
        private static extern void renderArrayToDevice(IntPtr hdc, int width, int height, [MarshalAs(UnmanagedType.SafeArray)] int[] input);




        public static void RenderBasic(bool gpu, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax)
        {
            render(gpu, hdc, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax);
        }
        public static void RenderJulia(double re, double im, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax)
        {
            renderJulia(re, im, hdc, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax);
        }

        public static void RenderBuddha(IntPtr hdc, bool antiBuddha, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax)
        {
            renderBuddha(hdc, antiBuddha, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax);
        }

        public static void SaveBitmapToFile(IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, string filename)
        {
            saveMandelbrotBitmap(hdc, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax, filename);
        }

        public static void SaveJPGToFile(IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, string filename)
        {
            saveMandelbrotJPG(hdc, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax, filename);
        }

        public static void SaveJuliaBitmapToFile(double re, double im, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, string filename)
        {
            saveJuliaBitmap(re, im, hdc, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax, filename);
        }

        public static void SaveJuliaJPGToFile(double re, double im, IntPtr hdc, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, string filename)
        {
            saveJuliaJPG(re, im, hdc, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax, filename);
        }

        public static void SaveBuddhaBitmapToFile(IntPtr hdc, bool antiBuddha, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, string filename)
        {
            saveBuddhaBitmap(hdc, antiBuddha, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax, filename);
        }

        public static void SaveBuddhaJPGToFile(IntPtr hdc, bool antiBuddha, int screenWidth, int screenHeight, int maxIterations, double xMin, double xMax, double yMin, double yMax, string filename)
        {
            saveBuddhaJPG(hdc, antiBuddha, screenWidth, screenHeight, maxIterations, xMin, xMax, yMin, yMax, filename);
        }

        public static int[] CalculateMandelbrot(bool gpu, int width, int height, int maxIterations, double xMin, double xMax, double yMin, double yMax)
        {
            int[] result;

            calculateMandelbrot(gpu, width, height, maxIterations, xMin, xMax, yMin, yMax, out result);

            return result;
        }

        public static int[] PaletteTransform(int[] input, int[] palette)
        {
            int[] result;

            paletteTransform(input, palette, out result);

            return result;
        }

        public static int[] StandardPalette(int max_iterations)
        {
            int[] palette = new int[max_iterations];

            for (int i = 0; i < max_iterations; ++i)
            {
                double s1 = (double)(i) * 1.0 / (double)max_iterations;
                double s2 = (double)(i) * 1.0 / (double)max_iterations;
                double s3 = (double)(i) * 1.0 / (double)max_iterations;

                double f = 1 - Math.Pow(s1 - 1, 2);
                double g = 1 - Math.Pow(s2 - 1, 2);
                double h = 1 - Math.Pow(s3 - 1, 2);

                palette[i] = ((int)(255.0 * f) << 16) | ((int)(255.0 * g) << 8) | (int)(255.0 * h);
            }

            return palette;
        }

        public static void RenderArrayToDevice(IntPtr hdc, int width, int height, [MarshalAs(UnmanagedType.SafeArray)] int[] input)
        {
            renderArrayToDevice(hdc, width, height, input);
        }
    }
}
