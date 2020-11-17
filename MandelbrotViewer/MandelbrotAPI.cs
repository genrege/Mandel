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
        public static void RenderBasic(IntPtr hdc, bool gpu, int maxIterations, CoordinateSpace cspace)
        {
            MandelbrotDLLInterface.render(hdc, gpu, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax);
        }

        public static void RenderJulia(IntPtr hdc, bool gpu, int maxIterations, double re, double im, CoordinateSpace cspace)
        {
            MandelbrotDLLInterface.renderJulia(hdc, gpu, maxIterations, re, im, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax);
        }

        public static void RenderBuddha(IntPtr hdc, int maxIterations, CoordinateSpace cspace)
        {
            MandelbrotDLLInterface.renderBuddha(hdc, false, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax);
        }

        public static void RenderAntiBuddha(IntPtr hdc, int maxIterations, CoordinateSpace cspace)
        {
            MandelbrotDLLInterface.renderBuddha(hdc, true, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax);
        }

        public static void SaveBitmapToFile(IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveMandelbrotBitmap(hdc, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename);
        }

        public static void SaveJPGToFile(IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveMandelbrotJPG(hdc, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename);
        }

        public static void SaveJuliaBitmapToFile(double re, double im, IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveJuliaBitmap(re, im, hdc, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename);
        }

        public static void SaveJuliaJPGToFile(double re, double im, IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveJuliaJPG(re, im, hdc, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename);
        }

        public static void SaveBuddhaBitmapToFile(IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveBuddhaBitmap(hdc, false, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename);
        }

        public static void SaveAntiBuddhaBitmapToFile(IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveBuddhaBitmap(hdc, true, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename);
        }

        public static void SaveBuddhaJPGToFile(IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveBuddhaJPG(hdc, false, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename);
        }

        public static void SaveAntiBuddhaJPGToFile(IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveBuddhaJPG(hdc, true, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename); 
        }

        public static int[] CalculateMandelbrot(bool gpu, int maxIterations, CoordinateSpace cspace)
        {
            int[] result;

            MandelbrotDLLInterface.calculateMandelbrot(gpu, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, out result);

            return result;
        }

        public static int[] CalculateJulia(double re, double im, bool gpu, int maxIterations, CoordinateSpace cspace)
        {
            int[] result;

            MandelbrotDLLInterface.calculateJulia(re, im, gpu, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, out result);

            return result;
        }

        public static int[] CalculateJulia2(double re, double im, bool gpu, int maxIterations, CoordinateSpace cspace, ref int[] result)
        {
            MandelbrotDLLInterface.calculateJulia2(re, im, gpu, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, ref result);

            return result;
        }

        public static int[] CalculateSpecial(int func, double re, double im, bool gpu, int maxIterations, CoordinateSpace cspace)
        {
            int[] result;

            MandelbrotDLLInterface.calculateSpecial(func, re, im, gpu, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, out result);

            return result;
        }

        public static int[] CalculateBuddha(bool antiBuddha, int maxIterations, CoordinateSpace cspace)
        {
            int[] result;

            MandelbrotDLLInterface.calculateBuddha(antiBuddha, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, out result);

            return result;
        }

        public static int[] PaletteTransform(int[] input, int[] palette)
        {
            int[] result;

            MandelbrotDLLInterface.paletteTransform(input, palette, out result);

            return result;
        }

        public static void PaletteTransform2(int[] input, int[] palette, ref int[] result)
        {
            MandelbrotDLLInterface.paletteTransform2(input, palette, ref result);
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

        public static int[] BuddhaPalette(int max_iterations)
        {
            int[] palette = new int[max_iterations];

            for (int i = 0; i < max_iterations; ++i)
            {
                double s1 = (double)(i) * 10.0 / (double)max_iterations;
                double s2 = (double)(i) * 10.0 / (double)max_iterations;
                double s3 = (double)(i) * 10.0 / (double)max_iterations;

                double f = Math.Min(1, 10 * (1 - Math.Pow(s1 - 1, 2)));
                double g = Math.Min(1, 10 * (1 - Math.Pow(s2 - 1, 2)));
                double h = Math.Min(1, 10 * (1 - Math.Pow(s3 - 1, 10)));

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
