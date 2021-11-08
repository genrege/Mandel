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
        public static void GPU([MarshalAs(UnmanagedType.SafeArray)] string[] gpus)
        {
            MandelbrotDLLInterface.GPU(ref gpus);
        }

        public static void RenderBasic(IntPtr hdc, bool gpu, int gpuIndex, int maxIterations, CoordinateSpace cspace)
        {
            MandelbrotDLLInterface.render(gpuIndex, hdc, gpu, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax);
        }

        public static void RenderJulia(int gpuIndex, IntPtr hdc, bool gpu, int maxIterations, double re, double im, CoordinateSpace cspace)
        {
            MandelbrotDLLInterface.renderJulia(gpuIndex, hdc, gpu, maxIterations, re, im, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax);
        }

        public static void RenderBuddha(int gpuIndex, IntPtr hdc, int maxIterations, CoordinateSpace cspace)
        {
            MandelbrotDLLInterface.renderBuddha(gpuIndex, hdc, false, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax);
        }

        public static void RenderAntiBuddha(int gpuIndex, IntPtr hdc, int maxIterations, CoordinateSpace cspace)
        {
            MandelbrotDLLInterface.renderBuddha(gpuIndex, hdc, true, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax);
        }

        public static void SaveBitmapToFile(int gpuIndex, IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveMandelbrotBitmap(gpuIndex, hdc, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename);
        }

        public static void SaveJPGToFile(int gpuIndex, IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveMandelbrotJPG(gpuIndex, hdc, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename);
        }

        public static void SaveJuliaBitmapToFile(int gpuIndex, double re, double im, IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveJuliaBitmap(gpuIndex, re, im, hdc, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename);
        }

        public static void SaveJuliaJPGToFile(int gpuIndex, double re, double im, IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveJuliaJPG(gpuIndex, re, im, hdc, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename);
        }

        public static void SaveBuddhaBitmapToFile(int gpuIndex, IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveBuddhaBitmap(gpuIndex, hdc, false, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename);
        }

        public static void SaveAntiBuddhaBitmapToFile(int gpuIndex, IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveBuddhaBitmap(gpuIndex, hdc, true, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename);
        }

        public static void SaveBuddhaJPGToFile(int gpuIndex, IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveBuddhaJPG(gpuIndex, hdc, false, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename);
        }

        public static void SaveAntiBuddhaJPGToFile(int gpuIndex, IntPtr hdc, int maxIterations, CoordinateSpace cspace, string filename)
        {
            MandelbrotDLLInterface.saveBuddhaJPG(gpuIndex, hdc, true, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, filename); 
        }

        public static int[] CalculateMandelbrot(int gpuIndex, bool gpu, int maxIterations, CoordinateSpace cspace)
        {
            int[] result;

            MandelbrotDLLInterface.calculateMandelbrot(gpuIndex, gpu, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, out result);

            return result;
        }

        public static int[] CalculateJulia(int gpuIndex, double re, double im, bool gpu, int maxIterations, CoordinateSpace cspace)
        {
            int[] result;

            MandelbrotDLLInterface.calculateJulia(gpuIndex, re, im, gpu, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, out result);

            return result;
        }

        public static int[] CalculateJulia2(int gpuIndex, double re, double im, bool gpu, int maxIterations, CoordinateSpace cspace, ref int[] result)
        {
            MandelbrotDLLInterface.calculateJulia2(gpuIndex, re, im, gpu, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, ref result);

            return result;
        }

        public static int[] CalculateSpecial(int gpuIndex, int func, double re, double im, bool gpu, int maxIterations, CoordinateSpace cspace)
        {
            int[] result;

            MandelbrotDLLInterface.calculateSpecial(gpuIndex, func, re, im, gpu, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, out result);

            return result;
        }

        public static int[] CalculateBuddha(int gpuIndex, bool antiBuddha, int maxIterations, CoordinateSpace cspace)
        {
            int[] result;

            MandelbrotDLLInterface.calculateBuddha(gpuIndex, antiBuddha, maxIterations, cspace.ScreenWidth, cspace.ScreenHeight, cspace.XMin, cspace.XMax, cspace.YMin, cspace.YMax, out result);

            return result;
        }

        public static int[] PaletteTransform(int gpuIndex, int[] input, int[] palette)
        {
            int[] result;

            MandelbrotDLLInterface.paletteTransform(gpuIndex, input, palette, out result);

            return result;
        }

        public static void PaletteTransform2(int gpuIndex, int[] input, int[] palette, ref int[] result)
        {
            MandelbrotDLLInterface.paletteTransform2(gpuIndex, input, palette, ref result);
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
