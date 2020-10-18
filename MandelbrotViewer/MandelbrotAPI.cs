using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

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
    }
}
