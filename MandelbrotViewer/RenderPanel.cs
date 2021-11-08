using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.Remoting.Messaging;
using System.Xml;
using System.Xml.Schema;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace MandelbrotViewer
{
    public partial class RenderPanel : Panel
    {
        public event EventHandler StatusChange;
        public event EventHandler PositionChange;

        public RenderPanel()
        {
            InitializeComponent();
            MaxIterations = 1024;
            this.MouseWheel += RenderPanel_OnMouseWheel;
            coord_ = new CoordinateSpace(Width, Height, -3.0, -2.0, 2.0);
            coord_.Align(-1.0, 0.0, Width / 2, Height / 2);
        }

        CoordinateSpace coord_ = null;

        public int gpuIndex { get; set; }
        public double MouseDownSetX {get; set;}
        public double MouseDownSetY { get; set;}
        public double CurrentSetX { get; set; }
        public double CurrentSetY { get; set; }
        public double JuliaSetX { get; set; }
        public double JuliaSetY { get; set; }
        public int SpecialFunc { get; set; }

        public int MaxIterations { get; set; }
        public int FractalSetIndex { get; set; }

        public bool useGpu { get; set; }
        int[] palette_ = null;
        int[] calculation_data_ = null;
        int[] bitmap_ = null;

        public CoordinateSpace coordinateSpace()
        {
            return coord_;
        }

        protected override void OnPaintBackground(PaintEventArgs e)
        {
        }

        public void setJulia(double x, double y)
        {
            JuliaSetX = x;
            JuliaSetY = y;
            if (FractalSetIndex == 1 || FractalSetIndex == 5 || FractalSetIndex == 8)
                Render();
        }

        private void RenderPanel_MouseUp(object sender, MouseEventArgs e)
        {
            Capture = false;
        }

        private void RenderPanel_Resize(object sender, EventArgs e)
        {
            coord_.ScreenWidth = Width;
            coord_.ScreenHeight = Height;
            Invalidate();

            calculation_data_ = new int[Width * Height];
            bitmap_ = new int[Width * Height];
        }

        private void RenderPanel_MouseDown(object sender, MouseEventArgs e)
        {
            if (MouseButtons == MouseButtons.Left)
            {
                Capture = true;

                var cc = coord_.SetFromScreen(e.Location.X, e.Location.Y);

                MouseDownSetX = cc.X;
                MouseDownSetY = cc.Y;
            }
        }

        Point lastMouseMoveLocation = Point.Empty;
        private void RenderPanel_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Location == lastMouseMoveLocation)
                return;
            lastMouseMoveLocation = e.Location;

            var cc = coord_.SetFromScreen(e.Location.X, e.Location.Y);
            CurrentSetX = cc.X;
            CurrentSetY = cc.Y;

            if (MouseButtons == MouseButtons.Left && Control.ModifierKeys == Keys.Control)
            {
                setJulia(cc.X, cc.Y);
            }
            else if (MouseButtons == MouseButtons.Left && Capture)
            {
                double nx = MouseDownSetX - CurrentSetX;
                double ny = MouseDownSetY - CurrentSetY;

                coord_.ShiftSpace(nx, ny);

                EventHandler handler1 = PositionChange;
                if (handler1 != null)
                    handler1.Invoke(this, new SetScaleInfo(cc.X, cc.Y, coord_.XMin, coord_.XMax, coord_.YMin, coord_.YMax));

                Render();

            }

            EventHandler handler = StatusChange;
            if (handler != null)
            {
                handler.Invoke(this, new EventArgs());
            }
        }

        public void CentreOn(double px, double py)
        {
            coord_.Align(px, py, Width / 2, Height / 2);

            Render();

            EventHandler handler1 = PositionChange;
            if (handler1 != null)
            {
                var cc = coord_.SetFromScreen(Location.X, Location.Y);

                handler1.Invoke(this, new SetScaleInfo(cc.X, cc.Y, coord_.XMin, coord_.XMax, coord_.YMin, coord_.YMax));
            }
        }

        private void Zoom(double ratio, double px, double py)
        {
            double wx = coord_.XMax - coord_.XMin;
            double wy = coord_.YMax - coord_.YMin;

            coord_.XMin = -ratio * (px - coord_.XMin) + px;
            coord_.YMin = -ratio * (py - coord_.YMin) + py;
            coord_.YMax = coord_.YMin + wy * ratio;
        }

        private void RenderPanel_OnMouseWheel(object sender, MouseEventArgs e)
        {
            var cc = coord_.SetFromScreen(e.Location.X, e.Location.Y);
            double px = cc.X;
            double py = cc.Y;

            if (e.Delta > 0)
            {
                Zoom(1 / 1.1, px, py);
            }
            else
            {
                Zoom(1.1, px, py);
            }

            Render();

            EventHandler handler1 = PositionChange;
            if (handler1 != null)
                handler1.Invoke(this, new SetScaleInfo(cc.X, cc.Y, coord_.XMin, coord_.XMax, coord_.YMin, coord_.YMax));

        }

        private void Render()
        {
            var gr = CreateGraphics();
            var hdc = gr.GetHdc();

            switch (FractalSetIndex)
            {
                case 0:
                    MandelbrotAPI.RenderBasic(hdc, useGpu, gpuIndex, MaxIterations, coord_);
                    break;
                case 1:
                    MandelbrotAPI.RenderJulia(gpuIndex, hdc, useGpu, MaxIterations, JuliaSetX, JuliaSetY, coord_);
                    break;
                case 2:
                    MandelbrotAPI.RenderBuddha(gpuIndex, hdc, MaxIterations, coord_);
                    break;
                case 3:
                    MandelbrotAPI.RenderAntiBuddha(gpuIndex, hdc, MaxIterations, coord_);
                    break;
                case 4:
                    {
                        palette_ = MandelbrotAPI.StandardPalette(MaxIterations);
                        var calculation_data = MandelbrotAPI.CalculateMandelbrot(gpuIndex, useGpu, MaxIterations, coord_);
                        var bitmap = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette_);
                        MandelbrotAPI.RenderArrayToDevice(hdc, coord_.ScreenWidth, coord_.ScreenHeight, bitmap);
                    }
                    break;
                case 5:
                    {
                        palette_ = MandelbrotAPI.StandardPalette(MaxIterations);
                        MandelbrotAPI.CalculateJulia2(gpuIndex, JuliaSetX, JuliaSetY, useGpu, MaxIterations, coord_, ref calculation_data_);
                        MandelbrotAPI.PaletteTransform2(gpuIndex, calculation_data_, palette_, ref bitmap_);
                        MandelbrotAPI.RenderArrayToDevice(hdc, coord_.ScreenWidth, coord_.ScreenHeight, bitmap_);
                    }
                    break;
                case 6:
                    {
                        palette_ = MandelbrotAPI.BuddhaPalette(MaxIterations);
                        var calculation_data = MandelbrotAPI.CalculateBuddha(gpuIndex, false, MaxIterations, coord_);
                        var bitmap = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette_);
                        MandelbrotAPI.RenderArrayToDevice(hdc, coord_.ScreenWidth, coord_.ScreenHeight, bitmap);
                    }
                    break;
                case 7:
                    {
                        palette_ = MandelbrotAPI.BuddhaPalette(MaxIterations);
                        var calculation_data = MandelbrotAPI.CalculateBuddha(gpuIndex, true, MaxIterations, coord_);
                        var bitmap = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette_);
                        MandelbrotAPI.RenderArrayToDevice(hdc, coord_.ScreenWidth, coord_.ScreenHeight, bitmap);
                    }
                    break;
                case 8:
                    {
                        palette_ = MandelbrotAPI.StandardPalette(MaxIterations);
                        var calculation_data = MandelbrotAPI.CalculateSpecial(gpuIndex, SpecialFunc, JuliaSetX, JuliaSetY, useGpu, MaxIterations, coord_);
                        var bitmap = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette_);
                        MandelbrotAPI.RenderArrayToDevice(hdc, coord_.ScreenWidth, coord_.ScreenHeight, bitmap);
                    }
                    break;
            }

            gr.ReleaseHdc(hdc);
            gr.Dispose();
        }

        private void RenderPanel_Paint(object sender, PaintEventArgs e)
        {
            double aspectRatio = (double)Width / (double)Height;
            var hdc = e.Graphics.GetHdc();

            Render();

            EventHandler handler1 = PositionChange;
            if (handler1 != null)
            {
                var cc = coord_.SetFromScreen(Location.X, Location.Y);

                handler1.Invoke(this, new SetScaleInfo(cc.X, cc.Y, coord_.XMin, coord_.XMax, coord_.YMin, coord_.YMax));
            }
        }
    }

    public class SetScaleInfo : EventArgs
    {
        public SetScaleInfo(double x, double y, double xmin, double xmax, double ymin, double ymax)
        {
            X    = x;
            Y    = y;
            xMin = xmin;
            xMax = xmax;
            yMin = ymin;
            yMax = ymax;
        }
        public double X { get; set; }
        public double Y { get; set; }
        public double xMin { get; set; }
        public double xMax { get; set; }
        public double yMin { get; set; }
        public double yMax { get; set; }
    }
}

