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

namespace MandelbrotViewer
{
    public partial class RenderPanel : Panel
    {
        public event EventHandler StatusChange;
        public event EventHandler PositionChange;

        CoordinateSpace coord_ = null;

        public CoordinateSpace coordinateSpace()
        {
            return coord_;
        }

        public int maxIterations { get; set; }

        double cx_ = 0.0;
        double cy_ = 0.0;

        public RenderPanel()
        {
            InitializeComponent();
            maxIterations = 1024;
            this.MouseWheel += RenderPanel_OnMouseWheel;
            coord_ = new CoordinateSpace(Width, Height, -3.0, -2.0, 2.0);
            coord_.Align(-1.0, 0.0, Width / 2, Height / 2);
        }

        private void RenderPanel_Load(object sender, EventArgs e)
        {
        }

        protected override void OnPaintBackground(PaintEventArgs e)
        {
        }

        private void RenderPanel_MouseClick(object sender, MouseEventArgs e)
        {
            var cc = coord_.SetFromScreen(e.Location.X, e.Location.Y);
        }

        private void RenderPanel_MouseDown(object sender, MouseEventArgs e)
        {
            if (MouseButtons == MouseButtons.Left)
            {
                Capture = true;

                var cc = coord_.SetFromScreen(e.Location.X, e.Location.Y);

                cx_ = cc.X;
                cy_ = cc.Y;
            }
        }

        private void RenderPanel_MouseUp(object sender, MouseEventArgs e)
        {
            Capture = false;
        }

        private void RenderPanel_MouseLeave(object sender, EventArgs e)
        {

        }

        private void RenderPanel_Resize(object sender, EventArgs e)
        {
            coord_.ScreenWidth = Width;
            coord_.ScreenHeight = Height;
            Invalidate();
        }

        double mx_ = 0.0;
        double my_ = 0.0;

        public double CtrlX { get; set; }
        public double CtrlY { get; set; }

        private void RenderPanel_MouseMove(object sender, MouseEventArgs e)
        {
            var cc = coord_.SetFromScreen(e.Location.X, e.Location.Y);

            mx_ = cc.X;
            my_ = cc.Y;

            if (MouseButtons == MouseButtons.Right)
            {
                CtrlX = mx_;
                CtrlY = my_;
                if (FractalSetIndex == 1 || FractalSetIndex == 5)
                    Render();
            }
            if (MouseButtons == MouseButtons.Left && Capture)
            {
                double nx = cx_ - mx_;
                double ny = cy_ - my_;

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

        public bool useGpu { get; set; }

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

        int[] palette_ = null;
        private void Render()
        {
            if (palette_ == null)
                palette_ = MandelbrotAPI.StandardPalette(maxIterations);

            switch (FractalSetIndex)
            {
                case 0:
                    MandelbrotAPI.RenderBasic(true, CreateGraphics().GetHdc(), maxIterations, coord_);
                    break;
                case 1:
                    MandelbrotAPI.RenderJulia(CtrlX, CtrlY, this.CreateGraphics().GetHdc(), maxIterations, coord_);
                    break;
                case 2:
                    MandelbrotAPI.RenderBuddha(this.CreateGraphics().GetHdc(), maxIterations, coord_);
                    break;
                case 3:
                    MandelbrotAPI.RenderAntiBuddha(this.CreateGraphics().GetHdc(), maxIterations, coord_);
                    break;
                case 4:
                    {
                        var calculation_data = MandelbrotAPI.CalculateMandelbrot(true, maxIterations, coord_);
                        var bitmap = MandelbrotAPI.PaletteTransform(calculation_data, palette_);
                        MandelbrotAPI.RenderArrayToDevice(this.CreateGraphics().GetHdc(), coord_.ScreenWidth, coord_.ScreenHeight, bitmap);
                    }
                    break;
                case 5:
                    {
                        var calculation_data = MandelbrotAPI.CalculateJulia(CtrlX, CtrlY, maxIterations, coord_);
                        var bitmap = MandelbrotAPI.PaletteTransform(calculation_data, palette_);
                        MandelbrotAPI.RenderArrayToDevice(this.CreateGraphics().GetHdc(), coord_.ScreenWidth, coord_.ScreenHeight, bitmap);
                    }
                    break;
            }
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

        public double mouseX
        {
            get
            {
                return mx_;
            }
        }

        public double mouseY
        {
            get
            {
                return my_;
            }
        }

        public int FractalSetIndex { get; set; }
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

