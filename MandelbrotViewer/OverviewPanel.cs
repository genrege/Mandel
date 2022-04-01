using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace MandelbrotViewer
{
    public partial class OverviewPanel : UserControl
    {
        CoordinateSpace coord_ = null;

        public event EventHandler OnOverviewSetPosition;

        public int maxIterations { get; set; }

        public OverviewPanel()
        {
            InitializeComponent();
            maxIterations = 1024;
        }

        public int gpuIndex { get; set; }

        private void OverviewPanel_Load(object sender, EventArgs e)
        {
            coord_ = new CoordinateSpace(Width, Height, -2.5, -1.8, 1.8);
        }

        protected override void OnPaintBackground(PaintEventArgs e)
        {
        }

        private void OverviewPanel_Paint(object sender, PaintEventArgs e)
        {
            double aspectRatio = (double)Width / (double)Height;
            var hdc = e.Graphics.GetHdc();
            MandelbrotAPI.RenderBasic(gpuIndex, hdc, false, false, maxIterations, coord_);
        }

        private void OverviewPanel_MouseClick(object sender, MouseEventArgs e)
        {
            EventHandler handler = OnOverviewSetPosition;
            if (handler != null)
            {
                var setPos = coord_.SetFromScreen(e.Location.X, e.Location.Y);
                var pi = new PositionInfo(setPos.X, setPos.Y, Control.ModifierKeys == Keys.Control);
                handler.Invoke(this, pi);
            }
        }

        private void OverviewPanel_Resize(object sender, EventArgs e)
        {
            coord_.ScreenWidth = Width;
            coord_.ScreenHeight = Height;
            Invalidate();
        }

        public void DrawBox(double x, double y, double x1, double x2, double y1, double y2, Color clr)
        {
            double aspectRatio = (double)Width / (double)Height;

            var p0 = coord_.ScreenFromSet(x1, y1);
            var p1 = coord_.ScreenFromSet(x2, y2);

            int mx1 = p0.X;
            int my1 = p0.Y;
            int mx2 = p1.X;
            int my2 = p1.Y;

            var hdc = this.CreateGraphics().GetHdc();
            MandelbrotAPI.RenderBasic(gpuIndex, hdc, true, false, maxIterations, coord_);

            int cx = mx1 + (mx2 - mx1) / 2;
            int cy = my1 + (my2 - my1) / 2;

            if (mx2 - mx1 < 4 || my2 - my1 < 4)
            {

                var pen2 = new Pen(Color.Red, 0);
                pen2.DashStyle = System.Drawing.Drawing2D.DashStyle.Dash;
                this.CreateGraphics().DrawLine(pen2, cx, 0, cx, Height);
                this.CreateGraphics().DrawLine(pen2, 0, cy, Width, cy);
            }
            else
            {
                var pen = new Pen(Color.Red, 1);
                pen.DashStyle = System.Drawing.Drawing2D.DashStyle.Dash;
                this.CreateGraphics().DrawRectangle(pen, new Rectangle(mx1, my1, Math.Max(1, mx2 - mx1), Math.Max(1, my2 - my1)));
            }
        }

        private void OverviewPanel_MouseMove(object sender, MouseEventArgs e)
        {
            var p = coord_.SetFromScreen(e.X, e.Y);

            EventHandler handler = OnOverviewSetPosition;
            if (Capture && handler != null)
            {
                var pi = new PositionInfo(p.X, p.Y, Control.ModifierKeys == Keys.Control);
                handler.Invoke(this, pi);
            }
        }

        private void OverviewPanel_MouseDown(object sender, MouseEventArgs e)
        {
            Capture = true;
        }

        private void OverviewPanel_MouseUp(object sender, MouseEventArgs e)
        {
            Capture = false;
        }
    }

    public class PositionInfo : EventArgs
    {
        public PositionInfo(double x, double y, bool juliaClick)
        {
            X = x;
            Y = y;
            JuliaClick = juliaClick;
        }

        public double X { get; set; }
        public double Y { get; set; }
        public bool JuliaClick { get; set; }
    }


}
