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

        public CoordinateSpace Coord { get; set; }

        public int gpuIndex { get; set; }
        public double MouseDownSetX { get; set; }
        public double MouseDownSetY { get; set; }
        public double CurrentSetX { get; set; }
        public double CurrentSetY { get; set; }
        public double JuliaSetX { get; set; }
        public double JuliaSetY { get; set; }
        public int SpecialFunc { get; set; }

        public int MaxIterations { get; set; }
        public int FractalSetIndex { get; set; }
        public Point LastMouseMoveLocation { get; set; }

        public int PaletteOffset { get; set; }
        public bool PaletteDefer { get; set; }


        public bool useGpu { get; set; }
        public bool displayFPS { get; set; }
        public bool useCUDA { get; set; }
        int[] palette_ = null;
        int[] calculation_data_ = null;
        int[] bitmap_ = null;

        public class RecordingItem
        {
            public CoordinateSpace Coord { get; set; }
            public bool useGPU { get; set; }
            public bool useCUDA { get; set; }
            public int gpuIndex { get; set; }
            public double MouseDownSetX { get; set; }
            public double MouseDownSetY { get; set; }
            public double CurrentSetX { get; set; }
            public double CurrentSetY { get; set; }
            public double JuliaSetX { get; set; }
            public double JuliaSetY { get; set; }
            public int SpecialFunc { get; set; }

            public int MaxIterations { get; set; }
            public int FractalSetIndex { get; set; }
            public Point LastMouseMoveLocation { get; set; }
            public int PaletteOffset { get; set; }

            public override string ToString()
            {
                var output = new StringBuilder();

                output.AppendLine(Coord.ToString());

                output.Append(gpuIndex).Append(",");
                output.Append(MouseDownSetX).Append(",");
                output.Append(MouseDownSetY).Append(",");
                output.Append(CurrentSetX).Append(",");
                output.Append(CurrentSetY).Append(",");
                output.Append(JuliaSetY).Append(",");
                output.Append(JuliaSetY).Append(",");
                output.Append(SpecialFunc).Append(",");
                output.Append(MaxIterations).Append(",");
                output.Append(FractalSetIndex).Append(",");
                output.Append(LastMouseMoveLocation.X).Append(",");
                output.Append(LastMouseMoveLocation.Y).Append(",");
                output.Append(useGPU).Append(",");
                output.Append(useCUDA).Append(",");
                output.Append(PaletteOffset).Append(",");
                output.AppendLine();

                return output.ToString();
            }

            public static RecordingItem FromString(string coord_line, string line)
            {
                var r = new RecordingItem();

                r.Coord = CoordinateSpace.FromString(coord_line);

                var line_toks = line.Split(',');
                if (line_toks.Length != 16)
                    throw new SystemException("Failed to parse RecordingItem.FromString");

                r.gpuIndex = int.Parse(line_toks[0]);
                r.MouseDownSetX = double.Parse(line_toks[1]);
                r.MouseDownSetY = double.Parse(line_toks[2]);
                r.CurrentSetX = double.Parse(line_toks[3]);
                r.CurrentSetY = double.Parse(line_toks[4]);
                r.JuliaSetX = double.Parse(line_toks[5]);
                r.JuliaSetY = double.Parse(line_toks[6]);
                r.SpecialFunc = int.Parse(line_toks[7]);
                r.MaxIterations = int.Parse(line_toks[8]);
                r.FractalSetIndex = int.Parse(line_toks[9]);
                r.LastMouseMoveLocation = new Point(int.Parse(line_toks[10]), int.Parse(line_toks[11]));
                r.useGPU = bool.Parse(line_toks[12]);
                r.useCUDA = bool.Parse(line_toks[13]);
                r.PaletteOffset = int.Parse(line_toks[14]);

                return r;
            }
        }

        public RenderPanel()
        {
            InitializeComponent();
            MaxIterations = 1024;
            this.MouseWheel += RenderPanel_OnMouseWheel;
            Coord = new CoordinateSpace(Width, Height, -3.0, -2.0, 2.0);
            Coord.Align(-1.0, 0.0, Width / 2, Height / 2);
            displayFPS = true;
            Recording = false;
            PaletteOffset = 0;
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
            Coord.ScreenWidth = Width;
            Coord.ScreenHeight = Height;
            Invalidate();

            calculation_data_ = new int[Width * Height];
            bitmap_ = new int[Width * Height];
        }

        private void RenderPanel_MouseDown(object sender, MouseEventArgs e)
        {
            if (MouseButtons == MouseButtons.Left)
            {
                Capture = true;

                var cc = Coord.SetFromScreen(e.Location.X, e.Location.Y);

                MouseDownSetX = cc.X;
                MouseDownSetY = cc.Y;
            }
        }

        private void RenderPanel_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Location == LastMouseMoveLocation)
                return;
            LastMouseMoveLocation = e.Location;

            var cc = Coord.SetFromScreen(e.Location.X, e.Location.Y);
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

                Coord.ShiftSpace(nx, ny);

                EventHandler handler1 = PositionChange;
                if (handler1 != null)
                    handler1.Invoke(this, new SetScaleInfo(cc.X, cc.Y, Coord.XMin, Coord.XMax, Coord.YMin, Coord.YMax));
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
            Coord.Align(px, py, Width / 2, Height / 2);

            Render();

            EventHandler handler1 = PositionChange;
            if (handler1 != null)
            {
                var cc = Coord.SetFromScreen(Location.X, Location.Y);

                handler1.Invoke(this, new SetScaleInfo(cc.X, cc.Y, Coord.XMin, Coord.XMax, Coord.YMin, Coord.YMax));
            }
        }

        private void Zoom(double ratio, double px, double py)
        {
            double wx = Coord.XMax - Coord.XMin;
            double wy = Coord.YMax - Coord.YMin;

            Coord.XMin = -ratio * (px - Coord.XMin) + px;
            Coord.YMin = -ratio * (py - Coord.YMin) + py;
            Coord.YMax = Coord.YMin + wy * ratio;
        }

        private void DoZoom(bool into, double px, double py, double factor)
        {
            if (into)
            {
                Zoom(1.0 / factor, px, py);
            }
            else
            {
                Zoom(factor, px, py);
            }

            Render();

            EventHandler handler1 = PositionChange;
            if (handler1 != null)
                handler1.Invoke(this, new SetScaleInfo(px, py, Coord.XMin, Coord.XMax, Coord.YMin, Coord.YMax));
        }

        private void RenderPanel_OnMouseWheel(object sender, MouseEventArgs e)
        {
            var factor = Control.ModifierKeys == Keys.Control ? 2.0 : 1.1;

            if (Control.ModifierKeys == Keys.Shift)
                factor = 1.005;

            var pnt = PointToClient(new Point(MousePosition.X, MousePosition.Y));
            var cc = Coord.SetFromScreen(pnt.X, pnt.Y);
            double px = cc.X;
            double py = cc.Y;

            if (ModifierKeys == Keys.Shift)
            {
                while (ModifierKeys == Keys.Shift)
                {
                    pnt = PointToClient(new Point(MousePosition.X, MousePosition.Y));
                    cc = Coord.SetFromScreen(pnt.X, pnt.Y);
                    px = cc.X;
                    py = cc.Y;
                    DoZoom(e.Delta > 0, px, py, factor);
                    Application.DoEvents();
                }
            }
            else
                DoZoom(e.Delta > 0, px, py, factor);

            DoZoom(e.Delta > 0, px, py, factor);
        }

        private void Render()
        {
            var gr = CreateGraphics();
            var hdc = gr.GetHdc();

            var stopwatch = new Stopwatch();
            stopwatch.Restart();

            var stopwatch_calc = new Stopwatch();
            stopwatch.Restart();

            float calctime_ms = 0;

            switch (FractalSetIndex)
            {
                case 0:
                    MandelbrotAPI.RenderBasic(gpuIndex, hdc, useGpu, useCUDA, MaxIterations, Coord, PaletteOffset);
                    calctime_ms = 1000.0f * stopwatch.ElapsedTicks / Stopwatch.Frequency;
                    break;
                case 1:
                    MandelbrotAPI.RenderJulia(gpuIndex, hdc, useGpu, useCUDA, MaxIterations, JuliaSetX, JuliaSetY, Coord, PaletteOffset);
                    calctime_ms = 1000.0f * stopwatch.ElapsedTicks / Stopwatch.Frequency;
                    break;
                case 2:
                    MandelbrotAPI.RenderBuddha(gpuIndex, useCUDA, hdc, MaxIterations, Coord, PaletteOffset);
                    calctime_ms = 1000.0f * stopwatch.ElapsedTicks / Stopwatch.Frequency;
                    break;
                case 3:
                    MandelbrotAPI.RenderAntiBuddha(gpuIndex, useCUDA, hdc, MaxIterations, Coord, PaletteOffset);
                    calctime_ms = 1000.0f * stopwatch.ElapsedTicks / Stopwatch.Frequency;
                    break;
                case 4:
                    {
                        var calculation_data = MandelbrotAPI.CalculateMandelbrot(gpuIndex, useGpu, useCUDA, MaxIterations, Coord);
                        calctime_ms = 1000.0f * stopwatch.ElapsedTicks / Stopwatch.Frequency;
                        palette_ = MandelbrotAPI.StandardPalette(MaxIterations);
                        var bitmap = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette_);
                        MandelbrotAPI.RenderArrayToDisplay(hdc, Coord.ScreenWidth, Coord.ScreenHeight, bitmap);
                    }
                    break;
                case 5:
                    {
                        MandelbrotAPI.CalculateJulia2(gpuIndex, JuliaSetX, JuliaSetY, useGpu, useCUDA, MaxIterations, Coord, ref calculation_data_);
                        calctime_ms = 1000.0f * stopwatch.ElapsedTicks / Stopwatch.Frequency;
                        palette_ = MandelbrotAPI.StandardPalette(MaxIterations);
                        MandelbrotAPI.PaletteTransform2(gpuIndex, calculation_data_, palette_, ref bitmap_);
                        MandelbrotAPI.RenderArrayToDisplay(hdc, Coord.ScreenWidth, Coord.ScreenHeight, bitmap_);
                    }
                    break;
                case 6:
                    {
                        var calculation_data = MandelbrotAPI.CalculateBuddha(gpuIndex, false, MaxIterations, Coord);
                        calctime_ms = 1000.0f * stopwatch.ElapsedTicks / Stopwatch.Frequency;
                        palette_ = MandelbrotAPI.BuddhaPalette(MaxIterations);
                        var bitmap = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette_);
                        MandelbrotAPI.RenderArrayToDisplay(hdc, Coord.ScreenWidth, Coord.ScreenHeight, bitmap);
                    }
                    break;
                case 7:
                    {
                        var calculation_data = MandelbrotAPI.CalculateBuddha(gpuIndex, true, MaxIterations, Coord);
                        calctime_ms = 1000.0f * stopwatch.ElapsedTicks / Stopwatch.Frequency;
                        palette_ = MandelbrotAPI.BuddhaPalette(MaxIterations);
                        var bitmap = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette_);
                        MandelbrotAPI.RenderArrayToDisplay(hdc, Coord.ScreenWidth, Coord.ScreenHeight, bitmap);
                    }
                    break;
                case 8:
                    {
                        var calculation_data = MandelbrotAPI.CalculateSpecial(gpuIndex, SpecialFunc, JuliaSetX, JuliaSetY, useGpu, MaxIterations, Coord);
                        calctime_ms = 1000.0f * stopwatch.ElapsedTicks / Stopwatch.Frequency;
                        palette_ = MandelbrotAPI.StandardPalette(MaxIterations);
                        var bitmap = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette_);
                        MandelbrotAPI.RenderArrayToDisplay(hdc, Coord.ScreenWidth, Coord.ScreenHeight, bitmap);
                    }
                    break;
            }

            var frame_time_ms = 1000.0f * stopwatch.ElapsedTicks / Stopwatch.Frequency;

            gr.ReleaseHdc(hdc);
            gr.FillRectangle(Brushes.Black, 0, 0, 150, 14);
            gr.DrawString(((int)frame_time_ms).ToString() + " ms", DefaultFont, Brushes.LightGreen, 0, 0);
            gr.DrawString(((int)calctime_ms).ToString() + " ms", DefaultFont, Brushes.LightGreen, 50, 0);
            gr.DrawString(frame_time_ms == 0 ? "XX fps" : ((int)(1000.0f / frame_time_ms)).ToString() + " fps", DefaultFont, Brushes.LightGreen, 100, 0);
            

            //gr.Dispose();
            CaptureRecording();
        }

        private void RenderPanel_Paint(object sender, PaintEventArgs e)
        {
            var hdc = e.Graphics.GetHdc();

            Render();

            EventHandler handler1 = PositionChange;
            if (handler1 != null)
            {
                var cc = Coord.SetFromScreen(Location.X, Location.Y);
                handler1.Invoke(this, new SetScaleInfo(cc.X, cc.Y, Coord.XMin, Coord.XMax, Coord.YMin, Coord.YMax));
            }
        }

        public bool Recording {  get; private set; }
        

        public List<RecordingItem> RecordedItems = null;

        void CaptureRecording()
        {
            if (Recording)
            {
                RecordingItem item = new RecordingItem();

                item.Coord = Coord.DeepCopy();
                item.MouseDownSetX = MouseDownSetX;
                item.MouseDownSetY = MouseDownSetY;
                item.CurrentSetX = CurrentSetX;
                item.CurrentSetY = CurrentSetY;
                item.JuliaSetX = JuliaSetX;
                item.JuliaSetY = JuliaSetY;
                item.SpecialFunc = SpecialFunc;
                item.MaxIterations = MaxIterations; 
                item.FractalSetIndex = FractalSetIndex;
                item.LastMouseMoveLocation = LastMouseMoveLocation;
                item.useGPU = useGpu;
                item.useCUDA = useCUDA;
                item.gpuIndex = gpuIndex;
                item.PaletteOffset = PaletteOffset;

                RecordedItems.Add(item);
            }
        }


        public void Record(bool record)
        {
            Recording = record;

            if (Recording)
            {
                RecordedItems = new List<RecordingItem>();
                CaptureRecording();
            }
            else
            {
                //todo
            }
        }

        public int Replay(int pos)
        {
            if (RecordedItems != null  && pos < RecordedItems.Count)
            {
                var item = RecordedItems[pos];

                Coord = item.Coord;
                Coord.ScreenWidth = Width;
                Coord.ScreenHeight = Height;

                MouseDownSetX = item.MouseDownSetX;
                MouseDownSetY = item.MouseDownSetY;
                CurrentSetX = item.CurrentSetX;
                CurrentSetY = item.CurrentSetY;
                JuliaSetX = item.JuliaSetX;
                JuliaSetY = item.JuliaSetY;
                SpecialFunc = item.SpecialFunc;
                MaxIterations = item.MaxIterations;
                FractalSetIndex = item.FractalSetIndex;
                LastMouseMoveLocation = item.LastMouseMoveLocation;
                //useGpu = item.useGPU;
                //gpuIndex = item.gpuIndex;
                PaletteOffset = item.PaletteOffset;

                Coord.UpdateState();
                return pos + 1;
            }
            return -1;
        }

        private void button1_Click(object sender, EventArgs e)
        {

        }

        private void RenderPanel_MouseDoubleClick(object sender, MouseEventArgs e)
        {
            //Quick and dirty performance test, CUDA vs C++ AMP

            MandelbrotAPI.CalculateMandelbrot(gpuIndex, useGpu, false, MaxIterations, Coord);
            var timer = new Stopwatch();
            timer.Start();
            for (int i = 0; i < 10; ++i)
            {
                MandelbrotAPI.CalculateMandelbrot(gpuIndex, useGpu, false, MaxIterations, Coord);
            }
            timer.Stop();
            var elapsedAMP = timer.ElapsedMilliseconds;

            MandelbrotAPI.CalculateMandelbrot(gpuIndex, true, true, MaxIterations, Coord);
            timer.Reset();
            timer.Start();
            for (int i = 0; i < 10; ++i)
            {
                MandelbrotAPI.CalculateMandelbrot(gpuIndex, true, true, MaxIterations, Coord);
            }
            timer.Stop();
            var elapsedCuda = timer.ElapsedMilliseconds;

            MessageBox.Show("Current (non-CUDA): " + elapsedAMP.ToString() + "ms, " + Environment.NewLine +
                            "CUDA: " + elapsedCuda.ToString() + "ms" + Environment.NewLine +
                            "Non-CUDA average:" + (elapsedAMP * 0.1f).ToString() + "ms" + Environment.NewLine + 
                            "CUDA average:" + (elapsedCuda * 0.1f).ToString() + "ms" + Environment.NewLine +
                            "Scale Factor: " + ((float)(elapsedAMP - elapsedCuda)/ (float)elapsedAMP).ToString() 
                            );

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

