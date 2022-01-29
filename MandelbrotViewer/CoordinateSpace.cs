using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Net.NetworkInformation;
using System.Security.Policy;
using System.Text;
using System.Threading.Tasks;

namespace MandelbrotViewer
{
    public class CoordinateSpace
    {
        public CoordinateSpace(int w, int h, double xMin, double yMin, double yMax)
        {
            screen_width_ = w;
            screen_height_ = h;

            xmin_ = xMin;
            ymin_ = yMin;
            ymax_ = yMax;
        }

        public override string ToString()
        {
            var output = new StringBuilder();

            output.Append(screen_width_).Append(",");
            output.Append(screen_height_).Append(",");
            output.Append(xmin_).Append(",");
            output.Append(ymin_).Append(",");
            output.Append(ymax_);

            return output.ToString();
        }

        public static CoordinateSpace FromString(string input)
        {
            var toks = input.Split(',');
            if (toks.Length != 5)
                throw new SystemException("Failed to parse CoordinateSpace.FromString");

            var cs = new CoordinateSpace(
                int.Parse(toks[0]),
                int.Parse(toks[1]),
                double.Parse(toks[2]),
                double.Parse(toks[3]),
                double.Parse(toks[4])
                );
            cs.UpdateState();
            return cs;
        }

        public void UpdateState()
        {
            AspectRatio = (double)screen_width_ / (double)screen_height_;
            xmax_ = xmin_ + AspectRatio * (ymax_ - ymin_);
        }

        private static double txSetFromScreen(int s, int pixels, double setMin, double setMax)
        {
            double ratio_s = (double)s / (double)pixels;
            double set_coord = setMin + ratio_s * (setMax - setMin);
            return set_coord;
        }

        private static int txScreenFromSet(double s, int pixels, double setMin, double setMax)
        {
            double ratio_s = (s - setMin) / (setMax - setMin);
            int screenCoord = (int)(ratio_s * (double)pixels);
            return screenCoord;
        }

        public (double X, double Y) SetFromScreen(int sx, int sy)
        {
            double x = txSetFromScreen(sx, ScreenWidth, XMin, XMax);
            double y = txSetFromScreen(sy, ScreenHeight, YMin, YMax);

            return (x, y);
        }

        public (int X, int Y) ScreenFromSet(double x, double y)
        {
            int sx = txScreenFromSet(x, ScreenWidth, XMin, XMax);
            int sy = txScreenFromSet(y, ScreenHeight, YMin, YMax);

            return (sx, sy);
        }

        // Change xmin, xmax, ymin, ymax such that (x,y) is at screen coordinate sx, sy
        public void Align(double x, double y, int sx, int sy)
        {
            double delta_x = x - txSetFromScreen(sx, ScreenWidth, XMin, XMax);
            XMin += delta_x;

            double delta_y = y - txSetFromScreen(sy, ScreenHeight, YMin, YMax);
            YMin += delta_y;
            YMax += delta_y;
        }

        public void ShiftSpace(double dx, double dy)
        {
            XMin += dx;
            YMin += dy;
            YMax += dy;

            Debug.WriteLine("{0} {1} {2} {3}", XMin, XMax, YMin, YMax);
            UpdateState();
        }

        int screen_width_;
        public int ScreenWidth
        {
            get => screen_width_;
            set
            {
                screen_width_ = value;
                UpdateState();
            }
        }

        int screen_height_;
        public int ScreenHeight
        {
            get => screen_height_;
            set
            {
                screen_height_ = value;
                UpdateState();
            }
        }

        double xmin_;
        public double XMin 
        { 
            get => xmin_;
            set
            {
                xmin_ = value;
                UpdateState();
            }
        }
        double xmax_;
        public double XMax
        {
            get => xmax_;
            private set
            {
                xmax_ = value;
            }
        }

        double ymin_;
        public double YMin 
        { 
            get => ymin_; 
            set
            {
                ymin_ = value;
                UpdateState();
            }
        }
        double ymax_;
        public double YMax
        {
            get => ymax_;
            set
            {
                ymax_ = value;
                UpdateState();
            }
        }

        public double AspectRatio { get; private set; }

        public CoordinateSpace DeepCopy()
        {
            var clone = (CoordinateSpace)this.MemberwiseClone();
            return clone;
        }
    }
}
