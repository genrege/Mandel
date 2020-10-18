using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Xml.Schema;

namespace MandelbrotViewer
{
    public partial class ViewerForm : Form
    {

        public ViewerForm()
        {
            InitializeComponent();
        }

        RenderPanel renderPanel = new RenderPanel();
        OverviewPanel overviewPanel = new OverviewPanel();

        private void ViewerForm_Load(object sender, EventArgs e)
        {
            renderPanel.Dock = DockStyle.Fill;
            overviewPanel.Dock = DockStyle.Fill;
            SplitControlContainer.Panel2.Controls.Add(overviewPanel);
            mainSplitter.Panel2.Controls.Add(renderPanel);
            mainSplitter.Dock = DockStyle.Fill;

            renderPanel.StatusChange += RenderPanel_OnStatusChange;
            renderPanel.PositionChange += RenderPanel_OnPositionChange;
            overviewPanel.OnOverviewSetPosition += OnOverviewSetPosition;

            trackBarMaxIterations.Value = renderPanel.maxIterations;
            renderPanel.useGpu = checkBox1.Checked;
            sliderMax.Text = "6000";
            txtMaxIterations.Text = "100";

            cbWhichSet.SelectedIndex = 0;
        }

        private void OnOverviewSetPosition(object sender, EventArgs e)
        {
            var pi = (PositionInfo)e;
            renderPanel.CentreOn(pi.X, pi.Y);
        }

        private void RenderPanel_OnStatusChange(object sender, EventArgs e)
        {
            txtMouseCoords.Text = string.Format("{0}, {1}", renderPanel.mouseX, renderPanel.mouseY);
        }

        private void RenderPanel_OnPositionChange(object sender, EventArgs e)
        {
            var ssi = (SetScaleInfo)e;
            overviewPanel.DrawBox(ssi.X, ssi.Y, ssi.xMin, ssi.xMax, ssi.yMin, ssi.yMax, Color.Red);

            txtXMin.Text = string.Format("XMin: {0}", ssi.xMin);
            txtXMax.Text = string.Format("XMax: {0}", ssi.xMax);
            txtYMin.Text = string.Format("YMin: {0}", ssi.yMin);
            txtYMax.Text = string.Format("YMax: {0}", ssi.yMax);
            txtBounds.Text = string.Format("Bounds: [{0} : {1}]", ssi.xMax - ssi.xMin, ssi.yMax - ssi.yMin);

            if (ssi.xMax - ssi.xMin < 1.0E-12 || ssi.yMax - ssi.yMin < 1.0E-12)
            {
                txtBounds.BackColor = Color.Red;
            }
            else if (ssi.xMax - ssi.xMin < 1.0E-11 || ssi.yMax - ssi.yMin < 1.0E-11)
            {
                txtBounds.BackColor = Color.Orange;
            }
            else
            {
                txtBounds.BackColor = txtXMin.BackColor;
            }
        }

        private void txtMaxIterations_TextChanged(object sender, EventArgs e)
        {
            int maxIter;
            if (int.TryParse(txtMaxIterations.Text, out maxIter))
            {
                if (maxIter >= trackBarMaxIterations.Minimum && maxIter <= trackBarMaxIterations.Maximum)
                {
                    renderPanel.maxIterations = maxIter;
                    trackBarMaxIterations.Value = maxIter;
                    renderPanel.Invalidate();
                }
                else
                {
                    txtMaxIterations.Text = trackBarMaxIterations.Value.ToString();
                }
            }
        }

        private void trackBarMaxIterations_Scroll(object sender, EventArgs e)
        {
            try
            {
                txtMaxIterations.Text = trackBarMaxIterations.Value.ToString();
            }
            catch(Exception)
            {
                txtMaxIterations.Text = trackBarMaxIterations.Maximum.ToString();
            }
        }

        private void mainSplitter_Panel2_Paint(object sender, PaintEventArgs e)
        {

        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            renderPanel.useGpu = checkBox1.Checked;
            renderPanel.Invalidate();
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void sliderMax_TextChanged(object sender, EventArgs e)
        {
            int maxSlider = 0;
            if (int.TryParse(sliderMax.Text, out maxSlider))
            {
                if (maxSlider > 0 && maxSlider <= 1000000)
                {
                    trackBarMaxIterations.Maximum = maxSlider;
                    txtMaxIterations.Text = maxSlider.ToString();
                }
                else
                {
                    sliderMax.Text = trackBarMaxIterations.Value.ToString();
                }
            }
        }

        private void sliderMin_TextChanged(object sender, EventArgs e)
        {
            int minSlider = 0;
            if (int.TryParse(sliderMin.Text, out minSlider))
            {
                if (minSlider > 0 && minSlider < int.Parse(sliderMax.Text))
                {
                    trackBarMaxIterations.Minimum = minSlider;
                    txtMaxIterations.Text = minSlider.ToString();
                }
                else
                {
                    sliderMin.Text = "0";
                }
            }
        }

        private void btnLoad_Click(object sender, EventArgs e)
        {

        }

        private void btnSave_Click(object sender, EventArgs e)
        {
        }

        private void btnReset_Click(object sender, EventArgs e)
        {
            renderPanel.coordinateSpace().XMin = -2.5;
            renderPanel.coordinateSpace().YMin = -2.0;
            renderPanel.coordinateSpace().YMax = 2.0;
            renderPanel.Invalidate();
        }

        SaveFileDialog saveBmpDialog = null;

        private void btnSaveBMP_Click(object sender, EventArgs e)
        {
            if (saveBmpDialog == null)
            {
                saveBmpDialog = new SaveFileDialog();
                saveBmpDialog.Filter = "JPEG file|*.jpg|Bitmap file|*.bmp|Medium JPEG file|*.jpg|Medium Bitmap file|*.bmp|Large JPEG file|*.jpg|Large Bitmap file|*.bmp|Huge JPEG file|*.jpg|Huge Bitmap file|*.bmp";
                saveBmpDialog.FileName = "mbrot.jpg";
                saveBmpDialog.DefaultExt = "jpg";
                saveBmpDialog.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);
            }
            if (saveBmpDialog.ShowDialog() == DialogResult.OK)
            {
                var coord = renderPanel.coordinateSpace();
                var oldCursor = Cursor;
                Cursor = Cursors.WaitCursor;

                Int64 factor = 1;
                if (saveBmpDialog.FilterIndex >= 3 && saveBmpDialog.FilterIndex <= 4)
                    factor = 4;
                if (saveBmpDialog.FilterIndex >= 5 && saveBmpDialog.FilterIndex <= 8)
                    factor = 12;
                if (saveBmpDialog.FilterIndex > 6)
                    factor = 50;

                Int64 wx = coord.ScreenWidth * factor;
                Int64 wy = coord.ScreenHeight * factor;
                for (;;)
                {
                    if (wx * 4 * wy < Int32.MaxValue)
                        break;
                    factor--;
                    wx = coord.ScreenWidth * factor;
                    wy = coord.ScreenHeight * factor;

                }

                bool antiBuddha = renderPanel.FractalSetIndex == 3;

                if (System.IO.Path.GetExtension(saveBmpDialog.FileName) == ".jpg")
                {
                    if (renderPanel.FractalSetIndex == 0)
                        MandelbrotAPI.SaveJPGToFile(this.CreateGraphics().GetHdc(), (int)wx, (int)wy, int.Parse(txtMaxIterations.Text), coord.XMin, coord.XMax, coord.YMin, coord.YMax, saveBmpDialog.FileName);
                    else if (renderPanel.FractalSetIndex == 1)
                        MandelbrotAPI.SaveJuliaJPGToFile(renderPanel.CtrlX, renderPanel.CtrlY, this.CreateGraphics().GetHdc(), (int)wx, (int)wy, int.Parse(txtMaxIterations.Text), coord.XMin, coord.XMax, coord.YMin, coord.YMax, saveBmpDialog.FileName);
                    else if (renderPanel.FractalSetIndex == 2 || renderPanel.FractalSetIndex == 3)
                        MandelbrotAPI.SaveBuddhaJPGToFile(this.CreateGraphics().GetHdc(), antiBuddha, (int)wx, (int)wy, int.Parse(txtMaxIterations.Text), coord.XMin, coord.XMax, coord.YMin, coord.YMax, saveBmpDialog.FileName);
                }
                else
                {
                    if (renderPanel.FractalSetIndex == 0)
                        MandelbrotAPI.SaveBitmapToFile(this.CreateGraphics().GetHdc(), (int)wx, (int)wy, int.Parse(txtMaxIterations.Text), coord.XMin, coord.XMax, coord.YMin, coord.YMax, saveBmpDialog.FileName);
                    else if (renderPanel.FractalSetIndex == 1)
                        MandelbrotAPI.SaveJuliaBitmapToFile(renderPanel.CtrlX, renderPanel.CtrlY, this.CreateGraphics().GetHdc(), (int)wx, (int)wy, int.Parse(txtMaxIterations.Text), coord.XMin, coord.XMax, coord.YMin, coord.YMax, saveBmpDialog.FileName);
                    else if (renderPanel.FractalSetIndex == 2 || renderPanel.FractalSetIndex == 3)
                        MandelbrotAPI.SaveBuddhaBitmapToFile(this.CreateGraphics().GetHdc(), antiBuddha, (int)wx, (int)wy, int.Parse(txtMaxIterations.Text), coord.XMin, coord.XMax, coord.YMin, coord.YMax, saveBmpDialog.FileName);
                }
                Cursor = oldCursor;
            }
        }

        private void SplitControlContainer_Panel1_Paint(object sender, PaintEventArgs e)
        {

        }

        private void cbWhichSet_SelectedIndexChanged(object sender, EventArgs e)
        {
            renderPanel.FractalSetIndex = cbWhichSet.SelectedIndex;
            renderPanel.Invalidate();
        }
    }
}

