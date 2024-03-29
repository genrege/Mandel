﻿using System;
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
using System.Runtime.Serialization.Formatters.Binary;

namespace MandelbrotViewer
{
    public partial class MainForm : Form
    {

        public MainForm()
        {
            InitializeComponent();
        }

        RenderPanel renderPanel = new RenderPanel();
        OverviewPanel overviewPanel = new OverviewPanel();

        public int gpuIndex { get; set; }

        private void MainForm_Load(object sender, EventArgs e)
        {
            renderPanel.Dock = DockStyle.Fill;
            overviewPanel.Dock = DockStyle.Fill;
            SplitControlContainer.Panel2.Controls.Add(overviewPanel);
            mainSplitter.Panel2.Controls.Add(renderPanel);
            mainSplitter.Dock = DockStyle.Fill;

            renderPanel.StatusChange += RenderPanel_OnStatusChange;
            renderPanel.PositionChange += RenderPanel_OnPositionChange;
            overviewPanel.OnOverviewSetPosition += OnOverviewSetPosition;

            trackBarMaxIterations.Value = renderPanel.MaxIterations;
            renderPanel.gpuIndex = gpuIndex;

            string[] gpus = new string[100];
            MandelbrotDLLInterface.GPU(ref gpus);

            foreach (var gpu in gpus)
            {
                if (gpu != "CPU accelerator")
                    listGPU.Items.Add(gpu);
            }
            listGPU.Items.Add("CPU");

            listGPU.SelectedIndex = 0;
            renderPanel.useCUDA = listGPU.Text.Contains("CUDA");
            overviewPanel.UseCuda = renderPanel.useCUDA;

            renderPanel.useGpu = listGPU.Text != "CPU";

            sliderMax.Text = "6000";
            txtMaxIterations.Text = "100";
            txtMaxIterations.Minimum = 1;
            txtMaxIterations.Maximum = 6000;
            upDown1.Value = 0;
            upDown1.Minimum = 0;
            upDown1.Maximum = 19;
            upDown1.Visible = false;

            cbWhichSet.SelectedIndex = 0;
        }

        private void OnOverviewSetPosition(object sender, EventArgs e)
        {
            var pi = (PositionInfo)e;
            if (pi.JuliaClick)
                renderPanel.setJulia(pi.X, pi.Y);
            else
                renderPanel.CentreOn(pi.X, pi.Y);
        }

        private void RenderPanel_OnStatusChange(object sender, EventArgs e)
        {
            txtMouseCoords.Text = string.Format("{0}, {1}", renderPanel.CurrentSetX, renderPanel.CurrentSetY);
        }

        private void RenderPanel_OnPositionChange(object sender, EventArgs e)
        {
            var ssi = (SetScaleInfo)e;
            overviewPanel.maxIterations = trackBarMaxIterations.Value;
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
                    renderPanel.MaxIterations = maxIter;
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

        private void sliderMax_TextChanged(object sender, EventArgs e)
        {
            int maxSlider = 0;
            if (int.TryParse(sliderMax.Text, out maxSlider))
            {
                if (maxSlider > 0 && maxSlider <= 1000000)
                {
                    trackBarMaxIterations.Maximum = maxSlider;
                    txtMaxIterations.Text = maxSlider.ToString();
                    txtMaxIterations.Maximum = maxSlider;
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
                    txtMaxIterations.Minimum = minSlider;
                }
                else
                {
                    sliderMin.Text = "0";
                }
            }
        }

        private void btnLoad_Click(object sender, EventArgs e)
        {
            MessageBox.Show("Load state - not implemented");
        }

        private void btnSave_Click(object sender, EventArgs e)
        {
            MessageBox.Show("Save state - not implemented");
        }

        private void btnReset_Click(object sender, EventArgs e)
        {
            renderPanel.Coord.XMin = -2.5;
            renderPanel.Coord.YMin = -2.0;
            renderPanel.Coord.YMax = 2.0;
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
                saveBmpDialog.InitialDirectory = System.IO.Path.GetDirectoryName(saveBmpDialog.FileName);

                var oldCursor = Cursor;
                Cursor = Cursors.WaitCursor;

                Int64 factor = 1;
                if (saveBmpDialog.FilterIndex >= 3 && saveBmpDialog.FilterIndex <= 4)
                    factor = 4;
                if (saveBmpDialog.FilterIndex >= 5 && saveBmpDialog.FilterIndex <= 8)
                    factor = 12;
                if (saveBmpDialog.FilterIndex > 6)
                    factor = 50;

                var localCoord = renderPanel.Coord;
                Int64 wx = localCoord.ScreenWidth * factor;
                Int64 wy = localCoord.ScreenHeight * factor;
                for (;;)
                {
                    if (wx * 4 * wy < Int32.MaxValue)
                        break;
                    factor--;
                    wx = localCoord.ScreenWidth * factor;
                    wy = localCoord.ScreenHeight * factor;

                }

                var extCoord = renderPanel.Coord.DeepCopy();
                extCoord.ScreenWidth = (int)wx;
                extCoord.ScreenHeight = (int)wy;

                if (System.IO.Path.GetExtension(saveBmpDialog.FileName) == ".jpg")
                {
                    if (renderPanel.FractalSetIndex == 0)
                        MandelbrotAPI.SaveJPGToFile(gpuIndex, this.CreateGraphics().GetHdc(), int.Parse(txtMaxIterations.Text), extCoord, Decimal.ToInt32(nudPaletteOffset.Value), saveBmpDialog.FileName);
                    else if (renderPanel.FractalSetIndex == 1)
                        MandelbrotAPI.SaveJuliaJPGToFile(gpuIndex, renderPanel.JuliaSetX, renderPanel.JuliaSetY, this.CreateGraphics().GetHdc(), int.Parse(txtMaxIterations.Text), extCoord, Decimal.ToInt32(nudPaletteOffset.Value), saveBmpDialog.FileName);
                    else if (renderPanel.FractalSetIndex == 2)
                        MandelbrotAPI.SaveBuddhaJPGToFile(gpuIndex, this.CreateGraphics().GetHdc(), int.Parse(txtMaxIterations.Text), extCoord, saveBmpDialog.FileName);
                    else if (renderPanel.FractalSetIndex == 3)
                        MandelbrotAPI.SaveAntiBuddhaJPGToFile(gpuIndex, this.CreateGraphics().GetHdc(), int.Parse(txtMaxIterations.Text), extCoord, saveBmpDialog.FileName);
                    else if (renderPanel.FractalSetIndex == 4)
                    {
                        var calculation_data = MandelbrotAPI.CalculateMandelbrot(gpuIndex, true, renderPanel.useCUDA, int.Parse(txtMaxIterations.Text), extCoord);
                        var palette = MandelbrotAPI.StandardPalette(int.Parse(txtMaxIterations.Text));
                        var result = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette);
                        MandelbrotAPI.RenderArrayToJPEG(this.CreateGraphics().GetHdc(), (int)wx, (int)wy, result, saveBmpDialog.FileName);
                    }
                    else if (renderPanel.FractalSetIndex == 5)
                    {
                        var calculation_data = MandelbrotAPI.CalculateJulia(gpuIndex, renderPanel.JuliaSetX, renderPanel.JuliaSetY, true, renderPanel.useCUDA, int.Parse(txtMaxIterations.Text), extCoord);
                        var palette = MandelbrotAPI.StandardPalette(int.Parse(txtMaxIterations.Text));
                        var result = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette);
                        MandelbrotAPI.RenderArrayToJPEG(this.CreateGraphics().GetHdc(), (int)wx, (int)wy, result, saveBmpDialog.FileName);
                    }
                    else if (renderPanel.FractalSetIndex == 6)
                    {
                        var calculation_data = MandelbrotAPI.CalculateBuddha(gpuIndex, false, int.Parse(txtMaxIterations.Text), extCoord);
                        var palette = MandelbrotAPI.BuddhaPalette(int.Parse(txtMaxIterations.Text));
                        var result = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette);
                        MandelbrotAPI.RenderArrayToJPEG(this.CreateGraphics().GetHdc(), (int)wx, (int)wy, result, saveBmpDialog.FileName);
                    }
                    else if (renderPanel.FractalSetIndex == 7)
                    {
                        var calculation_data = MandelbrotAPI.CalculateBuddha(gpuIndex, true, int.Parse(txtMaxIterations.Text), extCoord);
                        var palette = MandelbrotAPI.BuddhaPalette(int.Parse(txtMaxIterations.Text));
                        var result = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette);
                        MandelbrotAPI.RenderArrayToJPEG(this.CreateGraphics().GetHdc(), (int)wx, (int)wy, result, saveBmpDialog.FileName);
                    }
                    else if (renderPanel.FractalSetIndex == 8)
                    {
                        var calculation_data = MandelbrotAPI.CalculateSpecial(gpuIndex, renderPanel.SpecialFunc, renderPanel.JuliaSetX, renderPanel.JuliaSetY, true, int.Parse(txtMaxIterations.Text), extCoord);
                        var palette = MandelbrotAPI.StandardPalette(int.Parse(txtMaxIterations.Text));
                        var result = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette);
                        MandelbrotAPI.RenderArrayToJPEG(this.CreateGraphics().GetHdc(), (int)wx, (int)wy, result, saveBmpDialog.FileName);
                    }
                }
                else
                {
                    if (renderPanel.FractalSetIndex == 0)
                        MandelbrotAPI.SaveBitmapToFile(gpuIndex, this.CreateGraphics().GetHdc(), int.Parse(txtMaxIterations.Text), extCoord, Decimal.ToInt32(nudPaletteOffset.Value), saveBmpDialog.FileName);
                    else if (renderPanel.FractalSetIndex == 1)
                        MandelbrotAPI.SaveJuliaBitmapToFile(gpuIndex, renderPanel.JuliaSetX, renderPanel.JuliaSetY, this.CreateGraphics().GetHdc(), int.Parse(txtMaxIterations.Text), extCoord, Decimal.ToInt32(nudPaletteOffset.Value), saveBmpDialog.FileName);
                    else if (renderPanel.FractalSetIndex == 2)
                        MandelbrotAPI.SaveBuddhaBitmapToFile(gpuIndex, this.CreateGraphics().GetHdc(), int.Parse(txtMaxIterations.Text), extCoord, saveBmpDialog.FileName);
                    else if (renderPanel.FractalSetIndex == 3)
                        MandelbrotAPI.SaveAntiBuddhaBitmapToFile(gpuIndex, this.CreateGraphics().GetHdc(), int.Parse(txtMaxIterations.Text), extCoord, saveBmpDialog.FileName);
                    else if (renderPanel.FractalSetIndex == 4)
                    {
                        var calculation_data = MandelbrotAPI.CalculateMandelbrot(gpuIndex, true, renderPanel.useCUDA, int.Parse(txtMaxIterations.Text), extCoord);
                        var palette = MandelbrotAPI.StandardPalette(int.Parse(txtMaxIterations.Text));
                        var result = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette);
                        MandelbrotAPI.RenderArrayToBitmap(this.CreateGraphics().GetHdc(), (int)wx, (int)wy, result, saveBmpDialog.FileName);
                    }
                    else if (renderPanel.FractalSetIndex == 5)
                    {
                        var calculation_data = MandelbrotAPI.CalculateJulia(gpuIndex, renderPanel.JuliaSetX, renderPanel.JuliaSetY, true, renderPanel.useCUDA, int.Parse(txtMaxIterations.Text), extCoord);
                        var palette = MandelbrotAPI.StandardPalette(int.Parse(txtMaxIterations.Text));
                        var result = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette);
                        MandelbrotAPI.RenderArrayToBitmap(this.CreateGraphics().GetHdc(), (int)wx, (int)wy, result, saveBmpDialog.FileName);
                    }
                    else if (renderPanel.FractalSetIndex == 6)
                    {
                        var calculation_data = MandelbrotAPI.CalculateBuddha(gpuIndex, false, int.Parse(txtMaxIterations.Text), extCoord);
                        var palette = MandelbrotAPI.BuddhaPalette(int.Parse(txtMaxIterations.Text));
                        var result = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette);
                        MandelbrotAPI.RenderArrayToBitmap(this.CreateGraphics().GetHdc(), (int)wx, (int)wy, result, saveBmpDialog.FileName);
                    }
                    else if (renderPanel.FractalSetIndex == 7)
                    {
                        var calculation_data = MandelbrotAPI.CalculateBuddha(gpuIndex, true, int.Parse(txtMaxIterations.Text), extCoord);
                        var palette = MandelbrotAPI.BuddhaPalette(int.Parse(txtMaxIterations.Text));
                        var result = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette);
                        MandelbrotAPI.RenderArrayToBitmap(this.CreateGraphics().GetHdc(), (int)wx, (int)wy, result, saveBmpDialog.FileName);
                    }
                    else if (renderPanel.FractalSetIndex == 8)
                    {
                        var calculation_data = MandelbrotAPI.CalculateSpecial(gpuIndex, renderPanel.SpecialFunc, renderPanel.JuliaSetX, renderPanel.JuliaSetY, true, int.Parse(txtMaxIterations.Text), extCoord);
                        var palette = MandelbrotAPI.StandardPalette(int.Parse(txtMaxIterations.Text));
                        var result = MandelbrotAPI.PaletteTransform(gpuIndex, calculation_data, palette);
                        MandelbrotAPI.RenderArrayToBitmap(this.CreateGraphics().GetHdc(), (int)wx, (int)wy, result, saveBmpDialog.FileName);
                    }
                }
                Cursor = oldCursor;
            }
        }

        private void SplitControlContainer_Panel1_Paint(object sender, PaintEventArgs e)
        {

        }

        private void cbWhichSet_SelectedIndexChanged(object sender, EventArgs e)
        {
            upDown1.Visible = cbWhichSet.SelectedIndex == 8;
            renderPanel.FractalSetIndex = cbWhichSet.SelectedIndex;
            renderPanel.Invalidate();
        }

        private void menuMain_ItemClicked(object sender, ToolStripItemClickedEventArgs e)
        {

        }

        private void upDown1_ValueChanged(object sender, EventArgs e)
        {
            renderPanel.SpecialFunc = (int)upDown1.Value;
            renderPanel.Invalidate();
        }

        private void btnResetJulia_Click(object sender, EventArgs e)
        {
            renderPanel.JuliaSetX = 0;
            renderPanel.JuliaSetY = 0;
            renderPanel.Invalidate();
        }

        private void linkLabel1_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            System.Diagnostics.Process.Start(linkLabel1.Text);
        }

        private void listGPU_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (listGPU.Text != "CPU")
                gpuIndex = int.Parse(listGPU.Text.Split(';')[0]);
            renderPanel.useGpu = listGPU.Text != "CPU";
            renderPanel.useCUDA = listGPU.Text.Contains("CUDA");
            overviewPanel.UseCuda = renderPanel.useCUDA;
            renderPanel.gpuIndex = gpuIndex;
            overviewPanel.gpuIndex = gpuIndex;
            overviewPanel.Invalidate();
            renderPanel.Invalidate();
        }

        private void btnRecord_Click(object sender, EventArgs e)
        {
            if (btnRecord.Text == "Record")
            {
                renderPanel.Record(true);
                btnRecord.Text = "Stop";
            }
            else if (btnRecord.Text == "Stop")
            {
                renderPanel.Record(false);
                btnRecord.Text = "Record";
            }
        }

        int pos = 0;
        private void btnReplay_Click(object sender, EventArgs e)
        {
            pos = 0;
            if (btnReplay.Text == "Replay")
                btnReplay.Text = "Stop";
            else
                btnReplay.Text = "Replay";
        }

        private void btnSaveRecording_Click(object sender, EventArgs e)
        {
            var dlg = new SaveFileDialog();
            dlg.Filter = "Mandel Recorder file|*.mrc";
            dlg.FileName = "fractal.mrc";
            dlg.DefaultExt = "mrc";
            dlg.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            if (dlg.ShowDialog() == DialogResult.OK)
            {
                var output = new StringBuilder();
                foreach (var rec in renderPanel.RecordedItems)
                {
                    output.Append(rec.ToString());
                }
                System.IO.File.WriteAllText(dlg.FileName, output.ToString());
            }
        }

        private void btnLoadRec_Click(object sender, EventArgs e)
        {
            var dlg = new OpenFileDialog();
            dlg.Filter = "Mandel Recorder file|*.mrc";
            dlg.FileName = "fractal.mrc";
            dlg.DefaultExt = "mrc";
            dlg.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            if (dlg.ShowDialog() == DialogResult.OK)
            {

                renderPanel.RecordedItems = new List<RenderPanel.RecordingItem>();

                var lines = System.IO.File.ReadAllLines(dlg.FileName);
                for (int i = 0; i < lines.Length; i += 2)
                {
                    var recItem = RenderPanel.RecordingItem.FromString(lines[i], lines[i + 1]);
                    renderPanel.RecordedItems.Add(recItem);
                }
            }
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            if (btnReplay.Text == "Stop")
            {
                if (checkLoop.Checked)
                {
                    if (pos < 0)
                        pos = 0;
                    pos = renderPanel.Replay(pos);
                    txtGeneral.Text = "Iteration " + pos.ToString();
                    txtGeneral.Invalidate();
                    renderPanel.Invalidate();
                }
                else
                {
                    if (pos < 0)
                    {
                        btnReplay.Text = "Replay";
                        pos = 0;
                    }
                    else
                    {
                        pos = renderPanel.Replay(pos);
                        txtGeneral.Text = "Iteration " + pos.ToString();
                        txtGeneral.Invalidate();
                        renderPanel.Invalidate();
                    }
                }
            }
        }

        private void txtMaxIterations_ValueChanged(object sender, EventArgs e)
        {
            trackBarMaxIterations.Value = (int)txtMaxIterations.Value;
            renderPanel.MaxIterations = trackBarMaxIterations.Value;
            nudPaletteOffset.Maximum = trackBarMaxIterations.Value;
            nudPaletteOffset.Increment = Math.Max(1, trackBarMaxIterations.Value / 256);

            renderPanel.Invalidate();
        }


        private void nudPaletteOffset_ValueChanged(object sender, EventArgs e)
        {
            int offset = 0;
            if (int.TryParse(nudPaletteOffset.Text, out offset))
            {
                renderPanel.PaletteOffset = int.Parse(nudPaletteOffset.Text);
                renderPanel.Refresh();
            }
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
        }
    }
}

