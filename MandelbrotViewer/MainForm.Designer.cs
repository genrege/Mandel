namespace MandelbrotViewer
{
    partial class MainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.mainSplitter = new System.Windows.Forms.SplitContainer();
            this.SplitControlContainer = new System.Windows.Forms.SplitContainer();
            this.checkLoop = new System.Windows.Forms.CheckBox();
            this.btnLoadRec = new System.Windows.Forms.Button();
            this.btnSaveRecording = new System.Windows.Forms.Button();
            this.btnReplay = new System.Windows.Forms.Button();
            this.btnRecord = new System.Windows.Forms.Button();
            this.listGPU = new System.Windows.Forms.ComboBox();
            this.linkLabel1 = new System.Windows.Forms.LinkLabel();
            this.btnResetJulia = new System.Windows.Forms.Button();
            this.upDown1 = new System.Windows.Forms.NumericUpDown();
            this.cbWhichSet = new System.Windows.Forms.ComboBox();
            this.btnSaveBMP = new System.Windows.Forms.Button();
            this.btnResetZoom = new System.Windows.Forms.Button();
            this.btnSave = new System.Windows.Forms.Button();
            this.btnLoad = new System.Windows.Forms.Button();
            this.txtGeneral = new System.Windows.Forms.TextBox();
            this.sliderMin = new System.Windows.Forms.TextBox();
            this.sliderMax = new System.Windows.Forms.TextBox();
            this.txtBounds = new System.Windows.Forms.TextBox();
            this.txtYMax = new System.Windows.Forms.TextBox();
            this.txtYMin = new System.Windows.Forms.TextBox();
            this.txtXMax = new System.Windows.Forms.TextBox();
            this.txtXMin = new System.Windows.Forms.TextBox();
            this.checkBox1 = new System.Windows.Forms.CheckBox();
            this.trackBarMaxIterations = new System.Windows.Forms.TrackBar();
            this.txtMouseCoords = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.txtMaxIterations = new System.Windows.Forms.NumericUpDown();
            ((System.ComponentModel.ISupportInitialize)(this.mainSplitter)).BeginInit();
            this.mainSplitter.Panel1.SuspendLayout();
            this.mainSplitter.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.SplitControlContainer)).BeginInit();
            this.SplitControlContainer.Panel1.SuspendLayout();
            this.SplitControlContainer.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.upDown1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarMaxIterations)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.txtMaxIterations)).BeginInit();
            this.SuspendLayout();
            // 
            // mainSplitter
            // 
            this.mainSplitter.Dock = System.Windows.Forms.DockStyle.Fill;
            this.mainSplitter.Location = new System.Drawing.Point(0, 0);
            this.mainSplitter.Name = "mainSplitter";
            // 
            // mainSplitter.Panel1
            // 
            this.mainSplitter.Panel1.Controls.Add(this.SplitControlContainer);
            // 
            // mainSplitter.Panel2
            // 
            this.mainSplitter.Panel2.Paint += new System.Windows.Forms.PaintEventHandler(this.mainSplitter_Panel2_Paint);
            this.mainSplitter.Size = new System.Drawing.Size(1138, 654);
            this.mainSplitter.SplitterDistance = 379;
            this.mainSplitter.TabIndex = 0;
            // 
            // SplitControlContainer
            // 
            this.SplitControlContainer.Dock = System.Windows.Forms.DockStyle.Fill;
            this.SplitControlContainer.Location = new System.Drawing.Point(0, 0);
            this.SplitControlContainer.Name = "SplitControlContainer";
            this.SplitControlContainer.Orientation = System.Windows.Forms.Orientation.Horizontal;
            // 
            // SplitControlContainer.Panel1
            // 
            this.SplitControlContainer.Panel1.BackColor = System.Drawing.Color.Black;
            this.SplitControlContainer.Panel1.Controls.Add(this.txtMaxIterations);
            this.SplitControlContainer.Panel1.Controls.Add(this.checkLoop);
            this.SplitControlContainer.Panel1.Controls.Add(this.btnLoadRec);
            this.SplitControlContainer.Panel1.Controls.Add(this.btnSaveRecording);
            this.SplitControlContainer.Panel1.Controls.Add(this.btnReplay);
            this.SplitControlContainer.Panel1.Controls.Add(this.btnRecord);
            this.SplitControlContainer.Panel1.Controls.Add(this.listGPU);
            this.SplitControlContainer.Panel1.Controls.Add(this.linkLabel1);
            this.SplitControlContainer.Panel1.Controls.Add(this.btnResetJulia);
            this.SplitControlContainer.Panel1.Controls.Add(this.upDown1);
            this.SplitControlContainer.Panel1.Controls.Add(this.cbWhichSet);
            this.SplitControlContainer.Panel1.Controls.Add(this.btnSaveBMP);
            this.SplitControlContainer.Panel1.Controls.Add(this.btnResetZoom);
            this.SplitControlContainer.Panel1.Controls.Add(this.btnSave);
            this.SplitControlContainer.Panel1.Controls.Add(this.btnLoad);
            this.SplitControlContainer.Panel1.Controls.Add(this.txtGeneral);
            this.SplitControlContainer.Panel1.Controls.Add(this.sliderMin);
            this.SplitControlContainer.Panel1.Controls.Add(this.sliderMax);
            this.SplitControlContainer.Panel1.Controls.Add(this.txtBounds);
            this.SplitControlContainer.Panel1.Controls.Add(this.txtYMax);
            this.SplitControlContainer.Panel1.Controls.Add(this.txtYMin);
            this.SplitControlContainer.Panel1.Controls.Add(this.txtXMax);
            this.SplitControlContainer.Panel1.Controls.Add(this.txtXMin);
            this.SplitControlContainer.Panel1.Controls.Add(this.checkBox1);
            this.SplitControlContainer.Panel1.Controls.Add(this.trackBarMaxIterations);
            this.SplitControlContainer.Panel1.Controls.Add(this.txtMouseCoords);
            this.SplitControlContainer.Panel1.Controls.Add(this.label1);
            this.SplitControlContainer.Panel1.Paint += new System.Windows.Forms.PaintEventHandler(this.SplitControlContainer_Panel1_Paint);
            this.SplitControlContainer.Size = new System.Drawing.Size(379, 654);
            this.SplitControlContainer.SplitterDistance = 467;
            this.SplitControlContainer.TabIndex = 4;
            // 
            // checkLoop
            // 
            this.checkLoop.AutoSize = true;
            this.checkLoop.ForeColor = System.Drawing.SystemColors.ControlLightLight;
            this.checkLoop.Location = new System.Drawing.Point(284, 287);
            this.checkLoop.Name = "checkLoop";
            this.checkLoop.Size = new System.Drawing.Size(96, 17);
            this.checkLoop.TabIndex = 26;
            this.checkLoop.Text = "Loop playback";
            this.checkLoop.UseVisualStyleBackColor = true;
            // 
            // btnLoadRec
            // 
            this.btnLoadRec.Location = new System.Drawing.Point(283, 403);
            this.btnLoadRec.Name = "btnLoadRec";
            this.btnLoadRec.Size = new System.Drawing.Size(84, 23);
            this.btnLoadRec.TabIndex = 25;
            this.btnLoadRec.Text = "LoadRec";
            this.btnLoadRec.UseVisualStyleBackColor = true;
            this.btnLoadRec.Click += new System.EventHandler(this.btnLoadRec_Click);
            // 
            // btnSaveRecording
            // 
            this.btnSaveRecording.Location = new System.Drawing.Point(284, 374);
            this.btnSaveRecording.Name = "btnSaveRecording";
            this.btnSaveRecording.Size = new System.Drawing.Size(84, 23);
            this.btnSaveRecording.TabIndex = 24;
            this.btnSaveRecording.Text = "SaveRec";
            this.btnSaveRecording.UseVisualStyleBackColor = true;
            this.btnSaveRecording.Click += new System.EventHandler(this.btnSaveRecording_Click);
            // 
            // btnReplay
            // 
            this.btnReplay.Location = new System.Drawing.Point(283, 345);
            this.btnReplay.Name = "btnReplay";
            this.btnReplay.Size = new System.Drawing.Size(84, 23);
            this.btnReplay.TabIndex = 23;
            this.btnReplay.Text = "Replay";
            this.btnReplay.UseVisualStyleBackColor = true;
            this.btnReplay.Click += new System.EventHandler(this.btnReplay_Click);
            // 
            // btnRecord
            // 
            this.btnRecord.Location = new System.Drawing.Point(284, 316);
            this.btnRecord.Name = "btnRecord";
            this.btnRecord.Size = new System.Drawing.Size(83, 23);
            this.btnRecord.TabIndex = 22;
            this.btnRecord.Text = "Record";
            this.btnRecord.UseVisualStyleBackColor = true;
            this.btnRecord.Click += new System.EventHandler(this.btnRecord_Click);
            // 
            // listGPU
            // 
            this.listGPU.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.listGPU.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.listGPU.FormattingEnabled = true;
            this.listGPU.Location = new System.Drawing.Point(71, 117);
            this.listGPU.MaximumSize = new System.Drawing.Size(300, 0);
            this.listGPU.Name = "listGPU";
            this.listGPU.Size = new System.Drawing.Size(296, 21);
            this.listGPU.TabIndex = 21;
            this.listGPU.SelectedIndexChanged += new System.EventHandler(this.listGPU_SelectedIndexChanged);
            // 
            // linkLabel1
            // 
            this.linkLabel1.AutoSize = true;
            this.linkLabel1.LinkColor = System.Drawing.Color.Red;
            this.linkLabel1.Location = new System.Drawing.Point(6, 454);
            this.linkLabel1.Name = "linkLabel1";
            this.linkLabel1.Size = new System.Drawing.Size(179, 13);
            this.linkLabel1.TabIndex = 20;
            this.linkLabel1.TabStop = true;
            this.linkLabel1.Text = "https://github.com/genrege/Mandel";
            this.linkLabel1.LinkClicked += new System.Windows.Forms.LinkLabelLinkClickedEventHandler(this.linkLabel1_LinkClicked);
            // 
            // btnResetJulia
            // 
            this.btnResetJulia.Location = new System.Drawing.Point(9, 363);
            this.btnResetJulia.Name = "btnResetJulia";
            this.btnResetJulia.Size = new System.Drawing.Size(83, 23);
            this.btnResetJulia.TabIndex = 19;
            this.btnResetJulia.Text = "Reset Julia";
            this.btnResetJulia.UseVisualStyleBackColor = true;
            this.btnResetJulia.Click += new System.EventHandler(this.btnResetJulia_Click);
            // 
            // upDown1
            // 
            this.upDown1.Location = new System.Drawing.Point(223, 13);
            this.upDown1.Name = "upDown1";
            this.upDown1.Size = new System.Drawing.Size(45, 20);
            this.upDown1.TabIndex = 18;
            this.upDown1.ValueChanged += new System.EventHandler(this.upDown1_ValueChanged);
            // 
            // cbWhichSet
            // 
            this.cbWhichSet.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cbWhichSet.FormattingEnabled = true;
            this.cbWhichSet.Items.AddRange(new object[] {
            "Mandelbrot",
            "Julia",
            "Buddha",
            "Anti-Buddha",
            "Mandelbrot Experimental API",
            "Julia Experimental API",
            "Buddha Experimental API",
            "Anti-Buddha Experimental API",
            "Special Experimental API"});
            this.cbWhichSet.Location = new System.Drawing.Point(9, 12);
            this.cbWhichSet.Name = "cbWhichSet";
            this.cbWhichSet.Size = new System.Drawing.Size(208, 21);
            this.cbWhichSet.TabIndex = 17;
            this.cbWhichSet.SelectedIndexChanged += new System.EventHandler(this.cbWhichSet_SelectedIndexChanged);
            // 
            // btnSaveBMP
            // 
            this.btnSaveBMP.Location = new System.Drawing.Point(98, 287);
            this.btnSaveBMP.Name = "btnSaveBMP";
            this.btnSaveBMP.Size = new System.Drawing.Size(83, 23);
            this.btnSaveBMP.TabIndex = 16;
            this.btnSaveBMP.Text = "Save Image...";
            this.btnSaveBMP.UseVisualStyleBackColor = true;
            this.btnSaveBMP.Click += new System.EventHandler(this.btnSaveBMP_Click);
            // 
            // btnResetZoom
            // 
            this.btnResetZoom.Location = new System.Drawing.Point(9, 392);
            this.btnResetZoom.Name = "btnResetZoom";
            this.btnResetZoom.Size = new System.Drawing.Size(83, 23);
            this.btnResetZoom.TabIndex = 15;
            this.btnResetZoom.Text = "Reset Zoom";
            this.btnResetZoom.UseVisualStyleBackColor = true;
            this.btnResetZoom.Click += new System.EventHandler(this.btnReset_Click);
            // 
            // btnSave
            // 
            this.btnSave.Location = new System.Drawing.Point(9, 316);
            this.btnSave.Name = "btnSave";
            this.btnSave.Size = new System.Drawing.Size(83, 23);
            this.btnSave.TabIndex = 14;
            this.btnSave.Text = "Save...";
            this.btnSave.UseVisualStyleBackColor = true;
            this.btnSave.Click += new System.EventHandler(this.btnSave_Click);
            // 
            // btnLoad
            // 
            this.btnLoad.Location = new System.Drawing.Point(9, 287);
            this.btnLoad.Name = "btnLoad";
            this.btnLoad.Size = new System.Drawing.Size(83, 23);
            this.btnLoad.TabIndex = 13;
            this.btnLoad.Text = "Load...";
            this.btnLoad.UseVisualStyleBackColor = true;
            this.btnLoad.Click += new System.EventHandler(this.btnLoad_Click);
            // 
            // txtGeneral
            // 
            this.txtGeneral.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txtGeneral.BackColor = System.Drawing.Color.Navy;
            this.txtGeneral.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.txtGeneral.ForeColor = System.Drawing.Color.Yellow;
            this.txtGeneral.Location = new System.Drawing.Point(9, 256);
            this.txtGeneral.Name = "txtGeneral";
            this.txtGeneral.Size = new System.Drawing.Size(358, 13);
            this.txtGeneral.TabIndex = 12;
            // 
            // sliderMin
            // 
            this.sliderMin.Location = new System.Drawing.Point(133, 56);
            this.sliderMin.Name = "sliderMin";
            this.sliderMin.Size = new System.Drawing.Size(57, 20);
            this.sliderMin.TabIndex = 11;
            this.sliderMin.Text = "0";
            this.sliderMin.TextChanged += new System.EventHandler(this.sliderMin_TextChanged);
            // 
            // sliderMax
            // 
            this.sliderMax.Location = new System.Drawing.Point(196, 56);
            this.sliderMax.Name = "sliderMax";
            this.sliderMax.Size = new System.Drawing.Size(57, 20);
            this.sliderMax.TabIndex = 10;
            this.sliderMax.Text = "4000";
            this.sliderMax.TextChanged += new System.EventHandler(this.sliderMax_TextChanged);
            // 
            // txtBounds
            // 
            this.txtBounds.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txtBounds.BackColor = System.Drawing.Color.Navy;
            this.txtBounds.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.txtBounds.ForeColor = System.Drawing.Color.Yellow;
            this.txtBounds.Location = new System.Drawing.Point(9, 218);
            this.txtBounds.Name = "txtBounds";
            this.txtBounds.Size = new System.Drawing.Size(358, 13);
            this.txtBounds.TabIndex = 9;
            // 
            // txtYMax
            // 
            this.txtYMax.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txtYMax.BackColor = System.Drawing.Color.Navy;
            this.txtYMax.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.txtYMax.ForeColor = System.Drawing.Color.Yellow;
            this.txtYMax.Location = new System.Drawing.Point(9, 199);
            this.txtYMax.Name = "txtYMax";
            this.txtYMax.Size = new System.Drawing.Size(358, 13);
            this.txtYMax.TabIndex = 8;
            this.txtYMax.TextChanged += new System.EventHandler(this.textBox1_TextChanged);
            // 
            // txtYMin
            // 
            this.txtYMin.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txtYMin.BackColor = System.Drawing.Color.Navy;
            this.txtYMin.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.txtYMin.ForeColor = System.Drawing.Color.Yellow;
            this.txtYMin.Location = new System.Drawing.Point(9, 180);
            this.txtYMin.Name = "txtYMin";
            this.txtYMin.Size = new System.Drawing.Size(358, 13);
            this.txtYMin.TabIndex = 7;
            // 
            // txtXMax
            // 
            this.txtXMax.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txtXMax.BackColor = System.Drawing.Color.Navy;
            this.txtXMax.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.txtXMax.ForeColor = System.Drawing.Color.Yellow;
            this.txtXMax.Location = new System.Drawing.Point(9, 161);
            this.txtXMax.Name = "txtXMax";
            this.txtXMax.Size = new System.Drawing.Size(358, 13);
            this.txtXMax.TabIndex = 6;
            // 
            // txtXMin
            // 
            this.txtXMin.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txtXMin.BackColor = System.Drawing.Color.Navy;
            this.txtXMin.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.txtXMin.ForeColor = System.Drawing.Color.Yellow;
            this.txtXMin.Location = new System.Drawing.Point(9, 142);
            this.txtXMin.Name = "txtXMin";
            this.txtXMin.Size = new System.Drawing.Size(358, 13);
            this.txtXMin.TabIndex = 5;
            // 
            // checkBox1
            // 
            this.checkBox1.AutoSize = true;
            this.checkBox1.Checked = true;
            this.checkBox1.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBox1.ForeColor = System.Drawing.SystemColors.ButtonHighlight;
            this.checkBox1.Location = new System.Drawing.Point(9, 119);
            this.checkBox1.Name = "checkBox1";
            this.checkBox1.Size = new System.Drawing.Size(49, 17);
            this.checkBox1.TabIndex = 4;
            this.checkBox1.Text = "GPU";
            this.checkBox1.UseVisualStyleBackColor = true;
            this.checkBox1.CheckedChanged += new System.EventHandler(this.checkBox1_CheckedChanged);
            // 
            // trackBarMaxIterations
            // 
            this.trackBarMaxIterations.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.trackBarMaxIterations.Location = new System.Drawing.Point(9, 82);
            this.trackBarMaxIterations.Maximum = 16000;
            this.trackBarMaxIterations.Minimum = 1;
            this.trackBarMaxIterations.Name = "trackBarMaxIterations";
            this.trackBarMaxIterations.Size = new System.Drawing.Size(364, 45);
            this.trackBarMaxIterations.TabIndex = 3;
            this.trackBarMaxIterations.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarMaxIterations.Value = 1;
            this.trackBarMaxIterations.Scroll += new System.EventHandler(this.trackBarMaxIterations_Scroll);
            // 
            // txtMouseCoords
            // 
            this.txtMouseCoords.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txtMouseCoords.BackColor = System.Drawing.Color.Navy;
            this.txtMouseCoords.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.txtMouseCoords.ForeColor = System.Drawing.Color.Yellow;
            this.txtMouseCoords.Location = new System.Drawing.Point(9, 237);
            this.txtMouseCoords.Name = "txtMouseCoords";
            this.txtMouseCoords.Size = new System.Drawing.Size(358, 13);
            this.txtMouseCoords.TabIndex = 0;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.ForeColor = System.Drawing.Color.White;
            this.label1.Location = new System.Drawing.Point(6, 40);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(73, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Max Iterations";
            // 
            // timer1
            // 
            this.timer1.Enabled = true;
            this.timer1.Interval = 1;
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // txtMaxIterations
            // 
            this.txtMaxIterations.Location = new System.Drawing.Point(9, 56);
            this.txtMaxIterations.Name = "txtMaxIterations";
            this.txtMaxIterations.Size = new System.Drawing.Size(120, 20);
            this.txtMaxIterations.TabIndex = 27;
            this.txtMaxIterations.ValueChanged += new System.EventHandler(this.txtMaxIterations_ValueChanged);
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1138, 654);
            this.Controls.Add(this.mainSplitter);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "MainForm";
            this.Text = "Mandelbrot Viewer";
            this.Load += new System.EventHandler(this.MainForm_Load);
            this.mainSplitter.Panel1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.mainSplitter)).EndInit();
            this.mainSplitter.ResumeLayout(false);
            this.SplitControlContainer.Panel1.ResumeLayout(false);
            this.SplitControlContainer.Panel1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.SplitControlContainer)).EndInit();
            this.SplitControlContainer.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.upDown1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarMaxIterations)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.txtMaxIterations)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.SplitContainer mainSplitter;
        private System.Windows.Forms.TextBox txtMouseCoords;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TrackBar trackBarMaxIterations;
        private System.Windows.Forms.SplitContainer SplitControlContainer;
        private System.Windows.Forms.CheckBox checkBox1;
        private System.Windows.Forms.TextBox txtXMax;
        private System.Windows.Forms.TextBox txtXMin;
        private System.Windows.Forms.TextBox txtYMax;
        private System.Windows.Forms.TextBox txtYMin;
        private System.Windows.Forms.TextBox txtBounds;
        private System.Windows.Forms.TextBox sliderMax;
        private System.Windows.Forms.TextBox sliderMin;
        private System.Windows.Forms.TextBox txtGeneral;
        private System.Windows.Forms.Button btnSave;
        private System.Windows.Forms.Button btnLoad;
        private System.Windows.Forms.Button btnResetZoom;
        private System.Windows.Forms.Button btnSaveBMP;
        private System.Windows.Forms.ComboBox cbWhichSet;
        private System.Windows.Forms.NumericUpDown upDown1;
        private System.Windows.Forms.Button btnResetJulia;
        private System.Windows.Forms.LinkLabel linkLabel1;
        private System.Windows.Forms.ComboBox listGPU;
        private System.Windows.Forms.Button btnRecord;
        private System.Windows.Forms.Button btnReplay;
        private System.Windows.Forms.Button btnSaveRecording;
        private System.Windows.Forms.Button btnLoadRec;
        private System.Windows.Forms.Timer timer1;
        private System.Windows.Forms.CheckBox checkLoop;
        private System.Windows.Forms.NumericUpDown txtMaxIterations;
    }
}

