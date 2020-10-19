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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.mainSplitter = new System.Windows.Forms.SplitContainer();
            this.SplitControlContainer = new System.Windows.Forms.SplitContainer();
            this.cbWhichSet = new System.Windows.Forms.ComboBox();
            this.btnSaveBMP = new System.Windows.Forms.Button();
            this.btnReset = new System.Windows.Forms.Button();
            this.btnSave = new System.Windows.Forms.Button();
            this.btnLoad = new System.Windows.Forms.Button();
            this.txtValueUnderMouse = new System.Windows.Forms.TextBox();
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
            this.txtMaxIterations = new System.Windows.Forms.TextBox();
            ((System.ComponentModel.ISupportInitialize)(this.mainSplitter)).BeginInit();
            this.mainSplitter.Panel1.SuspendLayout();
            this.mainSplitter.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.SplitControlContainer)).BeginInit();
            this.SplitControlContainer.Panel1.SuspendLayout();
            this.SplitControlContainer.Panel2.SuspendLayout();
            this.SplitControlContainer.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarMaxIterations)).BeginInit();
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
            this.mainSplitter.Size = new System.Drawing.Size(840, 654);
            this.mainSplitter.SplitterDistance = 280;
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
            this.SplitControlContainer.Panel1.Controls.Add(this.cbWhichSet);
            this.SplitControlContainer.Panel1.Controls.Add(this.btnSaveBMP);
            this.SplitControlContainer.Panel1.Controls.Add(this.btnReset);
            this.SplitControlContainer.Panel1.Controls.Add(this.btnSave);
            this.SplitControlContainer.Panel1.Controls.Add(this.btnLoad);
            this.SplitControlContainer.Panel1.Controls.Add(this.txtValueUnderMouse);
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
            this.SplitControlContainer.Panel1.Controls.Add(this.txtMaxIterations);
            this.SplitControlContainer.Panel1.Paint += new System.Windows.Forms.PaintEventHandler(this.SplitControlContainer_Panel1_Paint);
            // 
            // SplitControlContainer.Panel2
            // 
            this.SplitControlContainer.Size = new System.Drawing.Size(280, 654);
            this.SplitControlContainer.SplitterDistance = 467;
            this.SplitControlContainer.TabIndex = 4;
            // 
            // cbWhichSet
            // 
            this.cbWhichSet.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cbWhichSet.FormattingEnabled = true;
            this.cbWhichSet.Items.AddRange(new object[] {
            "Mandelbrot",
            "Julia (right mouse to control)",
            "Buddha",
            "Anti-Buddha",
            "Mandelbrot Experimental API",
            "Julia Experimental API",
            "Buddha Experimental API",
            "Anti-Buddha Experimental API"});
            this.cbWhichSet.Location = new System.Drawing.Point(9, 12);
            this.cbWhichSet.Name = "cbWhichSet";
            this.cbWhichSet.Size = new System.Drawing.Size(226, 21);
            this.cbWhichSet.TabIndex = 17;
            this.cbWhichSet.SelectedIndexChanged += new System.EventHandler(this.cbWhichSet_SelectedIndexChanged);
            // 
            // btnSaveBMP
            // 
            this.btnSaveBMP.Location = new System.Drawing.Point(12, 328);
            this.btnSaveBMP.Name = "btnSaveBMP";
            this.btnSaveBMP.Size = new System.Drawing.Size(83, 23);
            this.btnSaveBMP.TabIndex = 16;
            this.btnSaveBMP.Text = "Save Image...";
            this.btnSaveBMP.UseVisualStyleBackColor = true;
            this.btnSaveBMP.Click += new System.EventHandler(this.btnSaveBMP_Click);
            // 
            // btnReset
            // 
            this.btnReset.Location = new System.Drawing.Point(190, 299);
            this.btnReset.Name = "btnReset";
            this.btnReset.Size = new System.Drawing.Size(83, 23);
            this.btnReset.TabIndex = 15;
            this.btnReset.Text = "Reset";
            this.btnReset.UseVisualStyleBackColor = true;
            this.btnReset.Click += new System.EventHandler(this.btnReset_Click);
            // 
            // btnSave
            // 
            this.btnSave.Location = new System.Drawing.Point(101, 299);
            this.btnSave.Name = "btnSave";
            this.btnSave.Size = new System.Drawing.Size(83, 23);
            this.btnSave.TabIndex = 14;
            this.btnSave.Text = "Save...";
            this.btnSave.UseVisualStyleBackColor = true;
            this.btnSave.Click += new System.EventHandler(this.btnSave_Click);
            // 
            // btnLoad
            // 
            this.btnLoad.Location = new System.Drawing.Point(12, 299);
            this.btnLoad.Name = "btnLoad";
            this.btnLoad.Size = new System.Drawing.Size(83, 23);
            this.btnLoad.TabIndex = 13;
            this.btnLoad.Text = "Load...";
            this.btnLoad.UseVisualStyleBackColor = true;
            this.btnLoad.Click += new System.EventHandler(this.btnLoad_Click);
            // 
            // txtValueUnderMouse
            // 
            this.txtValueUnderMouse.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txtValueUnderMouse.BackColor = System.Drawing.Color.Navy;
            this.txtValueUnderMouse.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.txtValueUnderMouse.ForeColor = System.Drawing.Color.Yellow;
            this.txtValueUnderMouse.Location = new System.Drawing.Point(9, 268);
            this.txtValueUnderMouse.Name = "txtValueUnderMouse";
            this.txtValueUnderMouse.Size = new System.Drawing.Size(259, 13);
            this.txtValueUnderMouse.TabIndex = 12;
            // 
            // sliderMin
            // 
            this.sliderMin.Location = new System.Drawing.Point(115, 56);
            this.sliderMin.Name = "sliderMin";
            this.sliderMin.Size = new System.Drawing.Size(57, 20);
            this.sliderMin.TabIndex = 11;
            this.sliderMin.Text = "0";
            this.sliderMin.TextChanged += new System.EventHandler(this.sliderMin_TextChanged);
            // 
            // sliderMax
            // 
            this.sliderMax.Location = new System.Drawing.Point(178, 56);
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
            this.txtBounds.Location = new System.Drawing.Point(9, 230);
            this.txtBounds.Name = "txtBounds";
            this.txtBounds.Size = new System.Drawing.Size(259, 13);
            this.txtBounds.TabIndex = 9;
            // 
            // txtYMax
            // 
            this.txtYMax.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txtYMax.BackColor = System.Drawing.Color.Navy;
            this.txtYMax.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.txtYMax.ForeColor = System.Drawing.Color.Yellow;
            this.txtYMax.Location = new System.Drawing.Point(9, 211);
            this.txtYMax.Name = "txtYMax";
            this.txtYMax.Size = new System.Drawing.Size(259, 13);
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
            this.txtYMin.Location = new System.Drawing.Point(9, 192);
            this.txtYMin.Name = "txtYMin";
            this.txtYMin.Size = new System.Drawing.Size(259, 13);
            this.txtYMin.TabIndex = 7;
            // 
            // txtXMax
            // 
            this.txtXMax.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txtXMax.BackColor = System.Drawing.Color.Navy;
            this.txtXMax.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.txtXMax.ForeColor = System.Drawing.Color.Yellow;
            this.txtXMax.Location = new System.Drawing.Point(9, 173);
            this.txtXMax.Name = "txtXMax";
            this.txtXMax.Size = new System.Drawing.Size(259, 13);
            this.txtXMax.TabIndex = 6;
            // 
            // txtXMin
            // 
            this.txtXMin.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txtXMin.BackColor = System.Drawing.Color.Navy;
            this.txtXMin.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.txtXMin.ForeColor = System.Drawing.Color.Yellow;
            this.txtXMin.Location = new System.Drawing.Point(9, 154);
            this.txtXMin.Name = "txtXMin";
            this.txtXMin.Size = new System.Drawing.Size(259, 13);
            this.txtXMin.TabIndex = 5;
            // 
            // checkBox1
            // 
            this.checkBox1.AutoSize = true;
            this.checkBox1.Checked = true;
            this.checkBox1.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBox1.ForeColor = System.Drawing.SystemColors.ButtonHighlight;
            this.checkBox1.Location = new System.Drawing.Point(9, 131);
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
            this.trackBarMaxIterations.Location = new System.Drawing.Point(3, 82);
            this.trackBarMaxIterations.Maximum = 16000;
            this.trackBarMaxIterations.Minimum = 1;
            this.trackBarMaxIterations.Name = "trackBarMaxIterations";
            this.trackBarMaxIterations.Size = new System.Drawing.Size(232, 45);
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
            this.txtMouseCoords.Location = new System.Drawing.Point(9, 249);
            this.txtMouseCoords.Name = "txtMouseCoords";
            this.txtMouseCoords.Size = new System.Drawing.Size(259, 13);
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
            // txtMaxIterations
            // 
            this.txtMaxIterations.Location = new System.Drawing.Point(9, 56);
            this.txtMaxIterations.Name = "txtMaxIterations";
            this.txtMaxIterations.Size = new System.Drawing.Size(100, 20);
            this.txtMaxIterations.TabIndex = 2;
            this.txtMaxIterations.Text = "1024";
            this.txtMaxIterations.TextChanged += new System.EventHandler(this.txtMaxIterations_TextChanged);
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(840, 654);
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
            this.SplitControlContainer.Panel2.ResumeLayout(false);
            this.SplitControlContainer.Panel2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.SplitControlContainer)).EndInit();
            this.SplitControlContainer.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.trackBarMaxIterations)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.SplitContainer mainSplitter;
        private System.Windows.Forms.TextBox txtMouseCoords;
        private System.Windows.Forms.TextBox txtMaxIterations;
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
        private System.Windows.Forms.TextBox txtValueUnderMouse;
        private System.Windows.Forms.Button btnSave;
        private System.Windows.Forms.Button btnLoad;
        private System.Windows.Forms.Button btnReset;
        private System.Windows.Forms.Button btnSaveBMP;
        private System.Windows.Forms.ComboBox cbWhichSet;
        private System.Windows.Forms.MenuStrip menuMain;
    }
}

