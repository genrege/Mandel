using System.Windows.Forms;

namespace MandelbrotViewer
{
    partial class RenderPanel
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.button1 = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // RenderPanel
            // 
            this.Size = new System.Drawing.Size(400, 356);
            this.Paint += new System.Windows.Forms.PaintEventHandler(this.RenderPanel_Paint);
            this.MouseDoubleClick += new System.Windows.Forms.MouseEventHandler(this.RenderPanel_MouseDoubleClick);
            this.MouseDown += new System.Windows.Forms.MouseEventHandler(this.RenderPanel_MouseDown);
            this.MouseMove += new System.Windows.Forms.MouseEventHandler(this.RenderPanel_MouseMove);
            this.MouseUp += new System.Windows.Forms.MouseEventHandler(this.RenderPanel_MouseUp);
            this.Resize += new System.EventHandler(this.RenderPanel_Resize);
            this.ResumeLayout(false);

        }

        #endregion

        private Button button1;
    }
}
