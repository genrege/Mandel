#pragma once


class __declspec(dllexport) mbrot_cuda
{
public:
	mbrot_cuda();
	~mbrot_cuda();
	unsigned int* alloc_cuda(int size);
	unsigned int* alloc_palette(int size);
	void render_mbrot(int wx, int wy, double x0, double x1, double y0, double y1, int max_iter, unsigned int* r, unsigned* palette = nullptr, unsigned palette_index = 0);
	void render_julia(int wx, int wy, double x0, double x1, double y0, double y1, double kr, double ki, int max_iter, unsigned int* r, unsigned* palette = nullptr, unsigned palette_index = 0);

private:
	unsigned int* m_dev_r;
	int m_csize;

	unsigned int* m_dev_p;
	int m_psize;
};
