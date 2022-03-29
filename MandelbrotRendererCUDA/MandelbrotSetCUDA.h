#pragma once


class __declspec(dllexport) mbrot_cuda
{
public:
	mbrot_cuda();
	~mbrot_cuda();
	unsigned int* alloc_cuda(int size);
	void render_mbrot(double x0, double x1, double y0, double y1, int wx, int wy, int max_iter, unsigned int* r);
	void render_julia(double x0, double x1, double y0, double y1, double kr, double ki, int wx, int wy, int max_iter, unsigned int* r);

private:
	unsigned int* m_dev_r;
	int m_csize;

};
