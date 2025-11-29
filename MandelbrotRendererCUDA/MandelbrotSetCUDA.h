#pragma once
#include <string>

class __declspec(dllexport) mbrot_cuda
{
public:
	mbrot_cuda(int gpu_index = 0);
	~mbrot_cuda();
	static size_t device_count();
	static std::string device_name(size_t index);
	unsigned int* alloc_cuda(int size);
	unsigned int* alloc_palette(int size);
	void render_mbrot(int wx, int wy, double x0, double x1, double y0, double y1, int max_iter, unsigned int* r, unsigned* palette = nullptr, unsigned palette_index = 0);
	void render_julia(int wx, int wy, double x0, double x1, double y0, double y1, double kr, double ki, int max_iter, unsigned int* r, unsigned* palette = nullptr, unsigned palette_index = 0);
	void render_buddha(bool anti_buddha, int wx, int wy, double x0, double x1, double y0, double y1, int max_iter, unsigned int* r);

private:
	unsigned int* m_dev_r;
	int m_csize;

	unsigned int* m_dev_p;
	int m_psize;
	int last_cuda_error_;
};
