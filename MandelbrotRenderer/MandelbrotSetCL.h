#pragma once
#if defined(USE_OPENCL) && defined(DEBUG_CL)
#include <Cl/cl.h>

#include <vector>
#include <string>

namespace mandelbrot_cl
{
struct cl_platform_details
{
    std::string     name;
    std::string     vendor;
    std::string     version;
    std::string     profile;
    std::string     extensions;
};

std::vector<cl_platform_details> open_cl_platforms()
{
    cl_uint platform_count;
    clGetPlatformIDs(5, NULL, &platform_count);

    cl_platform_id* pids = new cl_platform_id[platform_count];
    cl_platform_info attributes[] = {CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS};
    constexpr auto attribute_count = sizeof(attributes) / sizeof(attributes[0]);

    std::vector<cl_platform_details> retval(platform_count);
    clGetPlatformIDs(platform_count, pids, NULL);


    for (cl_uint i = 0; i < platform_count; i++)
    {
        for (cl_uint j = 0; j < attribute_count; ++j)
        {
            size_t info_size;
            clGetPlatformInfo(pids[i], attributes[j], 0, NULL, &info_size);

            char* info = new char[info_size];
            clGetPlatformInfo(pids[i], attributes[j], info_size, info, NULL);

            switch (attributes[j]) {
            case CL_PLATFORM_NAME:
                retval[i].name = info; 
                break;
            case CL_PLATFORM_VENDOR:
                retval[i].vendor = info;
                break;
            case CL_PLATFORM_VERSION:
                retval[i].version = info;
                break;
            case CL_PLATFORM_PROFILE:
                retval[i].profile = info;
                break;
            case CL_PLATFORM_EXTENSIONS:
                retval[i].extensions = info;
                break;
            };


            delete[] info;
        }
    }


    delete[] pids;

    return retval;
}

}//namespace mandelbrot_cl


#endif //USE_OPENCL

