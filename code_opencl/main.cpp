#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "lodepng.h"
#include "main.h"
#include <CL/cl.hpp>

// https://github.com/Dakkers/OpenCL-examples/blob/master/example01/main.cpp
// was used as a reference
std::string src_str = 
    "void kernel resize_and_grayscale(global const unsigned char *original_image, global unsigned char *new_image, const unsigned int w)"
    "{"
    "   int new_w = w/4;"
    "   int i = get_global_id(0);"
    "   int j = get_global_id(1);"
    ""
    "   int j_orig = (4*j-1*(j > 0));"
    "   int i_orig = (4*i-1*(i > 0));"
    "   unsigned char R = original_image[i_orig*w*4+j_orig*4+0];"
    "   unsigned char G = original_image[i_orig*w*4+j_orig*4+1];"
    "   unsigned char B = original_image[i_orig*w*4+j_orig*4+2];"
    ""
    "   new_image[i*new_w+j] = 0.2126*R + 0.7152*G + 0.0722*B;"
    "}";

int main(int argc, char *argv[])
{
    if  (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " [image1] [image2]\n";
        return 1;
    }

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size()==0) {
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    //In our case the NVIDIA cuda was found in platform 2
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    // get default device (CPUs, GPUs) of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    cl::Device default_device=all_devices[1];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
    std::cout << "CL_DEVICE_LOCAL_MEM_TYPE: " << default_device.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>() << "\n";
    std::cout << "CL_DEVICE_LOCAL_MEM_SIZE: " << default_device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << "\n";
    std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << default_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
    std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY: " << default_device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "\n";
    std::cout << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << default_device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() << "\n";
    std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << default_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
    //std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES: " << default_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>() << "\n";
    std::cout << "\n";
    cl::Context context({default_device});
    cl::Program::Sources sources;
    sources.push_back({src_str.c_str(), src_str.length()});

    cl::Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) 
    {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
        exit(1);
    }
    
    // set up kernels and vectors for GPU code
    cl::CommandQueue queue(context, default_device);
    cl::Kernel resize_and_grayscale = cl::Kernel(program, "resize_and_grayscale");

    auto start = Time::now();

    //output from decoding
    unsigned char *left_decoded, *right_decoded;
    //width and height for output image
    unsigned int left_w, left_h, right_w, right_h;

    if(lodepng_decode32_file(&left_decoded, &left_w, &left_h, argv[1]))
    {
         std::cerr << "Error reading file: \n" << argv[1];
        return 1;
    }

    if(lodepng_decode32_file(&right_decoded, &right_w, &right_h, argv[2]))
    {
         std::cerr << "Error reading file: \n" << argv[2];
        return 1;
    }

    if(left_w != right_w && left_h != right_h)
    {
         std::cerr << "Error, images are different sizes!" << argv[0];
        return 1;
    }

    unsigned int width = right_w/4, height = right_h/4;
    //allocate new image
    unsigned char *left_image = (unsigned char *) malloc(width*height);
    unsigned char *right_image = (unsigned char *) malloc(width*height);

    //allocate buffers on GPU memory
    cl::Buffer buffer_left_decoded(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * right_w * right_h);
    cl::Buffer buffer_right_decoded(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * right_w * right_h);
    cl::Buffer buffer_left_image(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * width * height);
    cl::Buffer buffer_right_image(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * width * height);

    // Push write commands to queue
    queue.enqueueWriteBuffer(buffer_left_decoded, CL_TRUE, 0, sizeof(unsigned char) * right_w * right_h, left_decoded);
    queue.enqueueWriteBuffer(buffer_right_decoded, CL_TRUE, 0, sizeof(unsigned char) * right_w * right_h, right_decoded);

    //Run the kernels
    resize_and_grayscale.setArg(0, buffer_left_decoded);
    resize_and_grayscale.setArg(1, buffer_left_image);
    resize_and_grayscale.setArg(2, left_w);

    queue.enqueueNDRangeKernel(resize_and_grayscale, cl::NullRange, cl::NDRange(504, 735), cl::NDRange(2, 3));

    //read buffer from GPU to host
    queue.enqueueReadBuffer(buffer_left_image, CL_TRUE, 0, sizeof(unsigned char) * width * height, left_image);
    queue.finish();

    if(lodepng_encode_file("left_image.png", left_image, width, height, LCT_GREY, 8))
    {
        printf("Error encoding image!\n");
    }

}
void zncc(const unsigned char *left_image, const unsigned char *right_image, 
                   unsigned char *disparity_map, int mindisp, int maxdisp, 
                   unsigned int h, unsigned int w)
{
    double left_mean, right_mean;
    double left_stdd, right_stdd;
    double current_value, best_value;
    int disparity;

    for(int i = 0; i < h; i++)
    {
        for(int j = 0; j < w; j++)
        {
            left_mean = mean(left_image, i, j, h, w);
            left_stdd = standard_deviation(left_image, left_mean, i, j, h, w);
            best_value = 0;
    
            for(int d = mindisp; d <= maxdisp; d++)
            {
                right_mean = mean(right_image, i, j+d, h, w);
                right_stdd = standard_deviation(right_image, right_mean, i, j+d, h, w);
                current_value = 0;
                for(int k = -Win_X/2; k <= Win_X/2; k++)
                {
                    for(int l = -Win_Y/2; l <= Win_Y/2; l++)
                    {
                        if( (i + k >= 0) && (i + k <= h) && (j+d + l >= 0) && (j+d + l <= w) && (j + l >= 0) && (j + l <= w) )
                            current_value += (left_image[(i+k)*w+j+l] - left_mean) * (right_image[(i+k)*w+j+d+l] - right_mean);
                    }
                }
                current_value /= (left_stdd*right_stdd);

                if(current_value > best_value)
                {
                    best_value = current_value;
                    disparity = d;
                }
            }
            //normalize data
            disparity = 255/65 * disparity;
            disparity_map[i*w+j] = (unsigned char) abs(disparity);
        }
    }
}

double mean(const unsigned char *image, int x, int y, unsigned int h, unsigned int w)
{
    double mean = 0;
    for(int i = -Win_X/2; i <= Win_X/2; i++)
    {
        for(int j = -Win_Y/2; j <= Win_Y/2; j++)
        {
            if( (x + i >= 0) && (x + i <= h) && (y + j >= 0) && (y + j <= w) )
                mean += image[(x+i)*w+y+j];
        }
    }
    return mean/(Win_X*Win_Y);
}

double standard_deviation(const unsigned char *image, double mean,int x, int y, unsigned int h, unsigned int w)
{
    double standard_deviation = 0;
    for(int i = -Win_X/2; i <= Win_X/2; i++)
    {
        for(int j = -Win_Y/2; j <= Win_Y/2; j++)
        {
            if( (x + i >= 0) && (x + i <= h) && (y + j >= 0) && (y + j <= w) )
                standard_deviation += pow((image[(x+i)*w+y+j] - mean), 2);
        }
    }
    return sqrt(standard_deviation);
}

void cross_checking(unsigned char *left_disparity, const unsigned char *right_disparity, unsigned int size)
{
    for(int i = 0; i < size; i++)
    {
        if(abs(left_disparity[i]-right_disparity[i]) > Threshold)
            left_disparity[i] = 0;
    }
}

void occlusion_filling(unsigned char *disparity_map, unsigned int h, unsigned int w)
{
    for(int i = 0; i < h; i++)
    {
        for(int j = 0; j < w; j++)
        {
            if(disparity_map[i*w+j] == 0)
                disparity_map[i*w+j] = mean(disparity_map, i, j, h, w);
        }
    }
}