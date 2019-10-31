#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "lodepng.h"
#include "main.h"

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " [image1] [image2]\n";
        return 1;
    }

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
    //initialize new image
    unsigned char *left_image = (unsigned char *) malloc(width*height);
    unsigned char *right_image = (unsigned char *) malloc(width*height);

    //downscaling and conversion to grayscale
#pragma omp parallel sections
{
    #pragma omp section
    {resize_and_grayscale(left_decoded, left_image, left_h, left_w);}
    #pragma omp section
    {resize_and_grayscale(right_decoded, right_image, right_h, right_w);}
}
    unsigned char *left_disparity = (unsigned char *) malloc(width*height);
    unsigned char *right_disparity = (unsigned char *) malloc(width*height);

    zncc(left_image, right_image, left_disparity, -Max_Disp, 0, height, width);
    zncc(right_image, left_image, right_disparity, -Max_Disp, Max_Disp, height, width);

    // Disparity map left to right
    // if(lodepng_encode_file("left_map.png", left_disparity, width, height, LCT_GREY, 8))
    // {
    //     printf("Error encoding image!\n");
    // }
    
    // // Disparity map right to left
    // if(lodepng_encode_file("right_map.png", right_disparity, width, height, LCT_GREY, 8))
    // {
    //     printf("Error encoding image!\n");
    // }

    cross_checking(left_disparity, right_disparity, height*width);
    occlusion_filling(left_disparity, height, width);
    
	auto duration =  Time::now() - start;
    std::cout << "\nTOTAL TIME TAKEN: " << duration.count()/1000000000.0 << "\n";

    // Disparity map left 
    if(lodepng_encode_file("disparity_map.png", left_disparity, width, height, LCT_GREY, 8))
    {
        printf("Error encoding image!\n");
    }
    
    free(left_decoded);
    free(left_image);
    free(left_disparity);
    free(right_decoded);
    free(right_image);
    free(right_disparity);

    return 0;
}

void resize_and_grayscale(const unsigned char *original_image, unsigned char *new_image, 
                          unsigned int h, unsigned int w)
{
    unsigned int new_w = w/4, new_h = h/4;

    //iterate new_image
    for(unsigned int i = 0; i < new_h; i++)
    {
        for(unsigned int j = 0; j < new_w; j++)
            {
            unsigned char R = original_image[i*w*16+j*16+0];
            unsigned char G = original_image[i*w*16+j*16+1];
            unsigned char B = original_image[i*w*16+j*16+2];

            new_image[i*new_w+j] = 0.2126*R + 0.7152*G + 0.0722*B;
            }
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
#pragma omp parallel private(left_mean, right_mean, left_stdd, right_stdd, current_value, best_value, disparity)
{
#pragma omp for collapse(2)
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
#pragma omp barrier
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
#pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        if(abs(left_disparity[i]-right_disparity[i]) > Threshold)
            left_disparity[i] = 0;
    }
}

void occlusion_filling(unsigned char *disparity_map, unsigned int h, unsigned int w)
{
#pragma omp parallel for collapse(2)
    for(int i = 0; i < h; i++)
    {
        for(int j = 0; j < w; j++)
        {
            if(disparity_map[i*w+j] == 0)
                disparity_map[i*w+j] = mean(disparity_map, i, j, h, w);
        }
    }
}