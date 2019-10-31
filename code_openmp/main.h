#ifndef MAIN_H
#define MAIN_H

#include <chrono>

//Window size for the ZNCC algorithm
static const int Win_X = 9;
static const int Win_Y = 9;
//Maximum amount of offset 
static const int Max_Disp = 65;
//Threshold value for cross checking
static const int Threshold = 8;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double> Duration;

//Resize image to 1/16 of the original size and transform image to 8 bit grayscale
void resize_and_grayscale(const unsigned char *original_image, unsigned char *new_image, 
                          unsigned int h, unsigned int w);
void zncc(const unsigned char *left_image, const unsigned char *right_image, unsigned char *disparity_map,
          int mindisp, int maxdisp, unsigned int h, unsigned int w);
//Mean for window around x, y with border checking
//x is height and y is width
double mean(const unsigned char *image, int x, int y, unsigned int h, unsigned int w);
//Deviation for a window with border checking
double standard_deviation(const unsigned char *image, double mean, int x, int y, unsigned int h, unsigned int w);
void cross_checking(unsigned char *left_disparity, const unsigned char *right_disparity, unsigned int size);
//assign each zero the mean of a surrounding window
void occlusion_filling(unsigned char *disparity_map, unsigned int h, unsigned int w);


#endif