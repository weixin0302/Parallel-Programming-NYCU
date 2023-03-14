#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"
#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    float *charFilter = (float *)malloc(filterWidth * filterWidth * sizeof(float));
    int new_filterWidth = filterWidth;
    int checkStart = 0;
    int checkEnd = filterWidth - 1;
    int check = 1;
    while(check == 1 && checkStart < checkEnd) {
        for (int i = 0; i < filterWidth && check == 1; i++)
                if(filter[checkStart * filterWidth + i] != 0) check = 0;  // upper
        for (int i = 0; i < filterWidth && check == 1; i++)
                if(filter[checkEnd * filterWidth + i] != 0) check = 0;  // lower
        for (int i = 0; i < filterWidth && check == 1; i++)
                if(filter[i * filterWidth + checkStart] != 0) check = 0;  // left
        for (int i = 0; i < filterWidth && check == 1; i++)
                if(filter[i * filterWidth + checkEnd] != 0) check = 0;  // right
        if (check == 1) new_filterWidth -= 2;
        checkStart++;
        checkEnd--;
    }
    int charFilter_start = (filterWidth - new_filterWidth) % 2 == 0 ? (filterWidth - new_filterWidth) / 2 : 0;
    for (register int i = 0; i < new_filterWidth; i++){
        for (register int j = 0; j < new_filterWidth; j++){
            charFilter[i * new_filterWidth + j] = filter[((charFilter_start + i) * filterWidth) + charFilter_start + j];
        } 
    }
    
    cl_int status;
    int filterSize = new_filterWidth * new_filterWidth;
    int imageSize = imageHeight * imageWidth ;

    cl_command_queue command = clCreateCommandQueue(*context, *device, 0, &status);
    cl_mem device_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize * sizeof(float) , NULL, &status);
    cl_mem device_input = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageSize * sizeof(float), NULL, &status);
    cl_mem device_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize * sizeof(float), NULL, &status);

    clEnqueueWriteBuffer(command, device_input, CL_TRUE, 0, imageSize * sizeof(float), inputImage, 0, NULL, NULL);
    clEnqueueWriteBuffer(command, device_filter, CL_TRUE, 0, filterSize * sizeof(float), charFilter, 0, NULL, NULL);

    cl_kernel kernel = clCreateKernel(*program, "convolution", status);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&device_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&device_output);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&device_filter);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&new_filterWidth);

    size_t globalItemSize = imageSize;
    size_t localItemSize = 8 * 8;
    clEnqueueNDRangeKernel(command, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);
    clEnqueueReadBuffer(command, device_output, CL_TRUE, 0, imageSize * sizeof(float), outputImage, 0, NULL, NULL);

    clReleaseCommandQueue(command);
    clReleaseMemObject(device_filter);
    clReleaseMemObject(device_input);
    clReleaseMemObject(device_output);
    clReleaseKernel(kernel);
    free(charFilter);
}