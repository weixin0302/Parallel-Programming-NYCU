__kernel void convolution(__global float *inputImage, __global float *outputImage, __global float *filter,
                        int imageHeight, int imageWidth, int filterWidth) 
{
    int row = get_global_id(0) /imageWidth;
    int col = get_global_id(0) % imageWidth;
    int halffilterSize = filterWidth / 2;
    int k, l;
    int sum = 0;

    int k_start = -halffilterSize + row >= 0 ? -halffilterSize : 0;
    int k_end = halffilterSize + row < imageHeight ? halffilterSize : halffilterSize + row - imageHeight - 1;
    int l_start = -halffilterSize + col >= 0 ? -halffilterSize : 0;
    int l_end = halffilterSize + col < imageWidth ? halffilterSize : halffilterSize + col - imageWidth - 1;


    for (k = k_start; k <= k_end; k++)
    {
        // for (l = l_start; l <= l_end; l++)
        // {
        l = l_start;
        sum += inputImage[(row + k) * imageWidth + col + l] * filter[(k + halffilterSize) * filterWidth + l++ + halffilterSize];
        sum += inputImage[(row + k) * imageWidth + col + l] * filter[(k + halffilterSize) * filterWidth + l++ + halffilterSize];
        sum += inputImage[(row + k) * imageWidth + col + l] * filter[(k + halffilterSize) * filterWidth + l + halffilterSize];
        // }
    }
    outputImage[row * imageWidth + col] = sum;
}
