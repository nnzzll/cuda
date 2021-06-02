#include<iostream>

int main()
{
    cudaDeviceProp prop;
    int count;

    cudaGetDeviceCount(&count);
    std::cout<<"GPU num:"<<count<<std::endl;
    cudaGetDeviceProperties(&prop,0);
    std::cout<<"Max threads/block:"<<prop.maxThreadsPerBlock<<std::endl;
    std::cout<<"Max threads/SM:"<<prop.maxThreadsPerMultiProcessor<<std::endl;
    std::cout<<"Max block/SM:"<<prop.maxBlocksPerMultiProcessor<<std::endl;
    std::cout<<"Max Grid size:"<<prop.maxGridSize[0]<<","<<prop.maxGridSize[1]<<","<<prop.maxGridSize[2]<<std::endl;
}