#include <iostream>
#include <math.h>

//__global__声明该函数为需要在GPU上计算的核函数
__global__ void add(int n, float *x, float *y)
{
    for (int i=0;i<n;i++)
        y[i] = x[i] + y[i];
}

int main()
{
    int N = 1<<20;
    float *x,*y;

    //在GPU上开辟内存
    cudaMallocManaged(&x,N*sizeof(float));
    cudaMallocManaged(&y,N*sizeof(float));

    // 初始化
    for (int i=0;i<N;i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // 在GPU上计算
    add<<<1,1>>>(N,x,y);

    // 在访问Host之前,先等待GPU的运算结束
    cudaDeviceSynchronize();

    //检查误差:数组y所有的值都应该为3
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error:" << maxError << std::endl;

    // 释放内存
    cudaFree(x);
    cudaFree(y);
    
    return 0;
}