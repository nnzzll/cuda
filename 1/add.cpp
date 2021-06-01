#include <iostream>
#include <math.h>

void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main()
{
    int N = 1 << 20; //计算1M次
    float *x = new float[N];
    float *y = new float[N];

    //初始化
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    //在CPU上计算
    add(N, x, y);

    //检查误差:数组y所有的值都应该为3
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error:" << maxError << std::endl;

    //释放内存
    delete []x;
    delete []y;

    return 0;
}