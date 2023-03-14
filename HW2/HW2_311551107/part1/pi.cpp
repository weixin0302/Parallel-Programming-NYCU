#include <iostream>
#include <pthread.h>
#include <sstream>
#include <stdlib.h>
#include <time.h>
#include <mutex>
#include <random>

using namespace std;

int thread_num;
long long tosses;
long long circle_points;
mutex g_mutex;

void *count_circles(void *arg)
{
    long long count = 0;
    long long points = tosses / thread_num;
    double x, y, origin_dist;
    unsigned int seed = 1;
    for (int i = 0; i < points; i++)
    {
        x = (double)rand_r(&seed)/RAND_MAX;
        y = (double)rand_r(&seed)/RAND_MAX;
        origin_dist = x * x + y * y;
        if (origin_dist <= 1)
            count++;
    }
    g_mutex.lock();
    circle_points += count;
    g_mutex.unlock();
    return NULL;
}

int main(int argc, char *argv[])
{
    istringstream ss1(argv[1]);
    istringstream ss2(argv[2]);
    ss1 >> thread_num;
    ss2 >> tosses;
    pthread_t threads[thread_num];
    for (int i = 0; i < thread_num; i++)
    {
        pthread_create(&threads[i], NULL, count_circles, NULL);
    }

    for (int i = 0; i < thread_num; i++)
    {
        pthread_join(threads[i], NULL);
    }

    double pi = double(4 * circle_points) / tosses;

    cout << pi << endl;

    return 0;
}