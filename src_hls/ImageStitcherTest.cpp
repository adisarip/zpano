
#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include <cassert>
#include "ImageStitcherTop.h"

using namespace std;


void work(int argc, char* argv[])
{
    vector<string> imgs;
    REPL(i, 1, argc)
    {
        imgs.emplace_back(argv[i]);
    }
    Mat32f res;

    calc_feature();
}


int main(int argc, char* argv[])
{
    if (argc <= 2)
    {
        error_exit("Need at least two images to stitch.\n");
    }

    //init_config();
    work(argc, argv);
}
