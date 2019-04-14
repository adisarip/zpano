
#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include <cassert>
#include "ImageStitcherTop.h"

#ifdef DISABLE_JPEG
#define IMGFILE(x) #x ".png"
#else
#define IMGFILE(x) #x ".jpg"
#endif

using namespace std;


void work(int argc, char* argv[])
{
    vector<string> imgs;
    REPL(i, 1, argc)
    {
        imgs.emplace_back(argv[i]);
    }
    Mat32f res;

    //Stitcher p(move(imgs));
    //res = p.build();
    calc_feature();
}

/*
void init_config()
{
#define CFG(x) x = Config.get(#x)
    const char* config_file = "config.cfg";
    ConfigParser Config(config_file);
    CFG(CYLINDER);
    CFG(TRANS);
    CFG(ESTIMATE_CAMERA);
    CFG(SIFT_WORKING_SIZE);
    
    if (int(CYLINDER) + int(TRANS) + int(ESTIMATE_CAMERA) >= 2)
    {
        error_exit("You set two many modes...\n");
    }

    if (CYLINDER)
    {
        print_debug("Run with cylinder mode.\n");
    }
    else if (TRANS)
    {
        print_debug("Run with translation mode.\n");
    }
    else if (ESTIMATE_CAMERA)
    {
        print_debug("Run with camera estimation mode.\n");
    }
    else
    {
        print_debug("Run with naive mode.\n");
    }

    CFG(ORDERED_INPUT);
    if (!ORDERED_INPUT && !ESTIMATE_CAMERA)
    {
        error_exit("Require ORDERED_INPUT under this mode!\n");
    }

    //CFG(CROP);
    //CFG(STRAIGHTEN);
    //CFG(FOCAL_LENGTH);
    //CFG(MAX_OUTPUT_SIZE);
    //CFG(LAZY_READ);    // TODO in cyl mode

    //CFG(SIFT_WORKING_SIZE);
    CFG(NUM_OCTAVE);
    CFG(NUM_SCALE);
    CFG(SCALE_FACTOR);
    CFG(GAUSS_SIGMA);
    CFG(GAUSS_WINDOW_FACTOR);
    CFG(JUDGE_EXTREMA_DIFF_THRES);
    CFG(CONTRAST_THRES);
    CFG(PRE_COLOR_THRES);
    CFG(EDGE_RATIO);
    CFG(CALC_OFFSET_DEPTH);
    CFG(OFFSET_THRES);
    CFG(ORI_RADIUS);
    CFG(ORI_HIST_SMOOTH_COUNT);
    CFG(DESC_HIST_SCALE_FACTOR);
    CFG(DESC_INT_FACTOR);
    CFG(MATCH_REJECT_NEXT_RATIO);
    CFG(RANSAC_ITERATIONS);
    CFG(RANSAC_INLIER_THRES);
    CFG(INLIER_IN_MATCH_RATIO);
    CFG(INLIER_IN_POINTS_RATIO);
    CFG(SLOPE_PLAIN);
    CFG(LM_LAMBDA);
    CFG(MULTIPASS_BA);
    CFG(MULTIBAND);
#undef CFG
}
*/

int main(int argc, char* argv[])
{
    if (argc <= 2)
    {
        error_exit("Need at least two images to stitch.\n");
    }

    //init_config();
    work(argc, argv);
}
