

#include <limits>
#include <string>
#include <cmath>
#include <queue>
#include "ImageStitcherTop.H"
using namespace std;

void StitcherTest::calc_feature(int imgIndex,
                                Mat32f* img_mat)
{
    GuardedTimer tm("calc_feature()");

    // detect feature
    feats[imgIndex] = feature_det->detect_feature(*img_mat);
    //print_debug("Image %d has %lu features\n", imgIndex, feats[imgIndex].size());
    keypoints[imgIndex].resize(feats[imgIndex].size());
    REP(i, feats[imgIndex].size())
    {
        keypoints[imgIndex][i] = feats[imgIndex][i].coor;
    }
}
