

#include <limits>
#include <string>
#include <cmath>
#include <queue>
#include "ImageStitcherTop.H"

using namespace std;

#define LAZY_READ 1


void calc_feature()
{
    GuardedTimer tm("calc_feature()");
    feats.resize(imgs.size());
    keypoints.resize(imgs.size());
    // detect feature
  #pragma omp parallel for schedule(dynamic)
    REP(k, (int)imgs.size())
    {
        imgs[k].load();
        feats[k] = feature_det->detect_feature(*imgs[k].img);
        if (LAZY_READ)
        {
            imgs[k].release();
        }
        if (feats[k].size() == 0)
        {
            //error_exit(ssprintf("Cannot find feature in image %d!\n", k));
        }
        print_debug("Image %d has %lu features\n", k, feats[k].size());
        keypoints[k].resize(feats[k].size());
        REP(i, feats[k].size())
        {
            keypoints[k][i] = feats[k][i].coor;
        }
    }
}
