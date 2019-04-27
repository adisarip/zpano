
#ifndef _IMAGE_STITCHER_TOP_H
#define _IMAGE_STITCHER_TOP_H


#include <vector>
#include <memory>
#include "ImageUtils.H"


class StitcherTest
{
    public:
    // TOP Function
    // get feature descriptor and keypoints for each image
    void calc_feature(int imgIndex,
                    Mat32f* img_mat);

    // feature and keypoints of each image
    // keypoints only work as a storage of feats, because feats will be freed
    std::vector<std::vector<Descriptor>> feats;    // [-w/2,w/2]
    std::vector<std::vector<Vec2D>> keypoints;    // store coordinates in [-w/2,w/2]
    // feature detector
    std::unique_ptr<FeatureDetector> feature_det;
};


#endif /* _IMAGE_STITCHER_TOP_H */
