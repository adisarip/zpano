
#include <vector>
#include <memory>
#include "ImageUtils.H"


std::vector<ImageRef> imgs;
// feature and keypoints of each image
// keypoints only work as a storage of feats, because feats will be freed
std::vector<std::vector<Descriptor>> feats;    // [-w/2,w/2]
std::vector<std::vector<Vec2D>> keypoints;    // store coordinates in [-w/2,w/2]
// feature detector
std::unique_ptr<FeatureDetector> feature_det;
// get feature descriptor and keypoints for each image
void calc_feature();
