
#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include <cassert>
#include "CImg.h"
#include "ImageStitcherTop.H"
#include "ImageStitcherTest.H"

using namespace std;
using namespace cimg_library;

#define IMAGE_COUNT 2

Mat32f read_img(const char* fname)
{
    if (! exists_file(fname))
        error_exit(ssprintf("File \"%s\" not exists!", fname));
    CImg<unsigned char> img(fname);
    m_assert(img.spectrum() == 3 || img.spectrum() == 1);
    Mat32f mat(img.height(), img.width(), 3);
    if (img.spectrum() == 3) {
        REP(i, mat.rows())
            REP(j, mat.cols()) {
                mat.at(i, j, 0) = (float)img(j, i, 0) / 255.0;
                mat.at(i, j, 1) = (float)img(j, i, 1) / 255.0;
                mat.at(i, j, 2) = (float)img(j, i, 2) / 255.0;
            }
    } else {
        REP(i, mat.rows())
            REP(j, mat.cols()) {
                mat.at(i, j, 0) = mat.at(i, j, 1) = mat.at(i, j, 2) = img(j, i);
            }
    }
    m_assert(mat.rows() > 1 && mat.cols() > 1);
    return mat;
}


void write_rgb(const char* fname, const Mat32f& mat)
{
    m_assert(mat.channels() == 3);
    CImg<unsigned char> img(mat.cols(), mat.rows(), 1, 3);
    REP(i, mat.rows())
        REP(j, mat.cols()) {
            // use white background. Color::NO turns to 1
            img(j, i, 0) = (mat.at(i, j, 0) < 0 ? 1 : mat.at(i, j, 0)) * 255;
            img(j, i, 1) = (mat.at(i, j, 1) < 0 ? 1 : mat.at(i, j, 1)) * 255;
            img(j, i, 2) = (mat.at(i, j, 2) < 0 ? 1 : mat.at(i, j, 2)) * 255;
        }
    img.save(fname);
}


inline void write_rgb(const std::string s, const Mat32f& mat)
{
    write_rgb(s.c_str(), mat);
}


void ImageRef::load()
{
    if (img)
    {
        return;
    }
    img = new Mat32f{read_img(fname.c_str())};
    _width = img->width();
    _height = img->height();
}

void ImageRef::release()
{
    if (img)
    {
        delete img;
    }
    img = nullptr;
}


int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        error_exit("Need two images to stitch.\n");
    }

    StitcherTest testSticther;
    testSticther.feats.resize(IMAGE_COUNT);
    testSticther.keypoints.resize(IMAGE_COUNT);
    testSticther.feature_det.reset(new SIFTDetector);

    string imgfilename1(argv[1]);
    string imgfilename2(argv[2]);
    ImageRef imgref1(imgfilename1);
    ImageRef imgref2(imgfilename2);

    imgref1.load();
    imgref2.load();
    cout << "Load Images ... done" << endl;

    testSticther.calc_feature(0, imgref1.img);
    testSticther.calc_feature(1, imgref2.img);
    cout << "calculate features ... done" << endl;

    imgref1.release();
    imgref2.release();
    cout << "Release Images ... done" << endl;
}
