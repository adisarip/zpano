
#ifndef _IMAGE_STITCHER_TEST_H
#define _IMAGE_STITCHER_TEST_H


#include <string>
#include <memory>
#include "ImageUtils.H"

Mat32f read_img(const char* fname);
void write_rgb(const char* fname, const Mat32f& mat);
inline void write_rgb(const std::string s, const Mat32f& mat);

// A transparent reference to a image in file
struct ImageRef
{
    std::string fname;
    Mat32f* img = nullptr;
    int _width, _height;

    ImageRef(const std::string& fname): fname(fname) {}
    ~ImageRef() { if (img) delete img; }
    inline int width() const {return _width;}
    inline int height() const { return _height;}
    void load();
    void release();
};


#endif /* _IMAGE_STITCHER_TEST_H */
