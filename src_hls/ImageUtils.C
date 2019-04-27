

#include <mutex>
#include "CImg.h"
#include "ImageUtils.H"


// jpeg can be disabled by compiling with -DDISABLE_JPEG
//#define cimg_display 0
#ifndef DISABLE_JPEG
#define cimg_use_jpeg
#endif

using namespace cimg_library;
using namespace std;


// keep unchanged
const float ORI_WINDOW_FACTOR = 1.5f;
const int ORI_HIST_BIN_NUM = 36;
const float ORI_HIST_PEAK_RATIO = 0.8f;

const int DESC_HIST_WIDTH = 4;
const int DESC_HIST_BIN_NUM = 8;
const int DESC_LEN = 128;       // (4x4)x8
const float DESC_NORM_THRESH = 0.2f;

/*===== Configuration parameters =====*/
#define SIFT_WORKING_SIZE 800         // working resolution for sift
#define NUM_SCALE 7
#define NUM_OCTAVE 4
#define PRE_COLOR_THRES 5e-2
#define JUDGE_EXTREMA_DIFF_THRES 2e-3 // smaller value gives more feature
#define EDGE_RATIO 6                  //larger value gives more feature
#define CONTRAST_THRES 4e-2           // 3e-2. smaller value gives more feature
#define GAUSS_SIGMA 1.4142135623
#define SCALE_FACTOR 1.4142135623
#define OFFSET_THRES 0.5              // 0.3 is still good, this settings has big impact
#define CALC_OFFSET_DEPTH 4
#define DESC_HIST_SCALE_FACTOR 3
#define ORI_HIST_SMOOTH_COUNT 2
#define ORI_RADIUS 4.5                // this radius might be large?
#define DESC_INT_FACTOR 512
#define GAUSS_WINDOW_FACTOR 6         // larger value gives less feature

//==============
//File: timer.cc
//==============


// static member
std::map<std::string, std::pair<int, double>> TotalTimer::rst;

void TotalTimer::print() {
    for (auto& itr : rst)
        print_debug("%s spent %lf secs in total, called %d times.\n",
                itr.first.c_str(), itr.second.second, itr.second.first);
}

TotalTimer::~TotalTimer() {
    static std::mutex mt;
    std::lock_guard<std::mutex> lg(mt);
    auto& p = rst[msg];
    p.second += timer.duration();
    p.first ++;
}

//==============
//File: utils.cc
//==============

std::string TERM_COLOR(int k) {
    // k = 0 ~ 4
    std::ostringstream ss;
    ss << "\x1b[3" << k + 2 << "m";
    return ss.str();
}

void c_printf(const char* col, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("%s", col);
    vprintf(fmt, ap);
    printf(COLOR_RESET);
    va_end(ap);
}

void c_fprintf(const char* col, FILE* fp, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(fp, "%s", col);
    vfprintf(fp, fmt, ap);
    fprintf(fp, COLOR_RESET);
    va_end(ap);
}


std::string ssprintf(const char *fmt, ...) {
    int size = 100;
    char *p = (char *)malloc(size);

    va_list ap;

    std::string ret;

    while (true) {
        va_start(ap, fmt);
        int n = vsnprintf(p, size, fmt, ap);
        va_end(ap);

        if (n < 0) {
            free(p);
            return "";
        }

        if (n < size) {
            ret = p;
            free(p);
            return ret;
        }

        size = n + 1;

        char *np = (char *)realloc(p, size);
        if (np == nullptr) {
            free(p);
            return "";
        } else
            p = np;
    }
}

//===================
//File: debugutils.cc
//===================


void __m_assert_check__(bool val, const char *expr, const char *file, const char *func, int line)
{
    if (val)
        return;
    c_fprintf(COLOR_RED, stderr, "assertion \"%s\" failed, in %s, (%s:%d)\n",
            expr, func, file, line);
    abort();
    //exit(1);
}


void __print_debug__(const char *file, const char *func, int line, const char *fmt, ...)
{
    static map<int, string> colormap;
    if (! colormap[line].length()) {
        int color = std::hash<int>()(line) % 5;
//#pragma omp critical
        colormap[line] = TERM_COLOR(color);
    }

  char *fdup = strdup(file);
  char *fbase = basename(fdup);
  c_fprintf(colormap[line].c_str(), stderr, "[%s@%s:%d] ", func, fbase, line);
  free(fdup);

    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}


void error_exit(const char *msg)
{
    c_fprintf(COLOR_RED, stderr, "error: %s\n", msg);
    exit(1);
}


void error_exit(const std::string& s)
{
    error_exit(s.c_str());
}


//===============
//File: matrix.cc
//===============

#undef Success
#include <eigen3/Eigen/Dense>

namespace {
inline Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    to_eigenmap(const Matrix& m) {
        return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                (double*)m.ptr(), m.rows(), m.cols());
    }
}

ostream& operator << (std::ostream& os, const Matrix & m) {
    os << "[" << m.rows() << " " << m.cols() << "] :" << endl;
    REP(i, m.rows()) REP(j, m.cols())
        os << m.at(i, j) << (j == m.cols() - 1 ? "\n" : ", ");
    return os;
}

Matrix Matrix::transpose() const {
    Matrix ret(m_cols, m_rows);
    REP(i, m_rows) REP(j, m_cols)
        ret.at(j, i) = at(i, j);
    return ret;
}

Matrix Matrix::prod(const Matrix & r) const {
    using namespace Eigen;
    Matrix ret(m_rows, r.cols());
    auto m1 = to_eigenmap(*this);
        auto m2 = to_eigenmap(r);
        auto res = to_eigenmap(ret);
    res = m1 * m2;
    return ret;
}

Matrix Matrix::elem_prod(const Matrix& r) const {
    m_assert(m_rows == r.rows() && m_cols == r.cols());
    Matrix ret(m_rows, m_cols);
    double* res = ret.ptr();
    const double *rl = ptr(), *rr = r.ptr();
    REP(i, pixels()) res[i] = rl[i] * rr[i];
    return ret;
}

Matrix Matrix::operator - (const Matrix& r) const {
    m_assert(rows() == r.rows() && cols() == r.cols());
    Matrix ret(rows(), cols());
    double* res = ret.ptr();
    const double *rl = ptr(), *rr = r.ptr();
    REP(i, pixels()) res[i] = rl[i] - rr[i];
    return ret;
}
Matrix Matrix::operator + (const Matrix& r) const {
    m_assert(rows() == r.rows() && cols() == r.cols());
    Matrix ret(rows(), cols());
    double* res = ret.ptr();
    const double *rl = ptr(), *rr = r.ptr();
    REP(i, pixels()) res[i] = rl[i] + rr[i];
    return ret;
}


bool Matrix::inverse(Matrix &ret) const {
    m_assert(m_rows == m_cols);
    using namespace Eigen;
    ret = Matrix(m_rows, m_rows);
    auto input = to_eigenmap(*this);
        auto res = to_eigenmap(ret);
    FullPivLU<Eigen::Matrix<double,Dynamic,Dynamic,RowMajor>> lu(input);
    if (! lu.isInvertible()) return false;
    res = lu.inverse().eval();
    return true;
}


// pseudo inverse by SVD
Matrix Matrix::pseudo_inverse() const {
    using namespace Eigen;
    m_assert(m_rows >= m_cols);
    auto input = to_eigenmap(*this);
    JacobiSVD<MatrixXd> svd(input, ComputeThinU | ComputeThinV);
    auto sinv = svd.singularValues();
    REP(i, m_cols) {
        if (sinv(i) > EPS)
            sinv(i) = 1.0 / sinv(i);
        else
            sinv(i) = 0;
    }
    Matrix ret(m_cols, m_rows);
    auto res = to_eigenmap(ret);
    res = svd.matrixV() * sinv.asDiagonal() * svd.matrixU().transpose();
    m_assert(ret.at(0, 2) == res(0, 2));
    return ret;
}


void Matrix::normrot() {
    m_assert(m_cols == 3);
    Vec p(at(0, 0), at(1, 0), at(2, 0));
    Vec q(at(0, 1), at(1, 1), at(2, 1));
    Vec r(at(0, 2), at(1, 2), at(2, 2));
    p.normalize();
    q.normalize();
    r.normalize();
    Vec vtmp = p.cross(q);
    double dist = (vtmp - r).mod();
    if (dist > 1e-6)
        r = vtmp;
    at(0, 0) = p.x, at(1, 0) = p.y, at(2, 0) = p.z;
    at(0, 1) = q.x, at(1, 1) = q.y, at(2, 1) = q.z;
    at(0, 2) = r.x, at(1, 2) = r.y, at(2, 2) = r.z;
}

double Matrix::sqrsum() const {
    m_assert(m_cols == 1);
    double sum = 0;
    REP(i, m_rows)
        sum += sqr(at(i, 0));
    return sum;
}

Matrix Matrix::col(int i) const {
    m_assert(i < m_cols);
    Matrix ret(m_rows, 1);
    REP(j, m_rows)
        ret.at(j, 0) = at(j, i);
    return ret;
}

Matrix Matrix::I(int k) {
    Matrix ret(k, k);
    ret.zero();
    REP(i, k)
        ret.at(i, i) = 1;
    return ret;
}

void Matrix::zero() {
    double* p = ptr();
    int n = pixels();
    memset(p, 0, n * sizeof(double));
}




//================
//File: imgproc.cc
//================

void resize_bilinear(const Mat32f &src, Mat32f &dst) {
    vector<int> tabsx(dst.rows());
    vector<int> tabsy(dst.cols());
    vector<float> tabrx(dst.rows());
    vector<float> tabry(dst.cols());

    const float fx = (float)(dst.rows()) / src.rows();
    const float fy = (float)(dst.cols()) / src.cols();
    const float ifx = 1.f / fx;
    const float ify = 1.f / fy;
    for (int dx = 0; dx < dst.rows(); ++dx) {
        float rx = (dx+0.5f) * ifx - 0.5f;
        int sx = floor(rx);
        rx -= sx;
        if (sx < 0) {
            sx = rx = 0;
        } else if (sx + 1 >= src.rows()) {
            sx = src.rows() - 2;
            rx = 1;
        }
        tabsx[dx] = sx;
        tabrx[dx] = rx;
    }
    for (int dy = 0; dy < dst.cols(); ++dy) {
        float ry = (dy+0.5f) * ify - 0.5f;
        int sy = floor(ry);
        ry -= sy;
        if (sy < 0) {
            sy = ry = 0;
            ry = 0;
        } else if (sy + 1 >= src.cols()) {
            sy = src.cols() - 2;
            ry = 1;
        }
        tabsy[dy] = sy;
        tabry[dy] = ry;
    }

    const int ch = src.channels();
    for (int dx = 0; dx < dst.rows(); ++dx) {
        const float *p0 = src.ptr(tabsx[dx]+0);
        const float *p1 = src.ptr(tabsx[dx]+1);
        float *pdst = dst.ptr(dx);
        float rx = tabrx[dx], irx = 1.0f - rx;
        for (int dy = 0; dy < dst.cols(); ++dy) {
            float *pcdst = pdst + dy*ch;
            const float *pc00 = p0 + (tabsy[dy]+0)*ch;
            const float *pc01 = p0 + (tabsy[dy]+1)*ch;
            const float *pc10 = p1 + (tabsy[dy]+0)*ch;
            const float *pc11 = p1 + (tabsy[dy]+1)*ch;
            float ry = tabry[dy], iry = 1.0f - ry;
            for (int c = 0; c < ch; ++c) {
                float res = rx * (pc11[c]*ry + pc10[c]*iry)
                    + irx * (pc01[c]*ry + pc00[c]*iry);
                pcdst[c] = res;
            }
        }
    }
}


template <>
void resize<float>(const Mat32f &src, Mat32f &dst) {
    m_assert(src.rows() > 1 && src.cols() > 1);
    m_assert(dst.rows() > 1 && dst.cols() > 1);
    m_assert(src.channels() == dst.channels());
    m_assert(src.channels() == 1 || src.channels() == 3);
    return resize_bilinear(src, dst);
}


Mat32f crop(const Mat32f& mat) {
    int w = mat.width(), h = mat.height();
    vector<int> height(w, 0),
        left(w), right(w);
    int maxarea = 0;
    int ll = 0, rr = 0, hh = 0, nl = 0;
    REP(line, h) {
        REP(k, w) {
            const float* p = mat.ptr(line, k);
            float m = max(max(p[0], p[1]), p[2]);
            height[k] = m < 0 ? 0 : height[k] + 1;    // find Color::NO
        }

        REP(k, w) {
            left[k] = k;
            while (left[k] > 0 && height[k] <= height[left[k] - 1])
                left[k] = left[left[k] - 1];
        }
        REPD(k, w - 1, 0) {
            right[k] = k;
            while (right[k] < w - 1 && height[k] <= height[right[k] + 1])
                right[k] = right[right[k] + 1];
        }
        REP(k, w)
            if (update_max(maxarea, (right[k] - left[k] + 1) * height[k]))
                ll = left[k], rr = right[k], hh = height[k], nl = line;
    }
    Mat32f ret(hh, rr - ll + 1, 3);
    int offsetx = ll, offsety = nl - hh + 1;
    REP(i, ret.height()) {
        float* dst = ret.ptr(i, 0);
        const float* src = mat.ptr(i + offsety, offsetx);
        memcpy(dst, src, 3 * ret.width() * sizeof(float));
    }
    return ret;
}


Mat32f rgb2grey(const Mat32f& mat) {
    m_assert(mat.channels() == 3);
    Mat32f ret(mat.height(), mat.width(), 1);
    const float* src = mat.ptr();
    float* dst = ret.ptr();
    int n = mat.pixels();
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        dst[i] = (src[idx] + src[idx+1] + src[idx+2]) / 3.f;
        idx += 3;
    }
    return ret;
}

//=============
//File: sift.cc
//=============

float Descriptor::euclidean_sqr(const float* x,
                                const float* y,
                                size_t size,
                                float now_thres)
{
    m_assert(size % 4 == 0);
    float ans = 0;
    float diff0, diff1, diff2, diff3;
    const float* end = x + size;
    while (x < end)
    {
        diff0 = x[0] - y[0];
        diff1 = x[1] - y[1];
        diff2 = x[2] - y[2];
        diff3 = x[3] - y[3];
        ans += sqr(diff0) + sqr(diff1) + sqr(diff2) + sqr(diff3);
        if (ans > now_thres)
        {
            return std::numeric_limits<float>::max();
        }
        x += 4, y += 4;
    }
    return ans;
}


int Descriptor::hamming(const float* x, const float* y, int n)
{
    int sum = 0;
    REP(i, n)
    {
        unsigned int* p1 = (unsigned int*)&x[i];
        unsigned int* p2 = (unsigned int*)&y[i];
        sum += __builtin_popcount((*p1) ^ *(p2));
    }
    return sum;
}



const int featlen = DESC_HIST_WIDTH * DESC_HIST_WIDTH * DESC_HIST_BIN_NUM;


Descriptor hist_to_descriptor(float* hist) {
    Descriptor ret;
    ret.descriptor.resize(featlen);
    memcpy(ret.descriptor.data(), hist, featlen * sizeof(float));
  float sum = 0;

    // using RootSIFT: rootsift= sqrt( sift / sum(sift) );
    // L1 normalize SIFT
    sum = 0;
    for (auto &i : ret.descriptor) sum += i;
    for (auto &i : ret.descriptor) i /= sum;
    // square root each element
    for (auto &i : ret.descriptor) i = std::sqrt(i) * DESC_INT_FACTOR;

    return ret;
}


void trilinear_interpolate(
        float xbin, float ybin, float hbin,
        float weight, float hist[][DESC_HIST_BIN_NUM]) {
    // WARNING: x,y can be -1
    int ybinf = floor(ybin),
    xbinf = floor(xbin),
    hbinf = floor(hbin);
    float ybind = ybin - ybinf,
                xbind = xbin - xbinf,
                hbind = hbin - hbinf;
    REP(dy, 2) if (between(ybinf + dy, 0, DESC_HIST_WIDTH)) {
        float w_y = weight * (dy ? ybind : 1 - ybind);
        REP(dx, 2) if (between(xbinf + dx, 0, DESC_HIST_WIDTH)) {
            float w_x = w_y * (dx ? xbind : 1 - xbind);
            int bin_2d_idx = (ybinf + dy) * DESC_HIST_WIDTH + (xbinf + dx);
            hist[bin_2d_idx][hbinf % DESC_HIST_BIN_NUM] += w_x * (1 - hbind);
            hist[bin_2d_idx][(hbinf + 1) % DESC_HIST_BIN_NUM] += w_x * hbind;
        }
    }
}


SIFT::SIFT(const ScaleSpace& ss,
        const vector<SSPoint>& keypoints):
    ss(ss), points(keypoints)
{ }


std::vector<Descriptor> SIFT::get_descriptor() const {
    TotalTimer tm("sift descriptor");
    vector<Descriptor> ret;
    for (auto& p : points) {
        auto desp = calc_descriptor(p);
        ret.emplace_back(move(desp));
    }
    return ret;
}


Descriptor SIFT::calc_descriptor(const SSPoint& p) const {
    const static float pi2 = 2 * M_PI;
    const static float nbin_per_rad = DESC_HIST_BIN_NUM / pi2;

    const GaussianPyramid& pyramid = ss.pyramids[p.pyr_id];
    int w = pyramid.w, h = pyramid.h;
    auto& mag_img = pyramid.get_mag(p.scale_id);
    auto& ort_img = pyramid.get_ort(p.scale_id);

    Coor coor = p.coor;
    float ort = p.dir,
                // size of blurred field of this point in orignal image
                hist_w = p.scale_factor * DESC_HIST_SCALE_FACTOR,
                // sigma is half of window width from lowe
                exp_denom = 2 * sqr(DESC_HIST_WIDTH);
    // radius of gaussian to use
    int radius = round(M_SQRT1_2 * hist_w * (DESC_HIST_WIDTH + 1));

    float hist[DESC_HIST_WIDTH * DESC_HIST_WIDTH][DESC_HIST_BIN_NUM];
    memset(hist, 0, sizeof(hist));
    float cosort = cos(ort),
                sinort = sin(ort);

    for (int xx = -radius; xx <= radius; xx ++) {
        int nowx = coor.x + xx;
        if (!between(nowx, 1, w - 1)) continue;
        for (int yy = -radius; yy <= radius; yy ++) {
            int nowy = coor.y + yy;
            if (!between(nowy, 1, h - 1)) continue;
            if (sqr(xx) + sqr(yy) > sqr(radius)) continue;        // to be circle
            // coordinate change, relative to major orientation
            // major orientation become (x, 0)
            float y_rot = (-xx * sinort + yy * cosort) / hist_w,
                        x_rot = (xx * cosort + yy * sinort) / hist_w;
            // calculate 2d bin idx (which bin do I fall into)
            // -0.5 to make the center of bin 1st (x=1.5) falls fully into bin 1st
            float ybin = y_rot + DESC_HIST_WIDTH / 2 - 0.5,
                        xbin = x_rot + DESC_HIST_WIDTH / 2 - 0.5;

            if (!between(ybin, -1, DESC_HIST_WIDTH) ||
                    !between(xbin, -1, DESC_HIST_WIDTH)) continue;

            float now_mag = mag_img.at(nowy, nowx),
                        now_ort = ort_img.at(nowy, nowx);
            // gaussian & magitude weight on histogram
            float weight = expf(-(sqr(x_rot) + sqr(y_rot)) / exp_denom);
            weight = weight * now_mag;

            now_ort -= ort;    // for rotation invariance
            if (now_ort < 0) now_ort += pi2;
            if (now_ort > pi2) now_ort -= pi2;
            // bin number in histogram
            float hist_bin = now_ort * nbin_per_rad;

            // all three bin idx are float, do trilinear interpolation
            trilinear_interpolate(
                    xbin, ybin, hist_bin, weight, hist);
        }
    }

    // build descriptor from hist

    Descriptor ret = hist_to_descriptor((float*)hist);
    ret.coor = p.real_coor;
    return ret;
}


//====================
//File: orientation.cc
//====================


OrientationAssign::OrientationAssign(
        const DOGSpace& dog, const ScaleSpace& ss,
        const std::vector<SSPoint>& keypoints):
    dog(dog),ss(ss), points(keypoints) {}


vector<SSPoint> OrientationAssign::work() const {
    vector<SSPoint> ret;
    for (auto& p : points) {
        auto major_orient = calc_dir(p);
        for (auto& o : major_orient) {
            ret.emplace_back(p);
            ret.back().dir = o;
        }
    }
    return ret;
}

std::vector<float> OrientationAssign::calc_dir(
        const SSPoint& p) const {
    const static float halfipi = 0.5f / M_PI;
    auto& pyramid = ss.pyramids[p.pyr_id];
    auto& orient_img = pyramid.get_ort(p.scale_id);
    auto& mag_img = pyramid.get_mag(p.scale_id);

    float gauss_weight_sigma = p.scale_factor * ORI_WINDOW_FACTOR;
    int rad = round(p.scale_factor * ORI_RADIUS);
    float exp_denom = 2 * sqr(gauss_weight_sigma);
    float hist[ORI_HIST_BIN_NUM];
    memset(hist, 0, sizeof(hist));

    // calculate gaussian/magnitude weighted histogram
    // of orientation inside a circle
    for (int xx = -rad; xx < rad; xx ++) {
        int newx = p.coor.x + xx;
        // because mag/ort on the border is zero
        if (! between(newx, 1, pyramid.w - 1)) continue;
        for (int yy = -rad; yy < rad; yy ++) {
            int newy = p.coor.y + yy;
            if (! between(newy, 1, pyramid.h - 1)) continue;
          // use a circular gaussian window
            if (sqr(xx) + sqr(yy) > sqr(rad)) continue;
            float orient = orient_img.at(newy, newx);
            int bin = round(ORI_HIST_BIN_NUM * halfipi * orient );
            if (bin == ORI_HIST_BIN_NUM) bin = 0;
            m_assert(bin < ORI_HIST_BIN_NUM);

            float weight = expf(-(sqr(xx) + sqr(yy)) / exp_denom);
            hist[bin] += weight * mag_img.at(newy, newx);
        }
    }

    // TODO do we need this?
    // smooth the histogram by interpolation
    for (int K = ORI_HIST_SMOOTH_COUNT; K --;)
        REP(i, ORI_HIST_BIN_NUM) {
            float prev = hist[i == 0 ? ORI_HIST_BIN_NUM - 1 : i - 1];
            float next = hist[i == ORI_HIST_BIN_NUM - 1 ? 0 : i + 1];
            hist[i] = hist[i] * 0.5 + (prev + next) * 0.25;
        }

    float maxbin = 0;
    for (auto i : hist) update_max(maxbin, i);
    float thres = maxbin * ORI_HIST_PEAK_RATIO;
    vector<float> ret;

    REP(i, ORI_HIST_BIN_NUM) {
        float prev = hist[i == 0 ? ORI_HIST_BIN_NUM - 1 : i - 1];
        float next = hist[i == ORI_HIST_BIN_NUM - 1 ? 0 : i + 1];
        // choose extreme orientation which is larger than thres
        if (hist[i] > thres && hist[i] > max(prev, next)) {
            // parabola interpolation
            real_t newbin = (float)i - 0.5 + (hist[i] - prev) / (prev + next - 2 * hist[i]);
            // I saw this elsewhere... although by derivation the above one should be correct
            //real_t newbin = (float)i - 0.5 + (next - prev) / (prev + next - 2 * hist[i]);
            if (newbin < 0)
                newbin += ORI_HIST_BIN_NUM;
            else if (newbin >= ORI_HIST_BIN_NUM)
                newbin -= ORI_HIST_BIN_NUM;
            float ort = newbin / ORI_HIST_BIN_NUM * 2 * M_PI;
            ret.push_back(ort);
        }
    }
    return ret;
}


//=================
//File: gaussian.cc
//=================

GaussCache::GaussCache(float sigma) {
    // TODO decide window size ?
    /*
     *const int kw = round(GAUSS_WINDOW_FACTOR * sigma) + 1;
     */
    kw = ceil(0.3 * (sigma / 2 - 1) + 0.8) * GAUSS_WINDOW_FACTOR;
    //cout << kw << " " << sigma << endl;
    if (kw % 2 == 0) kw ++;
    kernel_buf.reset(new float[kw]);
    const int center = kw / 2;
    kernel = kernel_buf.get() + center;

    kernel[0] = 1;

    float exp_coeff = -1.0 / (sigma * sigma * 2),
                 wsum = 1;
    for (int i = 1; i <= center; i ++)
        wsum += (kernel[i] = exp(i * i * exp_coeff)) * 2;

    float fac = 1.0 / wsum;
    kernel[0] = fac;
    for (int i = 1; i <= center; i ++)
        kernel[-i] = (kernel[i] *= fac);
}



//============
//File: dog.cc
//============

// fast approximation to atan2.
// atan2(a, b) = fast_atan(a, b), given max(abs(a),abs(b)) > EPS
// http://math.stackexchange.com/questions/1098487/atan2-faster-approximation
// save cal_mag_ort() 40% time
float fast_atan(float y, float x) {
    float absx = fabs(x), absy = fabs(y);
    float m = max(absx, absy);

    // undefined behavior in atan2.
    // but here we can safely ignore by setting ort=0
    if (m < EPS) return -M_PI;
    float a = min(absx, absy) / m;
    float s = a * a;
    float r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
    if (absy > absx)
        r = M_PI_2 - r;
    if (x < 0) r = M_PI - r;
    if (y < 0) r = -r;
    return r;
}

GaussianPyramid::GaussianPyramid(const Mat32f& m, int num_scale):
    nscale(num_scale),
    data(num_scale), mag(num_scale), ort(num_scale),
    w(m.width()), h(m.height())
{
    TotalTimer tm("build pyramid");
    if (m.channels() == 3)
        data[0] = rgb2grey(m);
    else
        data[0] = m.clone();

    MultiScaleGaussianBlur blurer(nscale, GAUSS_SIGMA, SCALE_FACTOR);
    for (int i = 1; i < nscale; i ++) {
        data[i] = blurer.blur(data[0], i);    // sigma needs a better one
        cal_mag_ort(i);
    }
}

void GaussianPyramid::cal_mag_ort(int i) {
    TotalTimer tm("cal_mag_ort");
    const Mat32f& orig = data[i];
    int w = orig.width(), h = orig.height();
    mag[i] = Mat32f(h, w, 1);
    ort[i] = Mat32f(h, w, 1);
    REP(y, h) {
        float *mag_row = mag[i].ptr(y),
                    *ort_row = ort[i].ptr(y);
        const float *orig_row = orig.ptr(y),
                    *orig_plus = orig.ptr(y + 1),
                    *orig_minus = orig.ptr(y - 1);
        // x == 0:
        mag_row[0] = 0;
        ort_row[0] = M_PI;

        REPL(x, 1, w-1) {
            if (between(y, 1, h - 1)) {
                float dy = orig_plus[x] - orig_minus[x],
                            dx = orig_row[x + 1] - orig_row[x - 1];
                mag_row[x] = hypotf(dx, dy);
                // approx here cause break working on myself/small*. fix later
                // when dx==dy==0, no need to set ort
                ort_row[x] = fast_atan(dy, dx) + M_PI;
            } else {
                mag_row[x] = 0;
                ort_row[x] = M_PI;
            }
        }
        // x == w-1
        mag_row[w-1] = 0;
        ort_row[w-1] = M_PI;

    }
}

ScaleSpace::ScaleSpace(const Mat32f& mat, int num_octave, int num_scale):
    noctave(num_octave), nscale(num_scale),
    origw(mat.width()), origh(mat.height())
{
    // #pragma omp parallel for schedule(dynamic)
    REP(i, noctave) {
        if (!i)
            pyramids.emplace_back(mat, nscale);
        else {
            float factor = pow(SCALE_FACTOR, -i);
            int neww = ceil(origw * factor),
                    newh = ceil(origh * factor);
            m_assert(neww > 5 && newh > 5);
            Mat32f resized(newh, neww, 3);
            resize(mat, resized);
            pyramids.emplace_back(resized, nscale);
        }
    }
}


DOGSpace::DOGSpace(ScaleSpace& ss):
    noctave(ss.noctave), nscale(ss.nscale),
    origw(ss.origw), origh(ss.origh),
    dogs(noctave)
{
//#pragma omp parallel for schedule(dynamic)
    REP(i, noctave) {
        auto& o = ss.pyramids[i];
        int ns = o.get_len();
        REP(j, ns - 1)
            dogs[i].emplace_back(diff(o.get(j), o.get(j+1)));
    }
}


Mat32f DOGSpace::diff(const Mat32f& img1, const Mat32f& img2) const
{
    int w = img1.width(), h = img1.height();
    m_assert(w == img2.width() && h == img2.height());
    Mat32f ret(h, w, 1);
    REP(i, h) {
        // speed up
        const float *p1 = img1.ptr(i),
                          *p2 = img2.ptr(i);
        float* p = ret.ptr(i);
        REP(j, w)
            p[j] = fabs(p1[j] - p2[j]);
    }
    return ret;
}


//================
//File: extrema.cc
//================



ExtremaDetector::ExtremaDetector(const DOGSpace& dg):
    dog(dg) {}

vector<Coor> ExtremaDetector::get_raw_extrema() const {
    vector<Coor> ret;
    int npyramid = dog.noctave, nscale = dog.nscale;
    REP(i, npyramid)
        REPL(j, 1, nscale - 2) {
            auto& now = dog.dogs[i][j];
            int w = now.width(), h = now.height();

            auto v = get_local_raw_extrema(i, j);
            for (auto& c : v) {
                ret.emplace_back((float)c.x / w * dog.origw,
                        (float)c.y / h * dog.origh);
            }
        }
    return ret;
}

vector<SSPoint> ExtremaDetector::get_extrema() const {
    TotalTimer tm("extrema");
    int npyramid = dog.noctave, nscale = dog.nscale;
    vector<SSPoint> ret;
//#pragma omp parallel for schedule(dynamic)
    REP(i, npyramid)
        REPL(j, 1, nscale - 2) {
            auto v = get_local_raw_extrema(i, j);
            //print_debug("raw extrema count: %lu\n", v.size());
            for (auto& c : v) {
                SSPoint sp;
                sp.coor = c;
                sp.pyr_id = i;
                sp.scale_id = j;
                bool succ = calc_kp_offset(&sp);
                if (! succ) continue;
                auto& img = dog.dogs[i][sp.scale_id];
                succ = ! is_edge_response(sp.coor, img);
                if (! succ) continue;

//#pragma omp critical
                ret.emplace_back(sp);
            }
        }
    return ret;
}

bool ExtremaDetector::calc_kp_offset(SSPoint* sp) const {
    auto& now_pyramid = dog.dogs[sp->pyr_id];
    auto& now_img = now_pyramid[sp->scale_id];
    int w = now_img.width(), h = now_img.height();
    int nscale = dog.nscale;

    Vec offset, delta;    // partial(d) / partial(offset)
    int nowx = sp->coor.x, nowy = sp->coor.y, nows = sp->scale_id;
    int niter = 0;
    for(;niter < CALC_OFFSET_DEPTH; ++niter) {
        if (!between(nowx, 1, w - 1) ||
                !between(nowy, 1, h - 1) ||
                !between(nows, 1, nscale - 2))
            return false;

        auto iter_offset = calc_kp_offset_iter(
                now_pyramid, nowx, nowy, nows);
        offset = iter_offset.first;
        delta = iter_offset.second;
        if (offset.get_abs_max() < OFFSET_THRES) // found
            break;

        nowx += round(offset.x);
        nowy += round(offset.y);
        nows += round(offset.z);
    }
    if (niter == CALC_OFFSET_DEPTH) return false;

    double dextr = offset.dot(delta);        // calc D(x~)
    dextr = now_pyramid[nows].at(nowy, nowx) + dextr / 2;
    // contrast too low
    if (dextr < CONTRAST_THRES) return false;

    // update the point
    sp->coor = Coor(nowx, nowy);
    sp->scale_id = nows;
    sp->scale_factor = GAUSS_SIGMA * pow(
                SCALE_FACTOR, ((double)nows + offset.z) / nscale);
    // accurate real-value coor
    sp->real_coor = Vec2D(
            ((double)nowx + offset.x) / w,
            ((double)nowy + offset.y) / h);
    return true;
}

std::pair<Vec, Vec> ExtremaDetector::calc_kp_offset_iter(
        const DOGSpace::DOG& now_pyramid,
        int x , int y, int s) const {
    Vec offset = Vec::get_zero();
    Vec delta;
    double dxx, dyy, dss, dxy, dys, dsx;

    auto& now_scale = now_pyramid[s];
#define D(x, y, s) now_pyramid[s].at(y, x)
#define DS(x, y) now_scale.at(y, x)
    float val = DS(x, y);

    delta.x = (DS(x + 1, y) - DS(x - 1, y)) / 2;
    delta.y = (DS(x, y + 1) - DS(x, y - 1)) / 2;
    delta.z = (D(x, y, s + 1) - D(x, y, s - 1)) / 2;

    dxx = DS(x + 1, y) + DS(x - 1, y) - val - val;
    dyy = DS(x, y + 1) + DS(x, y - 1) - val - val;
    dss = D(x, y, s + 1) + D(x, y, s - 1) - val - val;

    dxy = (DS(x + 1, y + 1) - DS(x + 1, y - 1) - DS(x - 1, y + 1) + DS(x - 1, y - 1)) / 4;
    dys = (D(x, y + 1, s + 1) - D(x, y - 1, s + 1) - D(x, y + 1, s - 1) + D(x, y - 1, s - 1)) / 4;
    dsx = (D(x + 1, y, s + 1) - D(x - 1, y, s + 1) - D(x + 1, y, s - 1) + D(x - 1, y, s - 1)) / 4;
#undef D
#undef DS

    Matrix m(3, 3);
    m.at(0, 0) = dxx; m.at(1, 1) = dyy; m.at(2, 2) = dss;
    m.at(0, 1) = m.at(1, 0) = dxy;
    m.at(0, 2) = m.at(2, 0) = dsx;
    m.at(1, 2) = m.at(2, 1) = dys;

    Matrix pdpx(3, 1);    // delta = dD / dx
    delta.write_to(pdpx.ptr());

    Matrix inv;
    if (! m.inverse(inv)) {      // pseudo inverse is slow
        inv = m.pseudo_inverse();
    }
    auto prod = inv.prod(pdpx);
    offset = Vec(prod.ptr());
    return {offset, delta};
}

bool ExtremaDetector::is_edge_response(Coor coor, const Mat32f& img) const {
    float dxx, dxy, dyy;
    int x = coor.x, y = coor.y;
    float val = img.at(y, x);

    dxx = img.at(y, x + 1) + img.at(y, x - 1) - val - val;
    dyy = img.at(y + 1, x) + img.at(y - 1, x) - val - val;
    dxy = (img.at(y + 1, x + 1) + img.at(y - 1, x - 1) -
            img.at(y + 1, x - 1) - img.at(y - 1, x + 1)) / 4;
    float det = dxx * dyy - dxy * dxy;
    if (det <= 0) return true;
    float tr2 = sqr(dxx + dyy);

    // Calculate principal curvature by hessian
    if (tr2 / det < sqr(EDGE_RATIO + 1) / EDGE_RATIO) return false;
    return true;
}

vector<Coor> ExtremaDetector::get_local_raw_extrema(
        int pyr_id, int scale_id) const {
    vector<Coor> ret;

    const Mat32f& now(dog.dogs[pyr_id][scale_id]);
    int w = now.width(), h = now.height();

    auto is_extrema = [this, &now, pyr_id, scale_id](int r, int c) {
        float center = now.at(r, c);
        if (center < PRE_COLOR_THRES)            // initial color is less than thres
            return false;

        bool max = true, min = true;
        float cmp1 = center - JUDGE_EXTREMA_DIFF_THRES,
                    cmp2 = center + JUDGE_EXTREMA_DIFF_THRES;
        // try same scale
        REPL(di, -1, 2) REPL(dj, -1, 2) {
            if (!di && !dj) continue;
            float newval = now.at(r + di, c + dj);
            if (newval >= cmp1) max = false;
            if (newval <= cmp2) min = false;
            if (!max && !min) return false;
        }

        // try adjencent scale
        for (int ds = -1; ds < 2; ds += 2) {
            int nl = scale_id + ds;
            auto& mat = this->dog.dogs[pyr_id][nl];

            REPL(di, -1, 2) {
                const float* p = mat.ptr(r + di) + c - 1;
                REP(i, 3) {
                    float newval = p[i];
                    if (newval >= cmp1) max = false;
                    if (newval <= cmp2) min = false;
                    if (!max && !min) return false;
                }
            }
        }
        return true;
    };

    REPL(i, 1, h - 1) REPL(j, 1, w - 1)
        if (is_extrema(i, j))
            ret.emplace_back(j, i);
    return ret;
}


//================
//File: feature.cc
//================


// return half-shifted image coordinate
vector<Descriptor>
FeatureDetector::detect_feature(const Mat32f& img) const
{
	auto ret = do_detect_feature(img);
    // convert scale-coordinate to half-offset image coordinate
    for (auto& d: ret)
    {
        d.coor.x = (d.coor.x - 0.5) * img.width();
        d.coor.y = (d.coor.y - 0.5) * img.height();
    }
    return ret;
}


// return [0, 1] coordinate
vector<Descriptor>
SIFTDetector::do_detect_feature(const Mat32f& mat) const
{
    // perform sift at this resolution
	float ratio = SIFT_WORKING_SIZE * 2.0f / (mat.width() + mat.height());
    Mat32f resized(mat.rows() * ratio, mat.cols() * ratio, 3);
    resize(mat, resized);

    ScaleSpace ss(resized, NUM_OCTAVE, NUM_SCALE);
    DOGSpace sp(ss);
    ExtremaDetector ex(sp);
    auto keyp = ex.get_extrema();
    OrientationAssign ort(sp, ss, keyp);
    keyp = ort.work();
    SIFT sift(ss, keyp);
    auto descp = sift.get_descriptor();
    return descp;
}
