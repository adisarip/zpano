
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <memory>
#include <cstring>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <limits>
#include <vector>
#include <map>
#include <list>
#include <iterator>
#include <algorithm>
#include <chrono>
#include <functional>
#include <sstream>
#include <memory>
#include <sys/stat.h>
#include <type_traits>

#include <libgen.h>
using namespace std;


//===============
//File: common.hh
//===============

#ifdef MSVC
#define not !
#endif

//==============
//File: utils.hh
//==============


#ifdef _WIN32
#define __attribute__(x)
#endif

typedef double real_t;
const real_t EPS = 1e-6;
const real_t GEO_EPS_SQR = 1e-14;
const real_t GEO_EPS = 1e-7;
inline float sqr(float x) { return x * x; }

#define between(a, b, c) ((a >= b) && (a <= c - 1))
#define REP(x, y) for (typename std::remove_cv<typename std::remove_reference<decltype(y)>::type>::type x = 0; x < (y); x ++)
#define REPL(x, y, z) for (typename std::remove_cv<typename std::remove_reference<decltype(y)>::type>::type x = y; x < (z); x ++)
#define REPD(x, y, z) for (typename std::remove_cv<typename std::remove_reference<decltype(y)>::type>::type x = y; x >= (z); x --)

std::string TERM_COLOR(int k);

#define COLOR_RED     "\x1b[31m"
#define COLOR_GREEN   "\x1b[32m"
#define COLOR_YELLOW  "\x1b[33m"
#define COLOR_BLUE    "\x1b[34m"
#define COLOR_MAGENTA "\x1b[35m"
#define COLOR_CYAN    "\x1b[36m"
#define COLOR_RESET   "\x1b[0m"

void c_printf(const char* col, const char* fmt, ...);

void c_fprintf(const char* col, FILE* fp, const char* fmt, ...);

__attribute__ (( format( printf, 1, 2 ) ))
std::string ssprintf(const char *fmt, ...);

template<typename T>
inline bool update_min(T &dest, const T &val) {
    if (val < dest) {
        dest = val; return true;
    }
    return false;
}

template<typename T>
inline bool update_max(T &dest, const T &val) {
    if (dest < val) {
        dest = val; return true;
    }
    return false;
}

template <typename T>
inline void free_2d(T** ptr, int w) {
    if (ptr != nullptr)
        for (int i = 0; i < w; i ++)
            delete[] ptr[i];
    delete[] ptr;
}

template <typename T>
std::shared_ptr<T> create_auto_buf(size_t len, bool init_zero = false) {
    std::shared_ptr<T> ret(new T[len], std::default_delete<T[]>());
    if (init_zero)
        memset(ret.get(), 0, sizeof(T) * len);
    return ret;
}


inline bool exists_file(const char* name) {
    struct stat buffer;
    return stat(name, &buffer) == 0;
}

inline bool endswith(const char* str, const char* suffix) {
    if (!str || !suffix) return false;
    auto l1 = strlen(str), l2 = strlen(suffix);
    if (l2 > l1) return false;
    return strncmp(str + l1 - l2, suffix, l2) == 0;
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
//File: debugutils.hh
//===================

#ifdef _WIN32
#define __attribute__(x)
#endif

#define P(a) std::cout << (a) << std::endl
#define PP(a) std::cout << #a << ": " << (a) << std::endl
#define PA(arr) \
	do { \
		std::cout << #arr << ": "; \
		std::copy(begin(arr), end(arr), std::ostream_iterator<std::remove_reference<decltype(arr)>::type::value_type>(std::cout, " ")); \
		std::cout << std::endl;  \
	} while (0)

void __m_assert_check__(bool val, const char *expr,
		const char *file, const char *func, int line);


void error_exit(const char *msg) __attribute__((noreturn));

inline void error_exit(const std::string& s) __attribute__((noreturn));
void error_exit(const std::string& s) {
	error_exit(s.c_str());
}

// keep print_debug
#define print_debug(fmt, ...) __print_debug__(__FILE__, __func__, __LINE__, fmt, ## __VA_ARGS__)
//#define print_debug(fmt, ...)

void __print_debug__(const char *file, const char *func, int line, const char *fmt, ...)
	__attribute__((format(printf, 4, 5)));

#ifdef DEBUG
#define m_assert(expr) \
	__m_assert_check__((expr), # expr, __FILE__, __func__ , __LINE__)
#else
#define m_assert(expr)
#endif



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
#pragma omp critical
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


void error_exit(const char *msg) {
	c_fprintf(COLOR_RED, stderr, "error: %s\n", msg);
	exit(1);
}


//===========
//File: mat.h
//===========


template <typename T>
class Mat {
    public:
                Mat(){}
                Mat(int rows, int cols, int channels):
                    m_rows(rows), m_cols(cols), m_channels(channels),
                    m_data{new T[rows * cols * channels], std::default_delete<T[]>() }
                { }

                virtual ~Mat(){}

                T &at(int r, int c, int ch = 0) {
                    m_assert(r < m_rows);
                    m_assert(c < m_cols);
                    m_assert(ch < m_channels);
                    return ptr(r)[c * m_channels + ch];
                }

                const T &at(int r, int c, int ch = 0) const {
                    m_assert(r < m_rows);
                    m_assert(c < m_cols);
                    m_assert(ch < m_channels);
                    return ptr(r)[c * m_channels + ch];
                }

                Mat<T> clone() const {
                    Mat<T> res(m_rows, m_cols, m_channels);
                    memcpy(res.ptr(0), this->ptr(0), sizeof(T) * m_rows * m_cols * m_channels);
                    return res;
                }

                const T *ptr(int r = 0) const
                { return m_data.get() + r * m_cols * m_channels; }
                T *ptr(int r = 0)
                { return m_data.get() + r * m_cols * m_channels; }
                const T *ptr(int r, int c) const
                { return m_data.get() + (r * m_cols + c) * m_channels; }
                T *ptr(int r, int c)
                { return m_data.get() + (r * m_cols + c) * m_channels; }
                int height() const { return m_rows; }
                int width() const { return m_cols; }
                int rows() const { return m_rows; }
                int cols() const { return m_cols; }
                int channels() const { return m_channels; }
                int pixels() const { return m_rows * m_cols; }

        protected:
                int m_rows, m_cols;
                int m_channels;
                std::shared_ptr<T> m_data;
};

using Mat32f = Mat<float>;
using Matuc = Mat<unsigned char>;


//===============
//File: matrix.hh
//===============

class Matrix : public Mat<double> {
	public:
		Matrix(){}

		Matrix(int rows, int cols):
			Mat<double>(rows, cols, 1) {}

		Matrix(const Mat<double>& r):
			Mat<double>(r) {}

		bool inverse(Matrix & ret) const;

		Matrix pseudo_inverse() const;

		Matrix transpose() const;

		Matrix prod(const Matrix & r) const;

		Matrix elem_prod(const Matrix& r) const;

		Matrix operator * (const Matrix& r) const
		{ return prod(r); }

		void mult(double m) {
			int n = pixels();
			double* p = ptr();
			for (int i = 0; i < n; i ++)
				*p *= m, p++;
		}

		Matrix operator - (const Matrix& r) const;
		Matrix operator + (const Matrix& r) const;

		bool SVD(Matrix & u, Matrix & s, Matrix & v) const;

		void normrot();

		double sqrsum() const;

		Matrix col(int i) const;

		void zero();

		static Matrix I(int);

		friend std::ostream& operator << (std::ostream& os, const Matrix & m);

};


//=================
//File: geometry.hh
//=================

template<typename T>
class Vector {
	public:
		T x = 0, y = 0, z = 0;

		constexpr explicit Vector(T m_x = 0, T m_y = 0, T m_z = 0):
			x(m_x), y(m_y), z(m_z) {}

		Vector(const Vector &p0, const Vector &p1):
			x(p1.x - p0.x), y(p1.y -p0.y), z(p1.z - p0.z) {}

		explicit Vector(const T* p):
			x(p[0]), y(p[1]), z(p[2]) {}

		T index(int c) const
		{ return c == 0 ? x : c == 1 ? y : z; }

		T& index(int c)
		{ return c == 0 ? x : c == 1 ? y : z; }

		T min_comp_abs() const {
			T a = fabs(x), b = fabs(y), c = fabs(z);
			::update_min(a, b), ::update_min(a, c);
			return a;
		}

		T sqr() const
		{ return x * x + y * y + z * z; }

		T mod() const
		{ return sqrt(sqr()); }

		T dot(const Vector &v) const
		{ return x * v.x + y * v.y + z * v.z; }

		Vector cross(const Vector &v) const
		{ return Vector(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }

		Vector& operator = (const Vector& v)
		{ x = v.x, y = v.y, z = v.z; return *this; }

		void normalize() {
			T m = 1 / mod();
			*this *= m;		// work?
			m_assert(std::isnormal(m));
		}

		Vector get_normalized() const
		{ Vector ret(*this); ret.normalize(); return ret; }

		bool is_zero(T threshold = EPS) const
		{ return fabs(x) < threshold && fabs(y) < threshold && fabs(z) < threshold; }

		bool is_positive(T threshold = EPS) const
		{ return x > threshold && y > threshold && z > threshold; }

		void update_min(const Vector &v)
		{ ::update_min(x, v.x); ::update_min(y, v.y); ::update_min(z, v.z); }

		void update_max(const Vector &v)
		{ ::update_max(x, v.x); ::update_max(y, v.y); ::update_max(z, v.z); }

		Vector operator + (const Vector &v) const
		{ return Vector(x + v.x, y + v.y, z + v.z); }

		Vector& operator += (const Vector &v)
		{ x += v.x; y += v.y; z += v.z; return *this; }

		Vector operator - (const Vector &v) const
		{ return Vector(x - v.x, y - v.y, z - v.z); }

		Vector operator - () const
		{ return Vector(-x, -y, -z); }

		Vector& operator -= (const Vector &v)
		{ x -= v.x; y -= v.y; z -= v.z; return *this; }

		Vector operator * (T p) const
		{ return Vector(x * p, y * p, z * p); }

		Vector& operator *= (T p)
		{ x *= p; y *= p; z *= p; return *this; }

		Vector operator / (T p) const
		{ return *this * (1.0 / p); }

		Vector& operator /= (T p)
		{ x /= p; y /= p; z /= p; return *this; }

		bool operator == (const Vector &v) const
		{ return fabs(x - v.x) < EPS && fabs(y - v.y) < EPS && fabs(z - v.z) < EPS; }

		bool operator != (const Vector &v) const
		{ return fabs(x - v.x) >= EPS || fabs(y - v.y) >= EPS || fabs(z - v.z) >= EPS; }

		friend std::ostream & operator << (std::ostream &os, const Vector& vec)
		{ return os << vec.x << " " << vec.y << " " << vec.z;}

		static Vector max()
		{ return Vector(std::numeric_limits<T>::max(), std::numeric_limits<T>::max()); }

		static Vector infinity()
		{ return Vector(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity()); }

		T get_max() const
		{ return std::max(x, std::max(y, z)); }

		T get_min() const
		{ return std::min(x, std::min(y, z)); }

		T get_abs_max()
		{ return std::max(fabs(x), std::max(fabs(y), fabs(z))); }

		void write_to(T* p) const
		{ p[0] = x, p[1] = y, p[2] = z; }

		static Vector get_zero()
		{ return Vector(0, 0, 0); }

		// i'm norm
		Vector reflection(const Vector& v) const {
			m_assert(fabs(v.sqr() - 1) < EPS && (sqr() - 1 < EPS));
			return *this * 2 * dot(v) - v;
		}
};


template<typename T>
class Vector2D {
    public:
        T x = 0, y = 0;

        Vector2D<T>(){};

        explicit Vector2D<T>(T m_x, T m_y):
            x(m_x), y(m_y) {}

        Vector2D<T> (const Vector2D<T> &p0, const Vector2D<T> &p1):
            x(p1.x - p0.x), y(p1.y -p0.y) {}

        T dot(const Vector2D<T> &v) const
        { return x * v.x + y * v.y; }

        T cross(const Vector2D<T> &v) const
        { return x * v.y - y * v.x; }

        Vector2D<T> operator + (const Vector2D<T> &v) const
        { return Vector2D<T>(x + v.x, y + v.y); }

        Vector2D<T>& operator += (const Vector2D<T> &v)
        { x += v.x; y += v.y; return *this; }

        Vector2D<T> operator - (const Vector2D<T> &v) const
        { return Vector2D<T>(x - v.x, y - v.y); }

        Vector2D<T> operator - () const
        { return Vector2D<T>(-x, -y); }

        Vector2D<T>& operator -= (const Vector2D<T> &v)
        { x -= v.x; y -= v.y; return *this; }

        Vector2D<T> operator * (T f) const
        { return Vector2D<T>(x * f, y * f); }

        Vector2D<T>& operator *= (T p)
        { x *= p; y *= p; return *this; }

        Vector2D<T> operator / (T f) const
        { return *this * (1.0 / f); }

        Vector2D<T> operator * (const Vector2D<T>& v) const
        { return Vector2D<T>(x * v.x, y * v.y); }

        Vector2D<T> operator / (const Vector2D<T>& v) const
        { return Vector2D<T>(x / v.x, y / v.y); }

        bool operator == (const Vector2D<T> &v) const
        { return fabs(x - v.x) < EPS && fabs(y - v.y) < EPS; }

        // take negative of the second component
        Vector2D<T> operator ! () const
        { return Vector2D<T>(x, -y); }

        // swap the two component
        Vector2D<T> operator ~ () const
        { return Vector2D<T>(y, x); }

        bool is_zero() const
        { return fabs(x) < EPS && fabs(y) < EPS; }

        T sqr() const
        { return x * x + y * y; }

        T mod() const
        { return hypot(x, y); }

        Vector2D<T> get_normalized() const {
            T m = mod();
            m_assert(m > EPS);
            m = 1.0 / m;
            return Vector2D<T>(x * m, y * m);
        }

        void normalize() {
            T m = (T)1.0 / mod();
            x *= m, y *= m;        // work?
            m_assert(std::isnormal(m));
        }

        template <typename TT>
        friend std::ostream& operator << (std::ostream& os, const Vector2D<TT>& v);

        void update_min(const Vector2D<T> &v)
        { ::update_min(x, v.x); ::update_min(y, v.y);}

        void update_max(const Vector2D<T> &v)
        { ::update_max(x, v.x); ::update_max(y, v.y);}

        bool isNaN() const { return std::isnan(x); }

        static Vector2D<T> NaN() { return Vector2D<T>(NAN, NAN); }

        static Vector2D<T> max()
        { return Vector2D<T>(
                std::numeric_limits<T>::max(),
                std::numeric_limits<T>::max()); }
};

template<typename T>
std::ostream& operator << (std::ostream& os, const Vector2D<T>& v) {
    os << v.x << ' ' << v.y;
    return os;
}


typedef Vector<double> Vec;
typedef Vector2D<int> Coor;
typedef Vector2D<double> Vec2D;



//=============
//File: time.hh
//=============


class Timer {
    public:
        using Clock = std::chrono::high_resolution_clock;
        Timer() {
            restart();
        }

        // return current unix timestamp
        void restart() {
            m_start_time = std::chrono::high_resolution_clock::now();
        }

        // return duration in seconds
        double duration() const {
            auto now = std::chrono::high_resolution_clock::now();
            auto m = std::chrono::duration_cast<std::chrono::microseconds>(now - m_start_time).count();
            return m * 1.0 / 1e6;
        }

    protected:
        std::chrono::time_point<Clock> m_start_time;

};


class GuardedTimer: public Timer {
    public:
        GuardedTimer(const std::string& msg,  bool enabled=true):
            GuardedTimer([msg](double duration){
                    std::cout << msg << ": " << std::to_string(duration * 1000.) << " milliseconds." << std::endl;
                })
        { enabled_ = enabled; }

        GuardedTimer(const char* msg, bool enabled=true):
            GuardedTimer(std::string(msg), enabled) {}

        GuardedTimer(std::function<void(double)> callback):
            m_callback(callback)
        { }

        ~GuardedTimer() {
            if (enabled_)
                m_callback(duration());
        }

    protected:
        bool enabled_;
        std::function<void(double)> m_callback;

};

// record the total running time of a region across the lifecycle of the whole program
// call TotalTimer::print() before exiting main()
class TotalTimer {
    public:
        TotalTimer(const std::string& msg):
            msg(msg) {
                timer.restart();
            }

        ~TotalTimer();

        static void print();

        std::string msg;
        Timer timer;

        static std::map<std::string, std::pair<int, double>> rst;
};

// Build a global instance of this class, to call print() before program exit.
struct TotalTimerGlobalGuard
{
    ~TotalTimerGlobalGuard() { TotalTimer::print(); }
};


#define GUARDED_FUNC_TIMER \
    GuardedTimer _long_long_name_guarded_timer(__func__)

#define TOTAL_FUNC_TIMER \
    TotalTimer _long_long_name_total_timer(__func__)


//================
//File: imgproc.hh
//================


Mat32f read_img(const char* fname);
void write_rgb(const char* fname, const Mat32f& mat);
inline void write_rgb(const std::string s, const Mat32f& mat) { write_rgb(s.c_str(), mat); }

template <typename T>
void resize(const Mat<T> &src, Mat<T> &dst);

Mat32f crop(const Mat32f& mat);

Mat32f rgb2grey(const Mat32f& mat);

//=================
//File: gaussian.hh
//=================

class GaussCache {
	public:
		std::unique_ptr<float, std::default_delete<float[]>> kernel_buf;
		float* kernel;
		int kw;
		GaussCache(float sigma);
};

class GaussianBlur {
	float sigma;
	GaussCache gcache;
	public:
		GaussianBlur(float sigma): sigma(sigma), gcache(sigma) {}

		// TODO faster convolution
		template <typename T>
		Mat<T> blur(const Mat<T>& img) const {
			m_assert(img.channels() == 1);
			TotalTimer tm("gaussianblur");
			const int w = img.width(), h = img.height();
			Mat<T> ret(h, w, img.channels());

			const int kw = gcache.kw;
			const int center = kw / 2;
			float * kernel = gcache.kernel;

			std::vector<T> cur_line_mem(center * 2 + std::max(w, h), 0);
			T *cur_line = cur_line_mem.data() + center;

			// apply to columns
			REP(j, w){
				const T* src = img.ptr(0, j);
				// copy a column of src
				REP(i, h) {
					cur_line[i] = *src;
					src += w;
				}

				// pad the border with border value
				T v0 = cur_line[0];
				for (int i = 1; i <= center; i ++)
					cur_line[-i] = v0;
				v0 = cur_line[h - 1];
				for (int i = 0; i < center; i ++)
					cur_line[h + i] = v0;

				T *dest = ret.ptr(0, j);
				REP(i, h) {
					T tmp{0};
					for (int k = -center; k <= center; k ++)
						tmp += cur_line[i + k] * kernel[k];
					*dest = tmp;
					dest += w;
				}
			}

			// apply to rows
			REP(i, h) {
				T *dest = ret.ptr(i);
				memcpy(cur_line, dest, sizeof(T) * w);
				{	// pad the border
					T v0 = cur_line[0];
					for (int j = 1; j <= center; j ++)
						cur_line[-j] = v0;
					v0 = cur_line[w - 1];
					for (int j = 0; j < center; j ++)
						cur_line[w + j] = v0;
				}
				REP(j, w) {
					T tmp{0};
					for (int k = -center; k <= center; k ++)
						tmp += cur_line[j + k] * kernel[k];
					*(dest ++) = tmp;
				}
			}
			return ret;
		}
};

class MultiScaleGaussianBlur {
	std::vector<GaussianBlur> gauss;		// size = nscale - 1
	public:
	MultiScaleGaussianBlur(
			int nscale, float gauss_sigma,
			float scale_factor) {
		REP(k, nscale - 1) {
			gauss.emplace_back(gauss_sigma);
			gauss_sigma *= scale_factor;
		}
	}

	Mat32f blur(const Mat32f& img, int n) const
	{ return gauss[n - 1].blur(img); }
};




//============
//File: dog.hh
//============

// Given an image, build an octave with different blurred version
class GaussianPyramid {
	private:
		int nscale;
		std::vector<Mat32f> data; // len = nscale
		std::vector<Mat32f> mag; // len = nscale
		std::vector<Mat32f> ort; // len = nscale, value \in [0, 2 * pi]

		void cal_mag_ort(int);

	public:
		int w, h;

		GaussianPyramid(const Mat32f&, int num_scale);

		inline const Mat32f& get(int i) const { return data[i]; }

		inline const Mat32f& get_mag(int i) const { return mag[i]; }

		inline const Mat32f& get_ort(int i) const { return ort[i]; }

		int get_len() const { return nscale; }
};

class ScaleSpace
{
    public:
        int noctave, nscale;
        const int origw, origh;
        std::vector<GaussianPyramid> pyramids;	// len = noctave

        ScaleSpace(const Mat32f&, int num_octave, int num_scale);
        ScaleSpace(const ScaleSpace&) = delete;
        ScaleSpace& operator = (const ScaleSpace&) = delete;
};


class DOGSpace {

    public:
        // Calculate difference of a list of image
        // diff[0] = orig[1] - orig[0]

        typedef std::vector<Mat32f> DOG;    // len = nscale - 1

        int noctave, nscale;
        int origw, origh;

        std::vector<DOG> dogs;        // len = noctave

        DOGSpace(const DOGSpace&) = delete;
        DOGSpace& operator = (const DOGSpace&) = delete;

        Mat32f diff(const Mat32f& img1, const Mat32f& img2) const;
        DOGSpace(ScaleSpace&);

};


//=============
//File: sift.hh
//=============

// A Scale-Space point. used as intermediate result
struct SSPoint
{
    Coor coor;                        // integer coordinate in the pyramid
    Vec2D real_coor;            // scaled [0,1) coordinate in the original image
    int pyr_id, scale_id; // pyramid / scale id
    float dir;
    float scale_factor;
};

struct Descriptor
{
    Vec2D coor;
    std::vector<float> descriptor;

    // square of euclidean. use now_thres to early-stop
    float euclidean_sqr(const Descriptor& r,
                        float now_thres)
    {
        return euclidean_sqr(descriptor.data(),
                             r.descriptor.data(),
                             (int)descriptor.size(),
                             now_thres);
    }

    float euclidean_sqr(const float* x,
                        const float* y,
                        size_t size,
                        float now_thres);

    int hamming(const Descriptor& r)
    {
        return hamming(descriptor.data(),
                       r.descriptor.data(),
                       (int)descriptor.size());
    }

    int hamming(const float* x, const float* y, int n);
};


// sift algorithm implementation
class SIFT
{
  public:
    SIFT(const ScaleSpace& ss,
         const std::vector<SSPoint>& keypoints);
    SIFT(const SIFT&) = delete;
    SIFT& operator = (const SIFT&) = delete;
    std::vector<Descriptor> get_descriptor() const;
  protected:
    const ScaleSpace& ss;
    const std::vector<SSPoint>& points;
    Descriptor calc_descriptor(const SSPoint&) const;
};


//====================
//File: orientation.hh
//====================


class OrientationAssign {
    public:
        OrientationAssign(
                const DOGSpace& dog, const ScaleSpace& ss,
                const std::vector<SSPoint>& keypoints);

        OrientationAssign(const OrientationAssign&) = delete;
        OrientationAssign& operator = (const OrientationAssign&) = delete;

        // assign orientation to each SSPoint
        std::vector<SSPoint> work() const;

    protected:
        const DOGSpace& dog;
        const ScaleSpace& ss;
        const std::vector<SSPoint>& points;

        std::vector<float> calc_dir(const SSPoint& p) const;
};


//================
//File: extrema.hh
//================


class ExtremaDetector {
    public:
        explicit ExtremaDetector(const DOGSpace&);

        ExtremaDetector(const ExtremaDetector&) = delete;
        ExtremaDetector& operator = (const ExtremaDetector&) = delete;

        std::vector<SSPoint> get_extrema() const;

        // return extrema in global coor
        std::vector<Coor> get_raw_extrema() const;

    protected:
        const DOGSpace& dog;

        // return extrema in local coor
        std::vector<Coor> get_local_raw_extrema(int pyr_id, int scale_id) const;

        // calculate keypoint offset of a point in scalespace
        // and remove low contrast
        // See Sec.4, Accurate Keypoint Localization of Lowe,IJCV04
        bool calc_kp_offset(SSPoint* sp) const;

        std::pair<Vec, Vec> calc_kp_offset_iter(
                const DOGSpace::DOG& now_pyramid,
                int newx, int newy, int news) const;

        // Eliminating edge responses. Sec 4.1 of Lowe,IJCV04
        bool is_edge_response(Coor coor, const Mat32f& img) const;
};


//================
//File: feature.hh
//================


class FeatureDetector {
    public:
        FeatureDetector() = default;
        virtual ~FeatureDetector() = default;
        FeatureDetector(const FeatureDetector&) = delete;
        FeatureDetector& operator = (const FeatureDetector&) = delete;

        // return [-w/2,w/2] coordinated
        std::vector<Descriptor> detect_feature(const Mat32f& img) const;
        virtual std::vector<Descriptor> do_detect_feature(const Mat32f& img) const = 0;
};

class SIFTDetector : public FeatureDetector {
    public:
        std::vector<Descriptor> do_detect_feature(const Mat32f& img) const override;
};



//=================
//File: imageref.hh
//=================


// A transparent reference to a image in file
struct ImageRef
{
    std::string fname;
    Mat32f* img = nullptr;
    int _width, _height;

    ImageRef(const std::string& fname): fname(fname) {}
    ~ImageRef() { if (img) delete img; }

    void load()
    {
        if (img)
        {
            return;
        }
        img = new Mat32f{read_img(fname.c_str())};
        _width = img->width();
        _height = img->height();
    }

    void release()
    {
        if (img)
        {
            delete img;
        }
    img = nullptr;
    }

    int width() const
    {
        return _width;
    }

    int height() const
    {
        return _height;
    }

    //Shape2D shape() const
    //{
    //    return {_width, _height};
    //}
};
