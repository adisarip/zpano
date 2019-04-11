// File: debugutils.hh
// Date: Sat May 04 12:03:06 2013 +0800
// Author: Yuxin Wu


#pragma once
#include <string>
#include <iterator>

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

//#define print_debug(fmt, ...)

#define m_assert(expr)

#endif
