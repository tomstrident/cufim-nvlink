/**
* Copyright (c) 2022 Thomas Schrotter. All rights reserved.
* @file excepts.h
* @brief 
* @author Thomas Schrotter
* @version 0.0.0
* @date 2022-12
*/
// TODO =======================================================================
// - breakpoint asserts
// - exception logic?
// CODE =======================================================================
#ifndef TW_EXCEPTS_H_
#define TW_EXCEPTS_H_

#include <stdexcept>
#include <string>
#include <cassert>

#define TW_ASSERTS_EN

#ifdef TW_ASSERTS_EN
#if _MSC_VER
#include <intrin.h>
#define debugBreak() __debugbreak()
#else
#define debugBreak() __builtin_trap() // raise(SIGTRAP);
#endif

void report_assertion_failure(const char* expression, const char* message, const char* file, int line);

#define tw_assert(expr)                                      \
{                                                            \
  if (expr) {                                                \
  } else {                                                   \
    report_assertion_failure(#expr, "", __FILE__, __LINE__); \
    debugBreak();                                            \
  }                                                          \
}

#define tw_assert_msg(expr, message)                              \
{                                                                 \
  if (expr) {                                                     \
  } else {                                                        \
    report_assertion_failure(#expr, message, __FILE__, __LINE__); \
    debugBreak();                                                 \
  }                                                               \
}

#ifdef _DEBUG
#define tw_assert_debug(expr)                                \
{                                                            \
  if (expr) {                                                \
  } else {                                                   \
    report_assertion_failure(#expr, "", __FILE__, __LINE__); \
    debugBreak();                                            \
  }                                                          \
}
#else
#define tw_assert_debug(expr)  // Does nothing at all
#endif

#else
#define tw_assert(expr)               // Does nothing at all
#define tw_assert_msg(expr, message)  // Does nothing at all
#define tw_assert_debug(expr)         // Does nothing at all
#endif

namespace tw { // =============================================================
// ----------------------------------------------------------------------------
// exception maybe also be able to log?
class tw_exception : public std::runtime_error
{
private:
  std::string msg;
public:
  tw_exception(const std::string& arg, const std::string& file, const std::string& line) :
    std::runtime_error(arg)
  {
    msg = "\033[1;31mError: "+file+":"+line+": "+arg+"\033[0m";
  }
  ~tw_exception() throw() {}
  const char *what() const throw() { return msg.c_str(); }
};
#define throw_tw_error(arg) throw tw_exception(arg, std::string(__FILE__), std::to_string(__LINE__));

// ----------------------------------------------------------------------------
void print_head();
// ----------------------------------------------------------------------------
/**
* @brief prints a given string (cyan)
*
* @param [in] s string to print
*/
//void print_status(const std::string& s);
// ----------------------------------------------------------------------------
/**
* @brief prints a given string (green)
*
* @param [in] s string to print
*/
//void print_success(const std::string& s);
// ----------------------------------------------------------------------------
/**
* @brief prints a given string (yellow)
*
* @param [in] s string to print
*/
//void print_warning(const std::string& s);
// ----------------------------------------------------------------------------

} // namespace tw

#endif // TW_EXCEPTS_H_