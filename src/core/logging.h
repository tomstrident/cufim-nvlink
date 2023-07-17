/**
 * Copyright (c) 2022 Thomas Schrotter. All rights reserved.
 * @file logging.h
 * @brief 
 * @author Thomas Schrotter
 * @version 0.0.0
 * @date 2022-12
 */
// TODO =======================================================================
// - varadic input + formatting
// - log file output
// - logger class?
// - exception handling on error and fatal?
// - exclude color formatting for log file output
// - batch log file write
// - warning magenta?
// CODE =======================================================================
#ifndef TW_LOGGING_H_
#define TW_LOGGING_H_

// stdlib includes
#include <string>
#include <array>

// defines
#define LOG_WARNING_EN 1
#define LOG_INFO_EN    1
#define LOG_DEBUG_EN   1
#define LOG_TRACE_EN   1
#define LOG_SUCCESS_EN 1

#ifdef TW_RELEASE
#define LOG_DEBUG_EN 0
#define LOG_TRACE_EN 0
#endif

namespace tw { // =============================================================

enum log_level {TW_FATAL, TW_ERROR, TW_WARNING, TW_INFO, TW_DEBUG, TW_TRACE, TW_SUCCESS}; // info = status?

const static std::array<std::string, 7> level_head {
"\033[1;31m[FATAL]\033[0m: ",
"\033[1;31m[ERROR]\033[0m: ",
"\033[1;33m[WARNING]\033[0m: ",
"\033[1;36m[INFO]\033[0m: ",
"\033[1;35m[DEBUG]\033[0m: ",
"\033[1;34m[TRACE]\033[0m: ",
"\033[1;32m[SUCCESS]\033[0m: "};

void log_msg(const log_level level, const std::string message, const char *file, const int line, ...);

#define log_fatal(message, ...) log_msg(TW_FATAL, message, __FILE__, __LINE__, ##__VA_ARGS__);
#define log_error(message, ...) log_msg(TW_ERROR, message, __FILE__, __LINE__, ##__VA_ARGS__);

#if LOG_WARNING_EN == 1
#define log_warning(message, ...) log_msg(TW_WARNING, message, __FILE__, __LINE__, ##__VA_ARGS__);
#else
#define log_warning(message, ...)
#endif

#if LOG_INFO_EN == 1
#define log_info(message, ...) log_msg(TW_INFO, message, __FILE__, __LINE__,  ##__VA_ARGS__);
#else
#define log_info(message, ...)
#endif

#if LOG_DEBUG_EN == 1
#define log_debug(message, ...) log_msg(TW_DEBUG, message, __FILE__, __LINE__,  ##__VA_ARGS__);
#else
#define log_debug(message, ...)
#endif

#if LOG_TRACE_EN == 1
#define log_trace(message, ...) log_msg(TW_TRACE, message, __FILE__, __LINE__,  ##__VA_ARGS__);
#else
#define log_trace(message, ...)
#endif

#if LOG_SUCCESS_EN == 1
#define log_success(message, ...) log_msg(TW_SUCCESS, message, __FILE__, __LINE__,  ##__VA_ARGS__);
#else
#define log_success(message, ...)
#endif

// ----------------------------------------------------------------------------
// uses file io later on, allocation, MT performance?
// one global logger always static for assertions?
// logger only logs within scope (function, class)
// pass logger to assert function? (optional argument)
class Logger
{
private:
public:
  Logger();
  ~Logger();
  void log(const log_level level, const std::string message, const char *file, const int line);
};

// ----------------------------------------------------------------------------
void test_logging();

} // namespace tw -------------------------------------------------------
#endif // TW_LOGGING_H_ =======================================================