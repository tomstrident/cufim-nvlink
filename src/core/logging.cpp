#include "logging.h"

namespace tw { // =============================================================
// ----------------------------------------------------------------------------
void log_msg(const log_level level, const std::string message, const char *file, const int line, ...)
{
  //va_list argptr;
  //va_start(argptr, message.c_str());
  //vfprintf(stderr, message.c_str(), argptr);
  //va_end(argptr);

  // write to console
  if (level < TW_WARNING)
    printf("%s%s:%d: %s\n", level_head[level].c_str(), file, line, message.c_str());
  else
    printf("%s%s\n", level_head[level].c_str(), message.c_str());

  // write to file (add to queue)
  //sprintf();

  // if (level < WARNING)
  // throw exception, handle!

  /*
  log_to_console(level_head[level], message);

  if(global_logger == declared)
    global_logger.log(level_head[level], message);
  */
}
// ----------------------------------------------------------------------------
Logger::Logger()
{

}
// ----------------------------------------------------------------------------
Logger::~Logger()
{

}
// ----------------------------------------------------------------------------
/*
void Logger::log(const log_level level, const std::string message, const char *file, const int line)
{

}
*/
// ----------------------------------------------------------------------------
void test_logging()
{
  // testcase message?
  log_fatal("message");
  log_error("message");
  log_warning("message");
  log_info("message");
  log_debug("message");
  log_trace("message");
  log_success("message");
}
// ----------------------------------------------------------------------------
} // namespace tw =======================================================