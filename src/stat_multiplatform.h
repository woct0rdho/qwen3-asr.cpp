#pragma once

#include "qwen3asr_win_export.h"

namespace qwen3_asr {
#ifdef _WIN32
    #define fstat64 _fstat64
    #define stat64  _stat64
    typedef struct _stat64 stat64_t;
#else // UNIX
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <unistd.h>
    #define fstat64 fstat
    #define stat64 stat
#endif // _WIN32
}