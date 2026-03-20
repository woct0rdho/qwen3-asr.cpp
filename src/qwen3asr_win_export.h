#pragma once

#ifdef _WIN32
#    ifdef QWEN3ASR_EXPORTS
#        define QWEN3ASR_API __declspec(dllexport)
#    else
#        define QWEN3ASR_API __declspec(dllimport)
#    endif
#else
#    define QWEN3ASR_API
#endif