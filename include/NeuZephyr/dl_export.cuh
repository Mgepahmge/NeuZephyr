#pragma once

// Cross-platform export macro
#if defined(_WIN32) || defined(_WIN64)
    #ifdef NEUZEPHYR_EXPORTS
        #define DL_API __declspec(dllexport)
    #else
        #define DL_API __declspec(dllimport)
    #endif
#elif defined(__GNUC__) && __GNUC__ >= 4
    #define DL_API __attribute__((visibility("default")))
#else
    #define DL_API
#endif
