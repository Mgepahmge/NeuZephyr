#pragma once

#ifdef _MSC_VER
    #ifdef DL_FRAMEWORK_EXPORTS
        #define DL_API __declspec(dllexport)
    #else
        #define DL_API __declspec(dllimport)
    #endif
#else
    #define DL_API
#endif