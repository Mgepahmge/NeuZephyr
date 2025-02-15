//
// Created by Administrator on 24-11-27.
//

#ifndef UTILS_CUH
#define UTILS_CUH
#include <iostream>

#define TILE_SIZE 32
#define FULL_MASK 0xffffffffu
#define WARP_SIZE 32
#define MMA 16
#define CEIL(X) (((X) + 15) & ~15)


#define WARN(message)                                                 \
(std::cerr <<  "Warning: " << message                                 \
<< " (File: " << __FILE__ << ", Line: " << __LINE__                   \
<< std::endl)


#endif //UTILS_CUH
