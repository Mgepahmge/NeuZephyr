//
// Created by Administrator on 24-11-27.
//

#ifndef UTILS_CUH
#define UTILS_CUH
#include <iostream>


#define WARN(message)                                                 \
(std::cerr <<  "Warning: " << message                                 \
<< " (File: " << __FILE__ << ", Line: " << __LINE__                   \
<< std::endl)



#endif //UTILS_CUH
