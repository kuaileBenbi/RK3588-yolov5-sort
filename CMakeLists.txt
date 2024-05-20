cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5sort_demo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5sort_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/include/lib/librknnrt.so)
set(RGA_LIB ${CMAKE_SOURCE_DIR}/include/lib/librga.so)

find_package(OpenCV REQUIRED)

set(CMAKE_INSTALL_RPATH "lib")

include_directories( ${CMAKE_SOURCE_DIR}/include)
include_directories( ${CMAKE_SOURCE_DIR}/include/rga)
include_directories( ${CMAKE_SOURCE_DIR}/include/rknn_api)
include_directories( ${CMAKE_SOURCE_DIR}/include/yolo)
include_directories( ${CMAKE_SOURCE_DIR}/include/sort)


add_executable(rknn_yolov5sort_demo
        src/main.cc
        src/yolo/postprocess.cc
        src/yolo/preprocess.cc
        src/yolo/rkYolov5s.cc
)

# 开启调试选项
add_definitions("-g")

target_link_libraries(rknn_yolov5sort_demo
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
  ${RGA_LIB}
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5sort_demo${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_yolov5sort_demo DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)
install(DIRECTORY data DESTINATION ./)
