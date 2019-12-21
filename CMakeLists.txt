cmake_minimum_required(VERSION 3.14)
project(tf_ops)


set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_SHARED_LIBRARY_SUFFIX .so)
#set(CMAKE_CXX_FLAGS -march=native)
set(CMAKE_SHARED_LINKER_FLAGS "-fPIC")
set(CMAKE_BUILD_TYPE Release)
#find_package(Protobuf REQUIRED VERSION 3.9.2)
find_library(Protobuf_LIBS NAMES protobuf HINTS /Users/sunjiahe/miniconda3/lib/)
message(STATUS ${Protobuf_LIBS})

#add_definitions("-DGOOGLE_CUDA")
#set(CMAKE_CXX_COMPILER_ABI 0)
#set(CMAKE_VERBOSE_MAKEFILE ON)
#target_link_libraries(ops PRIVATE $ENV{TF_LIB_PATH}/python/_pywrap_tensorflow_internal.so)

#set(ENV{TF_INCLUDE_PATH} /Users/sunjiahe/miniconda3/lib/python3.6/site-packages/tensorflow/include)
set(ENV{TF_INCLUDE_PATH} /Users/sunjiahe/miniconda3/envs/tf2.0/lib/python3.6/site-packages/tensorflow_core/include)
add_library(ops SHARED reader/skip_gram_kernels.cc reader/to_zeros.cc reader/image_threshold_op.cc reader/upsample.h
        reader/upsample.cc reader/relu.cpp reader/relu.h  reader/my_reader_dataset_op.cc reader/custom_dataset_op.cpp
        reader/custom_dataset_op.h reader/deform_conv.cpp reader/deform_conv.h reader/prefetch_dataset_op.cpp
        reader/prefetch_dataset_op.h reader/prefetch_autotuner.cpp reader/prefetch_autotuner.h reader/stats_utils.cpp
        reader/stats_utils.h reader/name_utils.cpp reader/name_utils.h reader/dataset_utils.cpp reader/dataset_utils.h
        reader/deform_conv.cu.cc reader/repeat_dataset_op.cpp reader/repeat_dataset_op.h)
target_include_directories(ops PRIVATE $ENV{TF_INCLUDE_PATH})
target_link_libraries(ops PRIVATE /Users/sunjiahe/miniconda3/envs/tf2.0/lib/python3.6/site-packages/tensorflow_core/libtensorflow_framework.dylib)
#target_link_libraries(ops PRIVATE /Users/sunjiahe/miniconda3/lib/python3.6/site-packages/tensorflow/libtensorflow_framework.so)
#target_compile_definitions(ops PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)

#add_library(dcnv2 SHARED deformable_v2/modulated_deformable_convolution-inl.h deformable_v2/modulated_deformable_convolution.cc
#         deformable_v2/deformable_im2col.cuh)
#target_include_directories(dcnv2 PRIVATE $ENV{TF_INCLUDE_PATH})
#target_link_libraries(dcnv2 PRIVATE /Users/sunjiahe/miniconda3/lib/python3.6/site-packages/tensorflow/libtensorflow_framework.so)
#target_link_options(dcnv2 PRIVATE -undefined dynamic_lookup)
#
#
#add_library(message SHARED proto/search_request.pb.h proto/search_request.pb.cc)
#target_link_libraries(message PRIVATE ${Protobuf_LIBRARIES})
#target_include_directories(message PRIVATE ${Protobuf_INCLUDE_DIRS})
#
#add_subdirectory(solutions)

add_executable(a main.cpp)
#target_link_libraries(a PRIVATE ${Protobuf_LIBRARIES})
#target_include_directories(a PRIVATE ${Protobuf_INCLUDE_DIRS})
target_link_libraries(a PRIVATE ${Protobuf_LIBS})
target_compile_options(a PRIVATE -march=native)
target_include_directories(a PRIVATE $ENV{TF_INCLUDE_PATH})
target_link_libraries(a PRIVATE /Users/sunjiahe/miniconda3/envs/tf2.0/lib/python3.6/site-packages/tensorflow_core/libtensorflow_framework.dylib)

