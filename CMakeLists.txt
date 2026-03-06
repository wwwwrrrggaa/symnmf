cmake_minimum_required(VERSION 4.0)
project(symnmf C)

set(CMAKE_C_STANDARD 11)

if(WIN32)
    set(Python3_ROOT_DIR "C:/dev/tools")
endif()

find_package(Python3 REQUIRED COMPONENTS Development)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(.)

add_executable(symnmf
        symnmf.c
        symnmf.h)

add_library(symnmf_module SHARED
        symnmfmodule.c
        symnmf.c
        symnmf.h)

target_link_libraries(symnmf_module Python3::Python)
