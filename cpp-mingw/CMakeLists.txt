cmake_minimum_required(VERSION 3.14)

project(myapp)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# include_directories("/nix/store/vsy4rn73szilj1k2lqdq786i18q1wjpz-x86_64-w64-mingw32-gcc-13.2.0/include/c++/13.2.0")
# include_directories("/nix/store/jbw84r41x9a52wl0iqbgrgxic4cgs4vq-mingw-w64-x86_64-w64-mingw32-11.0.1-dev/include/")

add_executable(myapp main.cpp)