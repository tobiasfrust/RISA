# RISA - Rofex In-Situ Analysis

Prerequisites:
- cmake (> 3.5)
- CUDA and compatible host compiler (e.g. gcc)
- libconfig++
- Boost (> 1.58)
- LibTiff

Installation on Ubuntu 16.04
- install libConfig:
    ```sudo apt-get install libconfig++-dev```
- install cmake:
    ```sudo apt-get install cmake cmake-curses-gui```
- install boost:
    ```sudo apt-get install libboost-all-dev```
- install libtiff:
    ```sudo apt-get install libtiff5-dev```
- install CUDA (download from NVIDIA homepage: https://developer.nvidia.com/cuda-toolkit and follow the  instructions)

Building the software:
- create a new path, outside of src-directory, called e.g. 'build'
- move into this path and enter following command in terminal:
    ```ccmake <path_to_src>```
    e.g.
    ```ccmake ../RISA/.```
- check if everything could be found and enter ```CMAKE_BUILD_TYPE```, options are:
    ```Debug, RelWithDebInfo, Release```
- if everything workout, make the project
    ```make -j all```
- if build was successful, there is an executable in the build/bin folder
