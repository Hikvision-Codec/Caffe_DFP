# Windows version of Caffe_DFP
It is modified based on Ristretto (https://github.com/pmgysel/caffe) and caffe-windows (https://github.com/pmgysel/caffe/tree/windows).
## Windows Setup

### Requirements

 - Visual Studio 2013 or 2015
     - Technically only the VS C/C++ compiler is required (cl.exe)
 - [CMake](https://cmake.org/) 3.4 or higher (Visual Studio and [Ninja](https://ninja-build.org/) generators are supported)


 We assume that `cmake.exe` are on your `PATH`.
 
### Configuring and Building Caffe

The fastest method to get started with caffe on Windows is by executing the following commands in a `cmd` prompt (we use `C:\Projects` as a root folder for the remainder of the instructions):
```cmd
C:\Projects> git clone https://github.com/Hikvision-Codec/Caffe_DFP.git
C:\Projects> cd Caffe_DFP
C:\Projects\caffe> git checkout Caffe_DFP_Windows
:: Edit any of the options inside build_win.cmd to suit your needs
::The default setting is for VS 2015. If you have VS 2013, please modify "MSVC_VERSION" to 13 in line 72 of "script\build_win.cmd".
::By default, CPU is used for test. If one wants to test with GPU, please refer to caffe-windows (https://github.com/pmgysel/caffe/tree/windows)
C:\Projects\caffe> scripts\build_win.cmd
```
The `build_win.cmd` script will download the dependencies, create the Visual Studio project files (or the ninja build files) and build the Release configuration. By default all the required DLLs will be copied (or hard linked when possible) next to the consuming binaries. If you wish to disable this option, you can by changing the command line option `-DCOPY_PREREQUISITES=0`. The prebuilt libraries also provide a `prependpath.bat` batch script that can temporarily modify your `PATH` environment variable to make the required DLLs available.

If you have GCC installed (e.g. through MinGW), then Ninja will detect it before detecting the Visual Studio compiler, causing errors.  In this case you have several options:

- [Pass CMake the path](https://cmake.org/Wiki/CMake_FAQ#How_do_I_use_a_different_compiler.3F) (Set `CMAKE_C_COMPILER=your/path/to/cl.exe` and `CMAKE_CXX_COMPILER=your/path/to/cl.exe`)
- or Use the Visual Studio Generator by setting `WITH_NINJA` to 0 (This is slower, but may work even if Ninja is failing.)
- or uninstall your copy of GCC 

The path to cl.exe is usually something like 
`"C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin/your_processor_architecture/cl.exe".`
If you don't want to install Visual Studio, Microsoft's C/C++ compiler [can be obtained here](http://landinghub.visualstudio.com/visual-cpp-build-tools). 

Below is a more complete description of some of the steps involved in building caffe.

### Install the caffe dependencies

By default CMake will download and extract prebuilt dependencies for your compiler and python version. It will create a folder called `libraries` containing all the required dependencies inside your build folder. Alternatively you can build them yourself by following the instructions in the [caffe-builder](https://github.com/willyd/caffe-builder) [README](https://github.com/willyd/caffe-builder/blob/master/README.md).


### Building only for CPU

If CUDA is not installed Caffe will default to a CPU_ONLY build. If you have CUDA installed but want a CPU only build you may use the CMake option `-DCPU_ONLY=1`.


### Building a shared library

CMake can be used to build a shared library instead of the default static library. To do so follow the above procedure and use `-DBUILD_SHARED_LIBS=ON`. Please note however, that some tests (more specifically the solver related tests) will fail since both the test executable and caffe library do not share static objects contained in the protobuf library.

### Troubleshooting

Should you encounter any error please post the output of the above commands by redirecting the output to a file and open a topic on the [caffe-users list](https://groups.google.com/forum/#!forum/caffe-users) mailing list.


## Further Details

Refer to the BVLC/caffe master branch README  and BVLC/caffe windows branch README for all other details such as license, citation, and so on.
