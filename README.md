# caffe-DFP
This is Caffe-DFP used in JVET-I0022(http://phenix.it-sudparis.eu/jvet/) for network inference, which is modified based on Ristretto (https://github.com/pmgysel/caffe) and Caffe (https://github.com/BVLC/caffe)
## 1. Requirements
The following third-party dependencies are necessary：
* OpenBLAS, version:0.2.18
* protobuf
* gflags
* glog
* boost
* hdf5

To test with GPU:
* CUDA 8.0
* CUDNN

## 2. Installation and compilation
### 2.1 Linux Users:
#### 2.1.1 Dependency installation:
1) Install OpenBLAS, version:0.2.18

   * Download: https://github.com/xianyi/OpenBLAS/releases
   * Compile and Installation: https://github.com/xianyi/OpenBLAS
   
2) If one want to test with GPU, you need to install
   *  CUDA 8.0, https://developer.nvidia.com/cuda-toolkit-archive
   *  CUDNN, https://developer.nvidia.com/rdp/cudnn-archive
   
3) Install third-party dependencies

   The following provides two ways to install the dependency:
   
   *   Install from binary
    
        This way is easy, but may fail when compile JVET-I0022 with higher version GCC. For higher versions, refer to "Install from 
        source code" or Section 2.2.
    
        * CENTOS (Test pass with gcc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-11)):
      
          ``sudo yum install protobuf-devel gflags-devel glog-devel boost-devel hdf5-devel``
     
        * Ubuntu (Test pass with gcc (Ubuntu 4.8.4-2ubuntu1~14.04.3) 4.8.4):
      
          ``sudo apt-get install protobuf-devel gflags-devel glog-devel boost-devel hdf5-devel``
        
        * Others:
          
          Please refer to Section 2.2.
        
    *   Install from source code 
    
        If you have a much higher version of GCC, you need to install the above required libraries from source code or refer to Section 2.2. Install from binaries may lead to error due to compatibility.
      
        * protobuf
          * Download：https://github.com/google/protobuf
          * Installation：https://github.com/google/protobuf/blob/master/src/README.md
        * gflags：
          * Download：https://github.com/gflags/gflags
          * Installation：https://github.com/gflags/gflags/blob/master/INSTALL.md
        * glog：
          * Download：https://github.com/google/glog
          * Installation：https://github.com/google/glog/blob/master/INSTALL
        * boost
          * Download: https://github.com/boostorg/boost
          * Installation: https://github.com/boostorg/boost/blob/master/INSTALL
        * hdf5
          * Download: https://www.hdfgroup.org/downloads/hdf5/
          * Installation: section "Support" in https://www.hdfgroup.org/downloads/hdf5/          
#### 2.1.2 Configuration:
Editing CMakeList.txt:
* Set option "BUILD_python" to OFF, if you do not have python installed.
* Set option "BUILD_matlab" to OFF, if you do not have matlabb installed.
* Set option "BUILD_python_layer" to OFF, if you do not have python installed or you do not want to build a neural network with python.
* Set option "USE_OPENCV" to OFF, if you do not have OPENCV installed.
* Set option "USE_LEVELDB" to OFF, if you do not have LEVELDB installed.
* Set option "USE_LMDB" to OFF, if you do not have LMDB installed.

#### 2.1.3 Compilation:
* For testing with CPU:

  ```
  cd caffe-DFP
  
  mkdir build
  
  cd build
  
  cmake -DCMAKE_BUILD_TYPE=Release -DCPU_ONLY=ON -DBLAS=Open ..
  ```
* For testing with GPU:

  ```
  cd caffe-DFP
  
  mkdir build
  
  cd build
  
  cmake -DCMAKE_BUILD_TYPE=Release -DCPU_ONLY=OFF -DBLAS=Open ..
 
  ```
### 2.2 Other users:
#### 2.2.1 Dependency installation:
Please refer to http://caffe.berkeleyvision.org/installation.html

Note that BLAS library has large impacts on testing time. The BLAS library we used in JVET-I0022 is OpenBLAS 0.2.18 and cuDNN 5.1.10.
#### 2.2.2 Source code modification:
The baseline caffe we used: https://github.com/Hikvision-Codec/Caffe_DFP/tree/Caffe_Baseline

By comparing it with Caffe-DFP, one can obtain the difference. After that, one can migrate the difference to any caffe installed.

Note that in the above website of Caffe_Baseline, you can find a button named "compare". With it, one can directly observe the difference. Another way is that one can download Caffe_Baseline and Caffe_DFP. With the help of SVN, it is easy to find the difference.
#### 2.2.3 Configuration:
The following tools are necessary to be configured via editing CMakeList.txt, Makefile, or options in using cmake command:
* Configure CPU_ONLY: turn it on when you test with CPU; turn it off when you test with GPU.
* Configure BLAS library to OpenBLAS since BLAS libraries have large impact on testing time 

#### 2.2.4 Compilation:
* Linux : http://caffe.berkeleyvision.org/installation.html#compilation
* Windows: https://github.com/BVLC/caffe/tree/windows
## 3. Testing with single thread on CPU
For fair comparison with JEM, ensure single thread is used for OpenBLAS during testing with CPU. The following gives an example for Linux users：
```
#execute the command in a terminal before testing

export OPENBLAS_NUM_THREADS=1

export GOTO_NUM_THREADS=1

export OMP_NUM_THREADS=1
```
## 4. Licence
Caffe, Ristretto and Caffe-DFP is released under the BSD 2-Clause license. The BAIR/BVLC reference models are released for unrestricted use.

If you have any questions or bug reports, please do not hesitate to contact songxiaodan@hikvision.com.
