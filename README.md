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
#### 2.1.1 Installation:
1) Install OpenBLAS, version:0.2.18

   * Download: https://github.com/xianyi/OpenBLAS/releases
   * Compile and Installation: https://github.com/xianyi/OpenBLAS
   
2) If one want to test with GPU, you need to install
   *  CUDA 8.0, https://developer.nvidia.com/cuda-toolkit-archive
   *  CUDNN, https://developer.nvidia.com/rdp/cudnn-archive
   
3) Install third-party dependencies

   The following provides two ways to install the dependency:
   
   *   Install from binary
    
        This way is easy, but may fail when compiled JVET-I0022 with higher version GCC. For higher versions, refer to "Install from 
        source code" or 2.2.
    
        * CENTOS (Test pass with gcc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-11)):
      
          ``sudo yum install protobuf-devel gflags-devel glog-devel boost-devel hdf5-devel``
     
        * Ubuntu (Test pass with gcc (Ubuntu 4.8.4-2ubuntu1~14.04.3) 4.8.4):
      
          ``sudo apt-get install protobuf-devel gflags-devel glog-devel boost-devel hdf5-devel``
        
        * Others:
          
          Please refer to Section 2.2.
        
    *   Install from source code : 
    
        If you have a much higher version of GCC, you need to install the above required libraries from source code or refer to Section 2.2. Install from 
        binaries may lead to error due to compatibility.
      
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

#### 2.1.2 Compilation:
* CPU for test:

  ```
  cd caffe-DFP
  
  mkdir build
  
  cd build
  
  cmake -DCMAKE_BUILD_TYPE=Release -DCPU_ONLY=ON -DBLAS=Open ..
  ```
* GPU for test:

  ```
  cd caffe-DFP
  
  mkdir build
  
  cd build
  
  cmake -DCMAKE_BUILD_TYPE=Release -DCPU_ONLY=OFF -DBLAS=Open ..
 
  ```
### 2.2 Other users:
#### 2.2.1 Caffe Installation:
Please refer to http://caffe.berkeleyvision.org/installation.html

Note that BLAS library has large impacts on testing time. The BLAS library we used in JVET-I0022 is OpenBLAS 0.2.18 and cuDNN 5.1.10.
#### 2.2.2 Source code modification:
The following gives the baseline caffe we used:

``
#to be added
``

By comparing it with Caffe-DFP, one can obtain the difference. One can migrate the difference to any caffe installed.

#### 2.2.3 Configuration:
In Caffe, the tools one used can be configured via editing CMakeList.txt, Makefile, or options in using cmake command. To obtain similar running time, pay attention to the following connfiguration.
* Configure CPU_ONLY: turn it on when you test with CPU; turn it off when you test with GPU;
* Configure BLAS library to OpenBLAS if one want to obtain similar time with JVET-I0022. 

#### 2.2.4 Compilation:
* Linux : http://caffe.berkeleyvision.org/installation.html#compilation
* Windows: https://github.com/BVLC/caffe/tree/windows
## 3. Licence
Caffe, Ristretto and Caffe-DFP is released under the BSD 2-Clause license. The BAIR/BVLC reference models are released for unrestricted use.
