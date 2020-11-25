Instructions to build and execute for HW4:

1. Assuming the source and executable objects are unzipped into a folder
2. cd into that folder
3. 2 ways to run:
    A. Give compile & run commands on your own
        Compile: nvcc matrixNormCuda.cu -o matrixNormCuda
        Run:     ./matrixNormCuda
    B. Using the Makefile
        make
        ./matrixNormCuda

***CRITICAL NOTE:
    - Depending on your N size, you will have to change it INSIDE the source code on line 11
    - I ran my code successfully on Tesla V100-PCIE-16GB. This machine is an IIT machine: Mystic Cloud
        - Ubuntu 18.04.4 LTS
        - cudaRuntimeGetVersion() returns: 9010
        - cudaDriverGetVersion()  returns: 10020
        - nvcc --version          returns: Cuda compilation tools, release 9.1, V9.1.85