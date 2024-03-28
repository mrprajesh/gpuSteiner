# gpuSteiner

[![Ubuntu/NVCC](https://github.com/mrprajesh/gpuSteiner/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/mrprajesh/gpuSteiner/actions/workflows/ubuntu.yml)


Accelerating Computation of Steiner Trees on GPUs at [https://doi.org/10.1007/s10766-021-00723-0](https://doi.org/10.1007/s10766-021-00723-0)


## Publication

**Accelerating Computation of Steiner Trees on GPUs**, 

<ins>Rajesh Pandian M </ins>, Rupesh Nasre. and N.S. Narayanaswamy,

*International Journal of Parallel Programming* **(IJPP)**, Volume 50, pages 152–185, **(2022)**
 
 [(DOI)](https://doi.org/10.1007/s10766-021-00723-0) [(Slides)](https://mrprajesh.co.in/pdfs/sem2-v4.pdf) [(Video)](https://youtu.be/BIecDhPdWaQ) [(Code)](https://github.com/mrprajesh/gpuSteiner)



## Requirements  

- Should work on every Linux Distribution. Tested on Ubuntu 22.04 and P100.
- Assumes Pascal+ Nvidia GPU and CUDA 10+ installed at default location. 
- GCC 7+

## How to use

### Build and run the executables

```
## To compile for GPU & seqCPU version
make

## To run on seqCPU
./2approxCpu2.out < tcSelected/instance137.gr

# to run on GPU
./gpuSteiner6-oddAgainWithKtimer2Sh3.out 16 < tcSelected/instance137.gr.txt

```

## Authors 
 * Rajesh Pandian M   | https://mrprajesh.co.in
 * Rupesh Nasre       | www.cse.iitm.ac.in/~rupesh
 * N.S.Narayanaswamy  | www.cse.iitm.ac.in/~swamy


# LICENSE
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
