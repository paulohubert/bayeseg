# Sequential Segmentation Algorithm

This is the Python + Cython implementation of the Bayesian Sequential Segmentation Algorithm.
If you use this algorithm, please cite:

Hubert, P., Padovese, L., Stern, J.M. A Sequential Algorithm for Signal Segmentation, Entropy 20 (55)  (2018) https://doi.org/10.20944/preprints201712.0001.v1

Hubert, P., Padovese, L., Stern, J.M. Fast Implementation of a Bayesian unsupervised segmentation algorithm, arXiv:1803.01801

The module is composed of two classes: 

OceanPod - an interface class to allow easy reading and processing of files from the OceanPod hydrophone

SeqSeg - the interface for the segmentation algortihm

The repository content is the following:

### Root folder

**setup.py**: the install script

### SeqSeg folder

**SeqSeg.pyx**: python code for the SeqSeg (segmentation) module

**SeqSeg.c**: the output of running cython on *SeqSeg.pyx*; this is the file that will be compiled by the setup script.

### OceanPod folder

**OceanPod.py**: python code for the OceanPod (interface with audio files) module


### Root (bayeseg) folder

**spectrograms.m**: MATLAB script to plot spectrograms

**spgram.m**: MATLAB function to calculate the spectrogram

Note: in order to use LB (Light and Bartlein) colormaps on MATLAB you'll need to download https://www.mathworks.com/matlabcentral/fileexchange/17555-light-bartlein-color-maps.

### notebooks Folder

**Calibration_real_samples.ipynb**: IPython notebook with an example of calibrating the SeqSeg algorithm on the real OceanPod samples. Illustrates both the usage of OceanPod and SeqSeg modules.

**FastImpl_calibration.ipynb**: IPython notebook illustrating the usage of the SeqSeg module with simulated data.

**MCMC_convergence.ipynb**: IPython notebook that reproduces the MCMC part of the SeqSeg module. To test the chain's convergence.

**Replication script.ipynb**: IPython notebook that allows exact replication of all the paper's results.


### Data folder

The .wav files obtained from the OceanPod (a hydrophone built at University of São Paulo - Brasil) with recordings from a depth of 20m. Each file corresponds to 15 minutes of audio, sampled at 11,025 Hz.

The .csv files obtained as the result of our segmentation algorithm for two of the samples. These files are input to the MATLAB script *spectrograms.m*. See paper for details.



## System requirements:

The SeqSeg.c file shipped with the package was created by cython, on Python 3.5 running on an Ubuntu 16.04 LTS operational system. The system requirements are as follows:

GNU Scientific Library (GSL) >= 2.4 

gcc >= 4.8.4

python >= 3.5.2

scipy >= 1.0.0

numpy >= 1.14.0

To compile the .pyx you'll also need

cython >= 0.27.3

cythonGSL >= 0.2.1


## To use the SeqSeg module

1. Install GSL:

```
$wget ftp://ftp.gnu.org/gnu/gsl/gsl-latest.tar.gz

$mv gsl-latest.tar.gz ~/

$cd 

$tar -zxvf gsl-latest.tar.gz
```

Access the folder created by the last command (e.g. ~/gsl-2.5):

```
$cd ~/gsl-2.5

$./configure

$make

$make check

$sudo make install
```

These commands will copy the GSL .h files to folder /usr/local/include, which is the standard include path. If using a different path, C_INCLUDE_PATH environment variable should be modified accordingly.

2. Run install script

```
python3 setup.py install
```

3. Start python and import the package

```
$python3
>>> from SeqSeg.SeqSeg import SeqSeg
```

## To compile the .pyx:


1. Install cython

```
$sudo apt-get install cython
```

2. Install CythonGSL

```
$pip3 install cythonGSL
```

3. Compile:

Note: Adjust the -I flags on gcc if necessary.

```
$cython SeqSeg.pyx

$gcc -m64 -pthread -fno-strict-aliasing -fopenmp -Wstrict-prototypes -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -I/usr/local/include -I/usr/include/python3.5m -c SeqSeg.c -o build/SeqSeg.o

$gcc -fopenmp -pthread -shared -L/usr/local/lib/ -L/usr/lib/python3.5 -o SeqSeg.so  build/SeqSeg.o -lpython2.7  -lgsl -lgslcblas -lm
```

## Troubleshooting

When importing SeqSeg module, if you get an error related to libgsl not being found, try updating your LD_LIBRARY_PATH as follows:

```
$export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

When compiling the .pyx, errors about usage of python objects inside nogil sections, if being raised by calls to gsl functions, indicate that cython cannot find the GSL include files. Check the path and adjust gcc -I accordingly.


