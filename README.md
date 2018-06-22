# Sequential Segmentation Algorithm

This is the Python + Cython implementation of the Bayesian Sequential Segmentation Algorithm.
If you use this algorithm, please cite:

Hubert, P., Padovese, L., Stern, J.M. A Sequential Algorithm for Signal Segmentation, Entropy 20 (55)  (2018) https://doi.org/10.20944/preprints201712.0001.v1

Hubert, P., Padovese, L., Stern, J.M. Fast Implementation of a Bayesian unsupervised segmentation algorithm, arXiv:1803.01801

The module is composed of two classes: 

OceanPod - an interface class to allow easy reading and processing of files from the OceanPod hydrophone

SeqSeg - the interface for the segmentation algortihm

In this repository we also included 3 signal files for testing, in the folder Data/. Please see main.py and the paper for details.

On folder notebooks/ we included one full replication script for the papers' results, and also two auxiliary notebooks that can be used as examples or tutorials.

Finally, we included two Matlab scripts, written by ourselves, to recreate the spectrogram plots that appear in the paper. In order to use LB (Light and Bartlein) colormaps on MATLAB you'll need to download https://www.mathworks.com/matlabcentral/fileexchange/17555-light-bartlein-color-maps.


## System requirements:

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

Enter the folder created by the last command (e.g. ~/gsl-2.5):

```
$cd ~/gsl-2.5

$./configure

$make

$make check

$sudo make install
```

These commands will insert the include files for GSL in folder /usr/local/include. If using a different include path, C_INCLUDE_PATH environment variable should be modified accordingly.

2. Run install script

```
python3 setup.py install
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

OBS: Adjust the -I flags on gcc where necessary.

```
$cython SeqSeg.pyx

$gcc -m64 -pthread -fno-strict-aliasing -fopenmp -Wstrict-prototypes -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -I/usr/local/include -I/usr/include/python3.5m -c SeqSeg.c -o build/SeqSeg.o

$gcc -fopenmp -pthread -shared -L/usr/local/lib/ -L/usr/lib/python3.5 -o SeqSeg.so  build/SeqSeg.o -lpython2.7  -lgsl -lgslcblas -lm
```
