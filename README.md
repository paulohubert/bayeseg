# Sequential Segmentation Algorithm

This is the Python + Cython implementation of the Bayesian Sequential Segmentation Algorithm.
If you use this algorithm, please cite:

Hubert, P., Padovese, L., Stern, J.M. A Sequential Algorithm for Signal Segmentation, Entropy 20 (55)  (2018) https://doi.org/10.20944/preprints201712.0001.v1

Hubert, P., Padovese, L., Stern, J.M. Fast Implementation of a Bayesian unsupervised segmentation algorithm, arXiv:1803.01801

The module is composed of two classes: 

OceanPod - an interface class to allow easy reading and processing of files from the OceanPod hydrophone

SeqSeg - the interface for the segmentation algortihm

In this repository we also included 3 signal files for testing, in the folder Data/. Please see main.py for details.

There are also two jupyter notebooks on /notebooks with the code used to generate tests of the Fast Implementation paper.




## System requirements:

GNU Scientific Library (GSL) >2.4
python >3.5.2
cython >0.27.3
cythonGSL >0.2.1
numpy >1.14.0


## To configure the cython + GSL environment:

1. Clone the repository

git clone http://github.com/paulohubert/bayeseg

2. Install cython

sudo apt-get install cython

3. Install GSL:

wget ftp://ftp.gnu.org/gnu/gsl/gsl-latest.tar.gz
mv gsl-latest.tar.gz ~/
cd 
tar -zxvf gsl-latest.tar.gz
mkdir (YOUR_PATH)/gsl
./configure --prefix=(YOUR_PATH)/gsl
make
make check
make install

4. Install CythonGSL

pip install CythonGSL


## To compile:

cython SeqSeg.pyx

gcc -m64 -pthread -fno-strict-aliasing -fopenmp -Wstrict-prototypes -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -I/usr/local/include -I/usr/include/python3.5m -c SeqSeg.c -o build/SeqSeg.o

gcc -fopenmp -pthread -shared -L/usr/local/lib/ -L/usr/lib/python3.5 -o SeqSeg.so  build/SeqSeg.o -lpython2.7  -lgsl -lgslcblas -lm

