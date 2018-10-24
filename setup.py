from setuptools import setup
from setuptools.extension import Extension
import numpy

extensions = [
    Extension(
    name="SeqSeg.SeqSeg", # name/path of generated .so file
    sources=["SeqSeg/SeqSeg.c"], # cython generated c file
    include_dirs = [numpy.get_include()], # gives access to numpy funcs inside cython code
    libraries = ["gsl", "gslcblas"], 
    extra_compile_args = ['-Wno-deprecated']),
]

setup(name='SeqSeg',
      version='1.0',
      description='Package for binary / sequential signal segmentation using Bayesian methods',
      author='Paulo Hubert',
      author_email='paulo.hubert@gmail.com',
      url='http?//github.com/paulohubert/bayeseg/', 
      keywords = 'changepoint signal segmentation bayesian',
      classifiers = [
          'Development Status :: 4 - Beta',
          'License :: GNU',
          'Programming Language :: Python :: 3.5'
      ],
      include_package_data=True,
      packages=['SeqSeg',
                'OceanPod',
      ],
      install_requires=['numpy>=1.14.0', 'scipy>=1.0.0'],
      ext_modules = extensions,
)
