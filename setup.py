from setuptools import setup
from setuptools.extension import Extension
import numpy

extensions = [
    Extension(
    name="SeqSeg", # name/path of generated .so file
    sources=["SeqSeg/SeqSeg.c"], # cython generated c file
    include_dirs = [numpy.get_include()]), # gives access to numpy funcs inside cython code 
]

setup(name='SeqSeg',
      version='1.0',
      description='...',
      author='Paulo Hubert',
      author_email='paulo.hubert@gmail.com',
      include_package_data=True,
      packages=['SeqSeg',
                'OceanPod',
      ],
      ext_modules = extensions,
)