import distutils
from distutils.core import setup, Extension
import numpy

data_files=[]

ext=Extension("gmix_image._gmix_image", 
              ["gmix_image/gmix_image_pywrap.c",
               "gmix_image/gmix_image.c",
               "gmix_image/gvec.c",
               "gmix_image/image.c",
               "gmix_image/matrix.c"])
setup(name="gmix_image", 
      packages=['gmix_image'],
      version="1.0",
      data_files=data_files,
      ext_modules=[ext],
      include_dirs=numpy.get_include())
