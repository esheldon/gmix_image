import distutils
from distutils.core import setup, Extension
import numpy

data_files=[]

em_ext=Extension("gmix_image._gmix_em", 
                 ["gmix_image/gmix_em_pywrap.c",
                  "gmix_image/gmix_em.c",
                  "gmix_image/gvec.c",
                  "gmix_image/image.c",
                  "gmix_image/matrix.c"])

fit_ext=Extension("gmix_image._gmix_fit", 
                  ["gmix_image/gmix_fit_pywrap.c",
                   "gmix_image/gvec.c",
                   "gmix_image/matrix.c",
                   "gmix_image/image.c"])

setup(name="gmix_image", 
      packages=['gmix_image'],
      version="1.0",
      data_files=data_files,
      ext_modules=[em_ext,fit_ext],
      include_dirs=numpy.get_include())
