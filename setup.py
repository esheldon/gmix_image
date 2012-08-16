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

render_ext=Extension("gmix_image._render", 
                     ["gmix_image/render_pywrap.c",
                      "gmix_image/gvec.c",
                      "gmix_image/matrix.c",
                      "gmix_image/image.c"])

nlsolve_ext=Extension("gmix_image._gmix_nlsolve", 
                      ["gmix_image/NLSolver.cpp",
                       "gmix_image/gmix_nlsolve_pywrap.cpp"],
                      libraries=['tmv','tmv_symband'],
                      define_macros=[('USE_TMV',None)])


setup(name="gmix_image", 
      packages=['gmix_image'],
      version="1.0",
      data_files=data_files,
      ext_modules=[em_ext,render_ext,nlsolve_ext],
      include_dirs=numpy.get_include())
