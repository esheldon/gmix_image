import distutils
from distutils.core import setup, Extension, Command
import numpy

data_files=[]

em_ext=Extension("gmix_image._gmix_em", 
                 ["gmix_image/gmix_em_pywrap.c",
                  "gmix_image/gmix_em.c",
                  "gmix_image/gvec.c",
                  "gmix_image/image.c",
                  "gmix_image/matrix.c",
                  "gmix_image/jacobian.c"])

render_sources=["gmix_image/render_pywrap.c",
                "gmix_image/gvec.c",
                "gmix_image/image.c",
                "gmix_image/jacobian.c"]
render_ext=Extension("gmix_image._render", render_sources)
                     #extra_compile_args=['-funroll-loops'])
                     #extra_compile_args=['-ffast-math','-funroll-loops'])

ext_modules=[em_ext,render_ext]


setup(name="gmix_image", 
      packages=['gmix_image'],
      version="1.0",
      data_files=data_files,
      ext_modules=ext_modules,
      include_dirs=numpy.get_include())
