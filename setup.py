import distutils
from distutils.core import setup, Extension, Command
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
gvec_ext=Extension("gmix_image._gvec", 
                     ["gmix_image/gvec_pywrap.c",
                      "gmix_image/gvec.c"])

ext_modules=[em_ext,render_ext,gvec_ext]

class WithNLSolver(Command):
    _ext_modules=ext_modules
    user_options=[]
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        nlsolve_ext=Extension("gmix_image._gmix_nlsolve", 
                              ["gmix_image/nlsolver.cpp",
                               "gmix_image/gmix_nlsolve_pywrap.cpp"],
                              libraries=['tmv','tmv_symband'],
                              define_macros=[('USE_TMV',None)])

        WithNLSolver._ext_modules.append(nlsolve_ext)

setup(name="gmix_image", 
      packages=['gmix_image'],
      cmdclass={"with_nlsolve": WithNLSolver},
      version="1.0",
      data_files=data_files,
      ext_modules=ext_modules,
      include_dirs=numpy.get_include())
