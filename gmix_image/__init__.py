import gmix_image
"""
gmix_image

Defines a class to fit a gaussian mixture model to an image using Expectation
Maximization.  

See docs for gmix_image.GMix for more details.

The code is primarily in a C library. The GMix object is a convenience wrapper
for that code.
"""

from gmix_image import GMix
from gmix_image import GMIX_ERROR_NEGATIVE_DET
from gmix_image import GMIX_ERROR_MAXIT
from gmix_image import GMIX_ERROR_NEGATIVE_DET_SAMECEN
import test
