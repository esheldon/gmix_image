gmix_image
==========

Python code to fit a gaussian mixture model to an image using various
techniques.

- Expectation Maximization.
- Levenberg-Marquardt optimization
- MCMC, of the affine invariant variety.

Under the hood, C librares are used for speed.

The workhorses modules are

- gmix_image.gmix_fit
    - maximum likelihood fitters
- gmix_image.gmix_mcmc
    - MCMC fitters
- gmix_image.em
    - EM fitters
