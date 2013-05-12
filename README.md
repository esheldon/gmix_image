gmix_image
==========

Python code to fit a gaussian mixture model to an image using various
techniques.

- Expectation Maximization.
- Levenberg-Marquardt optimization
- MCMC, of the affine invariant variety.

Under the hood, C librares are used for speed.

Example using EM
================

    import gmix_image

    # initial guesses
    guess = [{'p':0.4,'row':10,'col':10,'irr':2.5,'irc':0.1,'icc':3.1},
             {'p':0.6,'row':15,'col':17,'irr':1.7,'irc':0.3,'icc':1.5}]

    # create the gaussian mixture
    gm = gmix_image.GMixEM(image, guess, sky=100)

    # Work with the results
    flags=gm.get_flags()
    if flags != 0:
        print 'failed with flags:',flags

    numiter=gm.get_numiter()
    fdiff=gm.get_fdiff()
    print 'number of iterations:',numiter
    print 'fractional diff on last iteration:',fdiff

    gmix=gm.get_gmix()
    
    model=gmix_image.gmix2image(gmix, image.shape)

