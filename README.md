gmix_image
==========

Python code to fit a gaussian mixture model to an image using Expectation
Maximization.  Under the hood, a C library is used for speed.

Examples
========

    import gmix_image

    # initial guesses
    guess = [{'p':0.4,'row':10,'col':10,'irr':2.5,'irc':0.1,'icc':3.1},
             {'p':0.6,'row':15,'col':17,'irr':1.7,'irc':0.3,'icc':1.5}]

    # create the gaussian mixture
    gm = gmix_image.GMix(image, guess, sky=100, maxiter=2000, tol=1.e-6)

    # Work with the results
    if gm.flags != 0:
        print 'failed with flags:',gm.flags

    print 'number of iterations:',gm.numiter
    print 'fractional diff on last iteration:',gm.fdiff

    pars = gm.pars
    print 'center for first guassian:',pars[0]['row'],pars[0]['col']

    # Find the gaussian mixture accounting for a point spread function.  The
    # psf is just another gaussian mixture model.  The fit gaussian mixture
    # will thus be "pre-psf". Centers are not necessary for the psf.

    psf = [{'p':0.8,'irr':1.2,'irc':0.2,'icc':1.0},
           {'p':0.2,'irr':2.0,'irc':0.1,'icc':1.5}]
    gm = gmix_image.GMix(image, guess, psf=psf, sky=100)

    # run the test suite
    gmix_image.test()
    gmix_image.test(add_noise=True)
    gmix_image.test_psf(add_noise=False)
    gmix_image.test_psf_colocate(add_noise=True)

