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

    print 'number of iterations:',gm.fdiff
    print 'fractional diff on last iteration:',gm.numiter

    pars = gm.pars
    print 'center for first guassian:',pars[0]['row'],pars[0]['col']

