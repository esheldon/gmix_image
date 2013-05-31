"""
functions, mostly superceded by the GMix class

---------
total_moms:
    total moments of a gaussian mixture.

gmix_print:
    print out a gaussian mixture
"""

from sys import stdout,stderr
import math
import numpy
from numpy import zeros, array, where, ogrid, diag, sqrt, isfinite, \
        tanh, arctanh, cos, sin, exp, pi

from .gmix import GMix
from .gmix import gmix2pars

def srandu(num=None):
    return 2*(numpy.random.random(num)-0.5)


def measure_gmix_width(gmix0, thresh_vals, expand=8, debug=False):
    """
    Measure light-fraction width, e.g. 0.5 would be FWHM

    parameters
    ----------
    gmix: GMix
        A gausian mixture object
    thresh_vals: sequence
        threshold is, e.g. [0.25,0.5,0.75] which indicate pixels at that
        fraction of the max height
    expand: int, optional
        Expand the image by this factor to make pixelization less of an issue.

    output
    ------
    widths: ndarray
        sqrt(Area)/pi where Area is simply the number of pixels above
        the indicated threshold.
    """
    from .render import gmix2image

    expand=int(expand)
    if expand <1:
        expand=1

    if expand != 1:
        expand2 = expand**2
        dl=gmix0.get_dlist()
        for d in dl:
            d['irr'] = expand2*d['irr']
            d['irc'] = expand2*d['irc']
            d['icc'] = expand2*d['icc']
        gmix=GMix(dl)
    else:
        gmix=gmix0.copy()
    
    T=gmix.get_T()
    sig=sqrt(T/2)
    dim = 2*5*sig
    if (dim % 2) == 0:
        dim += 1

    dims=[dim]*2
    cen=[(dim-1)/2]*2

    gmix.set_cen(cen[0], cen[1])
    im=gmix2image(gmix, dims)

    im /= im.max()

    widths=numpy.zeros(len(thresh_vals))

    for i,thresh in enumerate(thresh_vals):
        w=numpy.where(im >= thresh)
        if w[0].size > 0:
            widths[i] = 2*sqrt(w[0].size/pi)

            if debug:
                print >>stderr,gmix
                _plot_gmix_thresh(im, cen, thresh, prompt=True)

    widths /= expand

    return widths

def _plot_gmix_thresh(im, cen, thresh, prompt=False):
    import images
    import biggles

    tab=biggles.Table(1,2)
    plt=biggles.FramedPlot()

    im_cenrows = im[:,cen[1]]
    im_cencols = im[cen[0],:]

    wrow,=numpy.where(im_cenrows >= thresh)
    wcol,=numpy.where(im_cencols >= thresh)

    hrow=biggles.Histogram(im_cenrows, x0=0, binsize=1,color='blue')
    hcol=biggles.Histogram(im_cencols, x0=0, binsize=1,color='red')
    plt.add(hrow,hcol)

    if wrow.size > 0:
        hrow_t=biggles.Histogram(im_cenrows[wrow], x0=wrow[0], binsize=1,
                                 color='blue',width=2)
        plt.add(hrow_t)
    if wcol.size > 0:
        hcol_t=biggles.Histogram(im_cencols[wcol], x0=wcol[0], binsize=1,
                                 color='red',width=2)
        plt.add(hcol_t)

    plt.title='thresh: %.2f' % thresh
    plt.aspect_ratio=1

    implt=images.view(im,show=False,type='dens-cont')
    tab[0,0]= implt
    tab[0,1] = plt
    tab.show()

    if prompt:
        key=raw_input('hit a key (q to quit): ')
        if key=='q':
            stop

def print_pars(pars, stream=stdout, fmt='%10.6g',front=None):
    """
    print the parameters with a uniform width
    """
    if front is not None:
        stream.write(front)
        stream.write(' ')
    if pars is None:
        stream.write('%s\n' % None)
    else:
        fmt = ' '.join( [fmt+' ']*len(pars) )
        stream.write(fmt % tuple(pars))
        stream.write('\n')

def check_jacobian(j):
    if not isinstance(j,dict):
        raise ValueError("jacobian must be a dict")
    for n in ['row0','col0','dudrow','dudcol','dvdrow','dvdcol']:
        if n not in j:
            raise ValueError("jacobian must contain '%s'" % n)


def total_moms(gauss_list, psf=None):
    """
    This is superceded by the GMix method get_T()


    parameters
    ----------
    gauss_list: 
        A gaussian mixture model as a list of dicts.
    psf: optional
        A PSF as a gaussian mixture model.  The result
        will be convolved with the PSF.
    """
    gm=GMix(gauss_list)
    if psf is not None:
        gmpsf=GMix(psf)
        gm=gm.convolve(gmpsf)

    return gm.get_T()
    """
    d={'irr':0.0, 'irc':0.0, 'icc':0.0}
    psum=0.0
    for g in gauss_list:
        p=g['p']
        psum += p
        d['irr'] += p*g['irr']
        d['irc'] += p*g['irc']
        d['icc'] += p*g['icc']

    d['irr'] /= psum
    d['irc'] /= psum
    d['icc'] /= psum
    return d
    """

def _total_moms_psf(gauss_list, psf_list):
    """
    Superceded by GMix method get_T()

    Only makes sense if the centers are the same
    """
    d={'irr':0.0, 'irc':0.0, 'icc':0.0}
    psf_totmom = total_moms(psf_list)

    psum=0.0
    for g in gauss_list:
        p=g['p']
        psum += p
        d['irr'] += p*(g['irr'] + psf_totmom['irr'])
        d['irc'] += p*(g['irc'] + psf_totmom['irc'])
        d['icc'] += p*(g['icc'] + psf_totmom['icc'])

    d['irr'] /= psum
    d['irc'] /= psum
    d['icc'] /= psum
    return d

def gmix_print(gmix, title=None):
    """
    Print a gaussian mixture.

    The gmix should be in list-of-dicts represenation
    """
    if title:
        print title
    hfmt = ['%10s']*6
    hfmt = ' '.join(hfmt)
    h = hfmt % ('p','row','col','irr','irc','icc')
    print h

    fmt = ['%10.6g']*6
    fmt = ' '.join(fmt)
    for g in gmix:
        print fmt % tuple([g[k] for k in ['p','row','col','irr','irc','icc']])

def pars2gmix(pars, coellip=False):
    """
    superceded by using GMix(pars)

    Convert a parameter array.  

    if coellip, the packing is
        [cen1,cen2,e1,e2,Tmax,Tfrac2,Tfrac3..,p1,p2,p3...]
    otherwise
        [p1,row1,col1,irr1,irc1,icc1,
         p2,row2,col2,irr2,irc2,icc2,
         ...]
    """

    if coellip:
        return _pars2gmix_coellip(pars)

    ngauss = len(pars)/6
    gmix=[]

    for i in xrange(ngauss):
        beg=i*6
        d={}

        d['p']   = pars[beg+0]
        d['row'] = pars[beg+1]
        d['col'] = pars[beg+2]
        d['irr'] = pars[beg+3]
        d['irc'] = pars[beg+4]
        d['icc'] = pars[beg+5]
        gmix.append(d)

    return gmix


def calculate_some_stats(image, ivar, gmix, npars, nsub=None):
    """
    pars are in g space

    Note likelihoods here are pure likelihood, no priors
    """
    from math import log
    from . import render
    import scipy.stats

    if nsub is not None:
        loglike,s2n,flags=render._render.loglike_subgrid(image, gmix, ivar,
                                                        int(nsub))
    else:
        loglike,s2n,flags=render._render.loglike(image, gmix, ivar)
    chi2=loglike/(-0.5)
    dof=image.size-npars
    chi2per = chi2/dof

    prob = scipy.stats.chisqprob(chi2, dof)

    aic = -2*loglike + 2*npars
    bic = -2*loglike + npars*log(image.size)

    return {'s2n_w':s2n,
            'loglike':loglike,
            'chi2per':chi2per,
            'dof':dof,
            'fit_prob':prob,
            'aic':aic,
            'bic':bic}
    #return s2n, loglike, chi2per, dof, prob


def calculate_some_stats_thresh(image, ivar, gmix, npars, nsigma, nsub=1):
    """
    pars are in g space

    Note likelihoods here are pure likelihood, no priors
    """
    from math import log
    from . import render
    import scipy.stats

    if nsub is None:
        nsub=1
    model=render.gmix2image(gmix, image.shape, nsub=nsub)

    skysig=sqrt(1./ivar)
    thresh=skysig*nsigma
    w=where(model > thresh)
    if w[0].size==0:
        raise ValueError("no values > %f*noise=%f" % (nsigma,thresh))

    npix=w[0].size

    loglike=-0.5*ivar*( ((model[w]-image[w])**2).sum() )

    sm=(model[w]*image[w]).sum()
    w2sum=(model[w]*model[w]).sum()
    s2n = sm/sqrt(w2sum)*sqrt(ivar);

    chi2=loglike/(-0.5)
    dof=npix-npars
    chi2per = chi2/dof

    prob = scipy.stats.chisqprob(chi2, dof)

    aic = -2*loglike + 2*npars
    bic = -2*loglike + npars*log(npix)

    return {'s2n_w':s2n,
            'loglike':loglike,
            'chi2per':chi2per,
            'dof':dof,
            'fit_prob':prob,
            'aic':aic,
            'bic':bic}


def _pars2gmix_coellip(pars):
    """
    Convert a parameter array.  

    [cen1,cen2,e1,e2,Tmax,Tfrac2,Tfrac3..,p1,p2,p3...]
    """
    ngauss = (len(pars)-4)/2
    gmix=[]

    row=pars[0]
    col=pars[1]
    e1 = pars[2]
    e2 = pars[3]
    Tmax = pars[4]

    for i in xrange(ngauss):
        d={}

        if i == 0:
            T = Tmax
        else:
            Tfrac = pars[4+i]
            T = Tmax*Tfrac

        p = pars[4+ngauss+i]
        
        d['p'] = p
        d['row'] = row
        d['col'] = col
        d['irr'] = (T/2.)*(1-e1)
        d['irc'] = (T/2.)*e2
        d['icc'] = (T/2.)*(1+e1)
        gmix.append(d)

    return gmix

def pars2full_coellip(pars):
    """
    Convert a coellip par array to a full
    pars array [p_1,cen1_1,cen2_1,irr_1,irc_1,icc_1,
                p_2,cen1_2,cen2_2,irr_2,irc_2,icc_2,
                ...]

    input is
    [cen1,cen2,e1,e2,Tmax,Tfrac2,Tfrac3..,p1,p2,p3...]
    """
    ngauss = (len(pars)-4)/2
    gmix=zeros(ngauss*6)

    row=pars[0]
    col=pars[1]
    e1 = pars[2]
    e2 = pars[3]
    Tmax = pars[4]

    for i in xrange(ngauss):
        beg=i*6

        if i == 0:
            T = Tmax
        else:
            Tfrac = pars[4+i]
            T = Tmax*Tfrac

        p = pars[4+ngauss+i]

        pars[beg+0] = p
        pars[beg+1] = row
        pars[beg+2] = col
        pars[beg+3] = (T/2.)*(1-e1)
        pars[beg+4] = (T/2.)*e2
        pars[beg+5] = (T/2.)*(1+e1)

    return gmix


def randomize_e1e2(e1start,e2start, width=0.1):
    if e1start == 0 or e1start is None or e2start==0 or e2start is None:
        e1rand = 0.05*srandu()
        e2rand = 0.05*srandu()
    else:
        e1rand = e1start*(1 + 2*width*srandu())
        e2rand = e2start*(1 + 2*width*srandu())
        etot = sqrt(e1rand**2 + e2rand**2)
        if etot > 0.95:
            e1rand,e2rand=randomize_e1e2(None,None)

    return e1rand, e2rand

def get_estyle_pars(pars):
    epars = pars.copy()

    T=pars[4]
    g1,g2=pars[2],pars[3]

    e1,e2,ok = g1g2_to_e1e2(g1,g2)
    if not ok:
        return None

    if T < 0:
        return None

    epars[2] = e1
    epars[3] = e2

    return epars

def g1g2_to_e1e2(g1, g2):
    """
    This version without exceptions

    returns e1,e2,okflag
    """
    g = math.sqrt(g1**2 + g2**2)
    if g >= 1.:
        return None,None,False

    if g == 0:
        return 0.,0.,True
    e = math.tanh(2*math.atanh(g))
    if e >= 1.:
        return None,None,False

    fac = e/g
    e1, e2 = fac*g1, fac*g2
    return e1,e2,True



def get_f_p_vals(fname=None, pars=None, Tfrac=True):
    """
    To check values from runs, send pars from e.g.
    exp file
        /gmix-fit-et10r99/outputs/gmix-fit-et10r99-001-000.rec
    dev file
        /gmix-fit-dt03r99/outputs/gmix-fit-dt03r99-001-000.rec
        ??
    """

    # make sure ordered
    if fname is not None:
        import esutil as eu
        t=eu.io.read(fname)
        pars=t['pars'].copy()
    elif pars is None:
        raise ValueError("send fname= or pars=")
    n=pars.shape[0]

    # smallest to largest
    T1s=numpy.zeros(n)
    T2s=numpy.zeros(n)
    T3s=numpy.zeros(n)
    p1s=numpy.zeros(n)
    p2s=numpy.zeros(n)
    p3s=numpy.zeros(n)

    for i in xrange(n):
        # these always fixed
        if Tfrac:
            Tmax=pars[i,4]
            T3s[i] = Tmax
            p3s[i] = pars[i,7]

            if pars[i,5] < pars[i,6]:
                T1s[i] = pars[i,5]*Tmax
                p1s[i] = pars[i,8]

                T2s[i] = pars[i,6]*Tmax
                p2s[i] = pars[i,9]
            else:
                T1s[i] = pars[i,6]*Tmax
                p1s[i] = pars[i,9]

                T2s[i] = pars[i,5]*Tmax
                p2s[i] = pars[i,8]
        else:
            s=pars[i,4:4+3].argsort()
            T1s[i]=pars[i,4+s[0]]
            T2s[i]=pars[i,4+s[1]]
            T3s[i]=pars[i,4+s[2]]

            p1s[i] = pars[i,7+s[0]]
            p2s[i] = pars[i,7+s[1]]
            p3s[i] = pars[i,7+s[2]]


    T1 = T1s.mean()
    T2 = T2s.mean()
    T3 = T3s.mean()
    p1 = p1s.mean()
    p2 = p2s.mean()
    p3 = p3s.mean()

    Tvals=numpy.array([T1,T2,T3])
    pvals=numpy.array([p1,p2,p3])
    psum=pvals.sum()
    pvals /= psum
    psum=pvals.sum()

    T = (Tvals*pvals).sum()/pvals.sum()
    Fvals=Tvals/T
    
    print 'T:    ',T
    print 'Tcalc:',(Fvals*T*pvals).sum()/psum
    print 'Fvals: [%.16g,%.16g,%.16g]' % tuple(Fvals)
    print 'pvals: [%.16g,%.16g,%.16g]' % tuple(pvals/psum)
    print 'sum(Fvals*pvals): %.16g' % ( (Fvals*pvals).sum(), )
    # these should equal
    print 'psum: %.16g' % psum
    print 'Fsum: %.16g' % Fvals.sum()

    Fpsum = (Fvals*pvals).sum()
    print 'Tvals: [%.16g,%.16g,%.16g]' % tuple(Tvals)
    print T/pvals[0]*(psum - Fpsum + pvals[0]*Fvals[0])
    print T/pvals[1]*(psum - Fpsum + pvals[1]*Fvals[1])
    print T/pvals[2]*(psum - Fpsum + pvals[2]*Fvals[2])

def get_f_p_vals_turb():
    import fimage
    from numpy.random import random as randu
    from .gmix_fit import GMixFitCoellip
    dims=[1000,1000]
    fwhm=100.
    im_nonoise=fimage.pixmodel.ogrid_turb_psf(dims, fwhm)
    im,skysig=fimage.noise.add_noise_admom(im_nonoise, 1.e8)
    #0.4422839982971848,0.9764735420431805,4.784430363572698
    #0.5356908850142901,0.3829828442516049,0.08132627073410502

    guess=array([100. + 0.01*(randu()-0.5),
                 100. + 0.01*(randu()-0.5),
                 0.01*(randu()-0.5),
                 0.01*(randu()-0.5),

                 fwhm + 0.01*(randu()-0.5),
                 0.2 + 0.01*(randu()-0.5),
                 0.08 + 0.01*(randu()-0.5),

                 0.08 + 0.01*(randu()-0.5),
                 0.4 + 0.01*(randu()-0.5),
                 0.54 + 0.01*(randu()-0.5)])

    width=guess.copy()
    width[:] = 100

    gm = GMixFitCoellip(im, skysig,
                        guess,width,
                        Tpositive=True)

    pars=gm.get_pars()
    perr=gm.get_perr()

    pars=pars.reshape(1,pars.size)
    get_f_p_vals(pars=pars)


def get_f_p_vals_exp(s2,show=False):
    """
    Determine high-resulution Fvals etc. for exp approximated
    by three gaussians
    """
    import fimage
    from numpy.random import random as randu
    from .gmix_fit import GMixFitCoellip
    import pprint

    Tpsf=1000
    Tobj=Tpsf/s2

    fvals=array(list(reversed([0.23,1.0,2.8])))
    
    fvals /= fvals.max()

    #pvals=array(list(reversed([0.06,0.56,0.37])))
    pvals=array(list(reversed([0.33,0.53,0.14])))
    

    objpars = {'model':'exp', 'cov':[Tobj/2.,0.0,Tobj/2.]}
    psfpars = {'model':'gauss', 'cov':[Tpsf/2.,0.0,Tpsf/2.]}
    ci0=fimage.convolved.ConvolverGaussFFT(objpars,psfpars)
    ci_nonoise = fimage.convolved.TrimmedConvolvedImage(ci0, fluxfrac=.9997)
    ci=fimage.convolved.NoisyConvolvedImage(ci_nonoise,1.e8, 1.e8,s2n_method='uw')

    psf_guess=array([ci['cen'][0] + 0.01*(randu()-0.5),
                     ci['cen'][1] + 0.01*(randu()-0.5),
                     0.01*(randu()-0.5),
                     0.01*(randu()-0.5),
                     Tpsf + 0.01*(randu()-0.5),
                     1.0 + 0.01*(randu()-0.5)])
    psf_width=psf_guess.copy()
    psf_width[:] = 100

    guess=array([ci['cen'][0] + 0.01*(randu()-0.5),
                 ci['cen'][1] + 0.01*(randu()-0.5),
                 0.01*(randu()-0.5),
                 0.01*(randu()-0.5),

                 Tobj + 0.01*(randu()-0.5),
                 fvals[1] + 0.01*(randu()-0.5),
                 fvals[2] + 0.01*(randu()-0.5),

                 pvals[0] + 0.01*(randu()-0.5),
                 pvals[1] + 0.01*(randu()-0.5),
                 pvals[2] + 0.01*(randu()-0.5)])

    width=guess.copy()
    width[0] = 0.01
    width[1] = 0.01
    width[5] = .001
    width[:] = 100

    gm_psf = GMixFitCoellip(ci.psf, ci['skysig_psf'],
                            psf_guess, psf_width,
                            Tpositive=True)
    psf_gmix=gm_psf.get_gmix()
    pprint.pprint(psf_gmix)
    gm = GMixFitCoellip(ci.image, ci['skysig'], guess, width,
                        psf=psf_gmix)

    pars=gm.get_pars()
    perr=gm.get_perr()
    gmix=gm.get_gmix()

    pars=pars.reshape(1,pars.size)
    get_f_p_vals(pars=pars)

    if show:
        import images
        from .render import gmix2image

        model=gmix2image(gmix, ci.image.shape, psf=psf_gmix)
        images.compare_images(ci.image, model)

     

def get_f_p_vals_dev(s2,show=False):
    """
    Determine high-resulution Fvals etc. for devaucouleur approximated
    by three gaussians
    """
    import fimage
    from numpy.random import random as randu
    from .gmix_fit import GMixFitCoellip
    import pprint

    Tpsf=1000
    Tobj=Tpsf/s2

    fvals=array(list(reversed([0.1, 1.86, 12.6])))
    #fvals=array(list(reversed([0.0, 0.9268795541243965, 9.627400726500005])))
    fvals /= fvals.max()

    pvals=array(list(reversed([0.77,0.18,0.046])))
    

    objpars = {'model':'dev', 'cov':[Tobj/2.,0.0,Tobj/2.]}
    psfpars = {'model':'gauss', 'cov':[Tpsf/2.,0.0,Tpsf/2.]}
    ci0=fimage.convolved.ConvolverGaussFFT(objpars,psfpars)
    ci_nonoise = fimage.convolved.TrimmedConvolvedImage(ci0, fluxfrac=.9997)
    ci=fimage.convolved.NoisyConvolvedImage(ci_nonoise,1.e8, 1.e8,s2n_method='uw')

    psf_guess=array([ci['cen'][0] + 0.01*(randu()-0.5),
                     ci['cen'][1] + 0.01*(randu()-0.5),
                     0.01*(randu()-0.5),
                     0.01*(randu()-0.5),
                     Tpsf + 0.01*(randu()-0.5),
                     1.0 + 0.01*(randu()-0.5)])
    psf_width=psf_guess.copy()
    psf_width[:] = 100

    guess=array([ci['cen'][0] + 0.01*(randu()-0.5),
                 ci['cen'][1] + 0.01*(randu()-0.5),
                 0.01*(randu()-0.5),
                 0.01*(randu()-0.5),

                 Tobj + 0.01*(randu()-0.5),
                 fvals[1] + 0.01*(randu()-0.5),
                 fvals[2] + 0.01*(randu()-0.5),

                 pvals[0] + 0.01*(randu()-0.5),
                 pvals[1] + 0.01*(randu()-0.5),
                 pvals[2] + 0.01*(randu()-0.5)])

    width=guess.copy()
    width[0] = 0.01
    width[1] = 0.01
    width[5] = .001
    width[:] = 100

    gm_psf = GMixFitCoellip(ci.psf, ci['skysig_psf'],
                            psf_guess, psf_width,
                            Tpositive=True)
    psf_gmix=gm_psf.get_gmix()
    pprint.pprint(psf_gmix)
    gm = GMixFitCoellip(ci.image, ci['skysig'], guess, width,
                        psf=psf_gmix)

    pars=gm.get_pars()
    perr=gm.get_perr()
    gmix=gm.get_gmix()

    pars=pars.reshape(1,pars.size)
    get_f_p_vals(pars=pars)

    if show:
        import images
        from .render import gmix2image

        model=gmix2image(gmix, ci.image.shape, psf=psf_gmix)
        images.compare_images(ci.image, model)

        

def compare_gmix_approx(type, s2):
    import fimage
    from .render import gmix2image
    from .gmix import GMixCoellip, GMixDev, GMixExp
    import images
    Tpsf=3.9
    Tobj=Tpsf/s2

    objpars = {'model':type, 'cov':[Tobj/2.,0.0,Tobj/2.]}
    psfpars = {'model':'gauss', 'cov':[Tpsf/2.,0.0,Tpsf/2.]}
    ci0=fimage.convolved.ConvolverGaussFFT(objpars,psfpars)
    ci = fimage.convolved.TrimmedConvolvedImage(ci0, fluxfrac=.9997)

    cen=ci['cen']
    psfcen=ci['cen_psf_uw']
    if type=='dev':
        gmix   = GMixDev(     [cen[0],cen[1], 0., 0., Tobj, 1.] )
    elif type=='exp':
        gmix   = GMixExp(     [cen[0],cen[1], 0., 0., Tobj, 1.] )
    else:
        raise ValueError("bad type: '%s'" % type)

    psf_gauss  = GMixCoellip( [psfcen[0],psfcen[1], 0., 0., Tpsf, 1.] )
    obj = gmix.convolve(psf_gauss)

    psf=gmix2image(psf_gauss,ci.image.shape,nsub=16)
    im=gmix2image(obj,ci.image.shape,nsub=16)

    im *= 1.0/im.sum()
    imdev = ci.image*1.0/ci.image.sum()

    images.compare_images(ci.psf, psf,
                          label1='psf0', label2='psf')

    images.compare_images(imdev, im,
                          label1='%s fft' % type, label2='%s gmix' % type)


def print_hogglang(nsersic, ngauss):
    if nsersic==4 and ngauss==6:
        Tvals=array([0.00263**2,
                     0.01202**2,
                     0.04031**2,
                     0.12128**2,
                     0.36229**2,
                     1.23604**2])
        pvals=array([0.01308,
                     0.12425,
                     0.63551,
                     2.22560,
                     5.63989,
                     9.81523])
    elif nsersic==4 and ngauss==10:
        Tvals=array([0.00087**2,
                     0.00296**2,
                     0.00792**2,
                     0.01902**2,
                     0.04289**2,
                     0.09351**2,
                     0.20168**2,
                     0.44126**2,
                     1.01833**2,
                     2.74555**2])
        pvals=array([0.00139,
                     0.00941,
                     0.04441,
                     0.16162,
                     0.48121,
                     1.20357,
                     2.54182,
                     4.46441,
                     6.22820,
                     6.15393])

    elif nsersic==1 and ngauss==4:
        Tvals=array([0.12068**2,
                     0.32730**2,
                     0.68542**2,
                     1.28089**2])

        pvals=array([0.09733,
                     1.12804,
                     4.99846,
                     5.63632])
    elif nsersic==1 and ngauss==6:
        Tvals=array([0.05072**2,
                     0.13756**2,
                     0.28781**2,
                     0.53195**2,
                     0.91209**2,
                     1.50157**2])
        pvals=array([0.00735,
                     0.09481,
                     0.63572,
                     2.60077,
                     5.42848,
                     3.16445])



    pvals /= pvals.sum()
    T = (Tvals*pvals).sum()
    Fvals = Tvals/T

    print 'Fvals:',tuple(Fvals)
    print 'pvals:',tuple(pvals)

    print 'pvals.sum():',pvals.sum()
    print 'Fvals.sum():',Fvals.sum()
    print '(Fvals*pvals).sum():',(Fvals*pvals).sum()

