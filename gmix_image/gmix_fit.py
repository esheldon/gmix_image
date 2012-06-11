import numpy
from numpy import zeros, array, where, ogrid, diag, sqrt, isfinite
from numpy.linalg import eig
from fimage import model_image
from sys import stderr

from gmix_image import gmix2image, total_moms_psf

GMIXFIT_MAXITER         = 2**0
GMIXFIT_SINGULAR_MATRIX = 2**4
GMIXFIT_NEG_COV_EIG     = 2**5
GMIXFIT_NEG_COV_DIAG    = 2**6
GMIXFIT_NEG_MCOV_DIAG   = 2**7 # the M sub-cov matrix not positive definite
GMIXFIT_MCOV_NOTPOSDEF  = 2**8 # more strict checks on cholesky decomposition
GMIXFIT_CALLS_NOT_CHANGING   = 2**9 # see fmin_cg
GMIXFIT_LOW_S2N = 2**8 # very low S/N for ixx+iyy


class GMixFitCoellip:
    """
    Use levenberg marquardt to fit a gaussian mixture model.  
    
    The gaussians are forced to be co-elliptical.  Image is assumed to be
    sky-subtracted

    When we get a mask image, we can use that too within ydiff

    Priors?
    """
    def __init__(self, image, guess, 
                 error=None, psf=None, method='lm', verbose=False):
        self.image=image
        self.error=error # per pixel
        self.guess=guess
        self.ngauss=(len(guess)-4)/2
        self.nsub=1
        self.verbose=verbose

        # can enter the psf model as a mixture model or
        # a coelliptical psf model; gmix is more flexible..
        # we keep the gmix version since we use gmix2image to
        # make models
        if isinstance(psf,numpy.ndarray):
            self.psf = pars2gmix_coellip(psf)
        else:
            self.psf = psf

        self.row,self.col=ogrid[0:image.shape[0], 0:image.shape[1]]

        if method=='lm':
            self.dofit_lm()
        elif method=='cg':
            self.dofit_cg()
        elif method=='ncg':
            self.dofit_ncg()
        else:
            raise ValueError("expected method 'lm' or 'cg'")

    def dofit_lm(self):
        """
        Run the fit using LM
        """
        from scipy.optimize import leastsq
        res = leastsq(self.ydiff,self.guess,
                      full_output=1,
                      Dfun=self.jacob,
                      col_deriv=1)
        self.popt, self.pcov0, self.infodict, self.errmsg, self.ier = res
        if self.ier == 0:
            # wrong args, this is a bug
            raise ValueError(self.errmsg)

        self.numiter = self.infodict['nfev']

        self.pcov=None
        self.perr=None


        if self.pcov0 is not None:
            self.pcov = self.scale_leastsq_cov(self.popt, self.pcov0)

            d=diag(self.pcov)
            w,=where(d <= 0)

            if w.size == 0:
                # only do if non negative
                self.perr = sqrt(d)

        self.set_lm_flags()

        if self.flags == 0:
            cov_inv = numpy.linalg.inv(self.pcov)
            mcov_inv = cov_inv[2:2+3, 2:2+3]
            self.mcov_fix = numpy.linalg.inv(mcov_inv)
            

    def dofit_cg(self):
        """
        Run the fit using Conjugate Gradient
        # TODO implement calculation of covariance from jacobian
        """
        from scipy.optimize import fmin_cg
        res = fmin_cg(self.ydiff_sq_sum,
                      self.guess,
                      fprime=self.ydiff_sq_jacob,
                      full_output=1)
        (self.popt, self.fopt, self.numiter, 
            self.grad_calls, self.warnflags) = res

        self.pcov=None
        self.perr=None

        self.set_cg_flags()

    def dofit_ncg(self):
        """
        Run the fit using Newton Conjugate Gradient 
        # TODO implement calculation of covariance from jacobian
        """
        from scipy.optimize import fmin_ncg
        res = fmin_ncg(self.ydiff_sq_sum,
                       self.guess,
                       fprime=self.ydiff_sq_jacob,
                       full_output=1)
        (self.popt, self.fopt, self.numiter, 
            self.grad_calls, self.warnflags) = res

        self.pcov=None
        self.perr=None

        self.set_cg_flags()
       
    def set_lm_flags(self):
        flags = 0
        if self.ier > 4:
            flags = 2**(self.ier-5)
            if self.verbose:
                print >>stderr,self.errmsg 

        if self.pcov is None:
            if self.verbose:
                print >>stderr,'singular matrix'
            flags += GMIXFIT_SINGULAR_MATRIX 
        else:
            e,v = eig(self.pcov)
            weig,=where(e <= 0)
            if weig.size > 0:
                if self.verbose:
                    import images
                    print >>stderr,'negative covariance eigenvalues'
                    images.imprint(self.pcov,stream=stderr)
                flags += GMIXFIT_NEG_COV_EIG 

            wneg,=where(diag(self.pcov) <= 0)
            if wneg.size > 0:
                if self.verbose:
                    import images
                    # only print if we didn't see negative eigenvalue
                    print >>stderr,'negative covariance diagonals'
                    images.imprint(self.pcov,stream=stderr)
                flags += GMIXFIT_NEG_COV_DIAG 

            mcov = self.pcov[2:2+3,2:2+3].copy()
            me,v = eig(mcov)
            weig,=where(me <= 0)
            if weig.size > 0:
                if self.verbose:
                    import images
                    print >>stderr,'negative M covariance eigenvalues'
                    print >>stderr,me
                    images.imprint(mcov,stream=stderr)
                flags += GMIXFIT_NEG_MCOV_DIAG

            # this has more strict checks it seems
            try:
                MM = numpy.linalg.cholesky(mcov)
            except numpy.linalg.LinAlgError as e:
                if self.verbose:
                    import images
                    print >>stderr,e
                    images.imprint(mcov,stream=stderr)
                flags += GMIXFIT_MCOV_NOTPOSDEF

            if ( ((flags & GMIXFIT_MCOV_NOTPOSDEF) != 0)
                    and ((flags & GMIXFIT_NEG_MCOV_DIAG) == 0) ):
                import images
                print >>stderr,'found one:'
                print >>stderr,me
                images.imprint(mcov,stream=stderr,fmt='%.16e')
                print >>stderr,mcov.dtype.descr
                print >>stderr,mcov.flags
                print >>stderr,mcov.shape
                stop


            '''
            if flags == 0 and True:
                #print 's2n on cen0:',self.popt[0]/sqrt(self.pcov[0,0])
                #print 's2n on popt:'
                #for i in xrange(len(self.popt)):
                #    print self.popt[i]/sqrt(self.pcov[i,i])
                T = self.popt[2]+self.popt[4]
                Terr = sqrt(self.pcov[2,2]+self.pcov[4,4]+2*self.pcov[2,4])
                Ts2n = T/Terr
                if T/Terr < 1.0:
                    if self.verbose:
                        print >>stderr,'S/N on T < 1: ',Ts2n
                    flags += GMIXFIT_LOW_S2N 
            '''
        self.flags = flags

    def set_cg_flags(self):
        self.flags = 0
        if self.warnflags == 1:
            if self.verbose:
                print >>stderr,"maxiter reached"
            self.flags += GMIXFIT_MAXITER
        if self.warnflags == 1:
            if self.verbose:
                print >>stderr,"func/fprime calls not changing"
            self.flags += GMIXFIT_CALLS_NOT_CHANGING

    def get_gmix(self):
        return pars2gmix_coellip(self.popt)
    gmix = property(get_gmix)

    def ydiff(self, pars):
        """
        Also apply hard priors on centroid range and determinant(s)
        """

        if not self.check_hard_priors(pars):
            return zeros(self.image.size) + numpy.inf

        model = self.make_model(pars)
        if not isfinite(model[0,0]):
            #raise ValueError("NaN in model")
            if self.verbose:
                print >>stderr,'found NaN in model'
            return zeros(self.image.size) + numpy.inf
        return (model-self.image).reshape(self.image.size)
    
    def ydiff_sq_sum(self, pars):
        """
        Using conjugate gradient algorithm, we minimize
        (y-data)**2
        """
        yd=self.ydiff(pars)
        return (yd**2).sum()

    def ydiff_sq_jacob(self, pars):
        """
        Using conjugate gradient algorithm, we minimize
            (model-data)**2 = ydiff**2
        so the jacobian is 2*(ydiff*jacob).sum()
        """
        # this is a list with each element the full array over pixels
        # both jacob and ydiff are reshaped to linear for use in the 
        # LM algorithm, but this doesn't matter
        model_j = self.jacob(pars)
        ydiff = self.ydiff(pars)

        j = zeros(len(pars))
        for i in xrange(len(model_j)):
            j[i] = 2*( ydiff*model_j[i] ).sum()
        return j

    def chi2(self, pars):
        ydiff = self.ydiff(pars)
        return (ydiff**2).sum()
    def chi2per(self, pars, skysig):
        ydiff = self.ydiff(pars)
        return (ydiff**2).sum()/(self.image.size-len(pars))/skysig**2

    def check_hard_priors(self, pars):
        # make sure p and f values are > 0
        vals = pars[5:]
        w,=where(vals <= 0)
        if w.size > 0:
            return False

        # make sure f values are reasonable, at least for exp, need
        # to try dev
        """
        if self.ngauss > 1:
            fvals = pars[5+self.ngauss:]
            w,=where((fvals < 0.05) | (fvals > 15.0))
            if w.size > 0:
                return False
        """

        # check determinant for all images we might have
        # to create with psf convolutions
        if self.psf is not None:
            gmix=pars2gmix_coellip(pars)
            moms = total_moms_psf(gmix, self.psf)
            pdet = moms['irr']*moms['icc']-moms['irc']**2
            if pdet <= 0:
                return False
            for g in gmix:
                for p in self.psf:
                    irr = g['irr'] + p['irr']
                    irc = g['irc'] + p['irc']
                    icc = g['icc'] + p['icc']
                    det = irr*icc-irc**2
                    if det <= 0:
                        return False

        # overall determinant
        det = pars[2]*pars[4]-pars[3]**2
        if (det <= 0 
                or pars[0] < 0 or pars[0] > (self.image.shape[0]-1)
                or pars[1] < 0 or pars[1] > (self.image.shape[1]-1)):
            return False

        return True

    def make_model(self, pars, aslist=False):
        """
        pars = [row,col,irr,irc,icc,pi...,fi...]
        """
        #fmt = ' '.join(['%10.6f']*len(pars))
        #print 'pars:' +fmt % tuple(pars)
        gmix = pars2gmix_coellip(pars)
        return gmix2image(gmix, 
                          self.image.shape, 
                          psf=self.psf,
                          aslist=aslist, 
                          order='f', 
                          nsub=self.nsub,
                          renorm=False)

    def jacob(self, pars):
        """
        Calculate the jacobian of the function for each parameter
        """
        if self.psf is not None:
            return self.jacob_psf(pars)

        # [r0, c0, Irr, Irc, Icc, pi, fi]
        det = pars[2]*pars[4]-pars[3]**2
        y = self.row-pars[0]
        x = self.col-pars[1]

        #f = self.make_model(pars)
        flist = self.make_model(pars, aslist=True)

        Myy = pars[2]
        Mxy = pars[3]
        Mxx = pars[4]

        Myy_x = Myy*x
        Mxy_y = Mxy*y
        Mxy_x = Mxy*x
        Mxx_y = Mxx*y

        jacob = []

        #
        # for cen,cov we sum up contributions from each gauss
        #

        # y0
        jtmp = zeros(self.image.shape)
        fac = (Mxx_y - Mxy_x)/det
        for i,Fi in enumerate(flist):
            fi = 1 if i == 0 else pars[5+self.ngauss+i-1]
            jtmp += Fi*fac/fi
        jacob.append(jtmp)

        # x0
        jtmp = zeros(self.image.shape)
        fac = (Myy_x - Mxy_y)/det
        for i,Fi in enumerate(flist):
            fi = 1 if i == 0 else pars[5+self.ngauss+i-1]
            jtmp += Fi*fac/fi
        jacob.append(jtmp)

        # Myy (a)
        #fac = (-Mxx/det  + 0.5*((Mxx_y-Mxy_x)/det)**2)
        jtmp = zeros(self.image.shape)
        for i,Fi in enumerate(flist):
            fi = 1 if i == 0 else pars[5+self.ngauss+i-1]
            fac = (-0.5*Mxx/det  + 0.5*((Mxx_y-Mxy_x)/det)**2/fi)
            jtmp += Fi*fac
        jacob.append(jtmp)

        # Mxy
        #fac = (2*Mxy/det + (Mxx_y-Mxy_x)*(Myy_x-Mxy_y)/det**2)
        jtmp = zeros(self.image.shape)
        for i,Fi in enumerate(flist):
            fi = 1 if i == 0 else pars[5+self.ngauss+i-1]
            fac = (Mxy/det + (Mxx_y-Mxy_x)*(Myy_x-Mxy_y)/det**2/fi)
            jtmp += Fi*fac
        jacob.append(jtmp)

        # Mxx
        #fac = (-Myy/det  + 0.5*( (Myy_x - Mxy_y)/det)**2 )
        jtmp = zeros(self.image.shape)
        for i,Fi in enumerate(flist):
            fi = 1 if i == 0 else pars[5+self.ngauss+i-1]
            fac = (-0.5*Myy/det  + 0.5*( (Myy_x - Mxy_y)/det)**2/fi )
            jtmp += Fi*fac
        jacob.append(jtmp)

        # pi
        # have an entry for *each* gauss rather than summed
        for i,Fi in enumerate(flist):
            pi = pars[5+i]
            jtmp = Fi/pi
            jacob.append(jtmp)

        # fi
        # an entry for each gauss after the first
        if self.ngauss > 1:
            fac = 0.5*(Myy*x**2 - 2.*Mxy*x*y + Mxx*y**2)/det
            for i in xrange(1,self.ngauss):
                Fi = flist[i]
                fi = pars[5+self.ngauss+i-1]
                jtmp = Fi*(fac-fi)/fi**2
                jacob.append(jtmp)

        for i in xrange(len(jacob)):
            jacob[i] = jacob[i].reshape(self.image.size)
        return jacob

    def jacob_psf(self, pars):
        """
        Calculate the jacobian for each parameter
        """
        #import images
        # [r0, c0, Irr, Irc, Icc, pi, fi]
        y = self.row-pars[0]
        x = self.col-pars[1]

        #f = self.make_model(pars)
        flist = self.make_model(pars, aslist=True)

        Myy = pars[2]
        Mxy = pars[3]
        Mxx = pars[4]

        jacob = []

        #
        # for cen,cov we sum up contributions from each gauss
        #

        # y0
        #fac = (Mxx_y - Mxy_x)/det
        # x0
        #fac = (Myy_x - Mxy_y)/det

        jtmp_y = zeros(self.image.shape)
        jtmp_x = zeros(self.image.shape)
        for i,plist in enumerate(flist):
            fi = 1 if i == 0 else pars[5+self.ngauss+i-1]
            for j in xrange(len(plist)):
                p = self.psf[j]
                pim = plist[j]
                
                tMyy = Myy*fi + p['irr']
                tMxy = Mxy*fi + p['irc']
                tMxx = Mxx*fi + p['icc']

                det = (tMyy*tMxx - tMxy**2)
                fac_y = (tMxx*y - tMxy*x)/det
                fac_x = (tMyy*x - tMxy*y)/det

                jtmp_y += pim*fac_y
                jtmp_x += pim*fac_x
        jacob.append(jtmp_y)
        jacob.append(jtmp_x)

        # Myy
        # Mxy
        # Mxx
        # fi
        jtmp_yy = zeros(self.image.shape)
        jtmp_xy = zeros(self.image.shape)
        jtmp_xx = zeros(self.image.shape)

        jf_list=[] # these will go at the end of the jacobian list
        for i,plist in enumerate(flist):
            fi = 1 if i == 0 else pars[5+self.ngauss+i-1]

            if i > 0:
                jtmp = zeros(self.image.shape)
            for j in xrange(len(plist)):
                p = self.psf[j]
                pim = plist[j]

                tMyy = Myy*fi + p['irr']
                tMxy = Mxy*fi + p['irc']
                tMxx = Mxx*fi + p['icc']
                #print j,fi,p['irr'],p['irc'],p['icc']
                #print 'M:',Myy,Mxy,Mxx
                #print 'tM:',tMyy,tMxy,tMxx

                det = (tMyy*tMxx - tMxy**2)
                
                tMxx_y = tMxx*y
                tMxy_x = tMxy*x
                tMxy_y = tMxy*y
                tMyy_x = tMyy*x

                yy_sum = .5*(-tMxx/det  + ((tMxx_y-tMxy_x)/det)**2)
                xx_sum = .5*(-tMyy/det  + ((tMyy_x-tMxy_y)/det)**2 )
                xy_sum = (tMxy/det + (tMxx_y-tMxy_x)*(tMyy_x-tMxy_y)/det**2)

                jtmp_yy += fi*pim*yy_sum
                jtmp_xy += fi*pim*xy_sum
                jtmp_xx += fi*pim*xx_sum

                if i > 0:
                    jtmp += pim*(Myy*yy_sum + Mxy*xy_sum + Mxx*xx_sum)
            if i > 0:
                jf_list.append(jtmp)

        jacob.append(jtmp_yy)
        jacob.append(jtmp_xy)
        jacob.append(jtmp_xx)

        # pi
        # have an entry for *each* gauss rather than summed

        for i,plist in enumerate(flist):
            gp = pars[5+i]
            jtmp = zeros(self.image.shape)
            for j in xrange(len(plist)):
                pim = plist[j]
                jtmp += pim
            jtmp /= gp
            jacob.append(jtmp)

        # f derivatives go at the end
        jacob += jf_list

        for i in xrange(len(jacob)):
            jacob[i] = jacob[i].reshape(self.image.size)
        return jacob


    def scale_leastsq_cov(self, popt, pcov):
        """
        Scale the covariance matrix returned from leastsq; this will
        recover the covariance of the parameters in the right units.
        """
        dof = (self.image.size-len(popt))
        s_sq = (self.ydiff(popt)**2).sum()/dof
        return pcov * s_sq 


def pars2gmix_coellip(pars):
    """
    Convert a parameter array as used for the LM code into a gaussian mixture
    model.  This is for the case of co-elliptical gaussians.
    """
    ngauss = (len(pars)-4)/2
    gmix=[]
    gmix.append({'row':pars[0],
                 'col':pars[1],
                 'irr':pars[2],
                 'irc':pars[3],
                 'icc':pars[4],
                 'p':pars[5]})

    if ngauss > 1:
        for i in xrange(1,ngauss):
            p = pars[5+i]
            f = pars[5+ngauss+i-1]

            gmix.append({'row':pars[0],
                         'col':pars[1],
                         'irr':pars[2]*f,
                         'irc':pars[3]*f,
                         'icc':pars[4]*f,
                         'p':p})

    return gmix

def gmix2pars_coellip(gmix):
    ngauss=len(gmix)
    if ngauss > 2:
        raise ValueError("ngauss <= 2")
    pars=zeros(2*ngauss+4)

    imin=-1
    Tmin=9999
    for i,g in enumerate(gmix):
        T = g['irr']+g['icc']
        if T < Tmin:
            Tmin=T
            imin=i

    pars[0] = gmix[0]['row']
    pars[1] = gmix[0]['col']
    pars[2] = gmix[imin]['irr']
    pars[3] = gmix[imin]['irc']
    pars[4] = gmix[imin]['icc']
    pars[5] = gmix[imin]['p']

    if ngauss > 1:
        for i in xrange(ngauss):
            if i != imin:
                pars[6] = gmix[i]['p']
                T = gmix[i]['irr']+gmix[i]['icc']
                pars[7] = T/Tmin
                break
    return pars


def print_pars(stream, pars, front=None):
    fmt = ' '.join( ['%10.6g ']*len(pars) )
    if front is not None:
        stream.write(front)
        stream.write(' ')
    stream.write(fmt % tuple(pars))
    stream.write('\n')
