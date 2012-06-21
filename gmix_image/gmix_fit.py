import numpy
from numpy import zeros, array, where, ogrid, diag, sqrt, isfinite, \
        tanh, arctanh, cos, sin, exp
from numpy.linalg import eig
from fimage import model_image
from sys import stderr

from .gmix_em import gmix2image, total_moms_psf

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
    Perform a non-linear fit of the image to a gaussian mixture model.  
    
    The gaussians are forced to be co-elliptical.  Image is assumed to be
    sky-subtracted

    parameters
    ----------
    image: numpy array, dim=2
        A background-subtracted image.
    guess:
        A starting guess for the parameters.  If ptype=='ellip'
        then this array is
            [cen0,cen1,eta,theta,pi,Ti]
        Where i runs over all gaussians. Eta is defined by
            ellip=(1+tanh(eta))/2.
        Which allows the bounds of eta to b -inf,inf.

        Note ngauss=(len(guess)-4)/2
        
        If ptype=='cov' then the parameters are
            [cen0,cen1,cov00,cov01,cov11,pi,fi]

    ptype: string
        Either 'ellip' or 'cov'
    error: float
        The error per pixel.  Note currently used.
    psf: list of dictionaries
        A gaussian mixture model specified as a list of
        dictionaries.  Each dict has these entries
            p: A normalization
            row: center of the gaussian in the first dimension
            col: center of the gaussian in the second dimension
            irr: Covariance matrix element for row*row
            irc: Covariance matrix element for row*col
            icc: Covariance matrix element for col*col

    method: string
        method for the fit.  Only 'lm' is useful currently.
    verbose: bool
    """
    def __init__(self, image, guess, 
                 ptype='cov',
                 error=None, 
                 psf=None, 
                 method='lm', 
                 Tmin=0.0,
                 use_jacob=True,
                 verbose=False):
        self.image=image
        self.error=error # per pixel
        self.guess=guess
        self.ptype=ptype
        self.ngauss=(len(guess)-4)/2
        self.nsub=1
        self.use_jacob=use_jacob

        self.Tmin=Tmin
        self.verbose=verbose

        # can enter the psf model as a mixture model or
        # a coelliptical psf model; gmix is more flexible..
        # we keep the gmix version since we use gmix2image to
        # make models
        if isinstance(psf,numpy.ndarray):
            self.psf = pars2gmix_coellip(psf,ptype=self.ptype)
        else:
            self.psf = psf

        self.row,self.col=ogrid[0:image.shape[0], 0:image.shape[1]]

        if method=='lm':
            self.dofit_lm()
        else:
            raise ValueError("expected method 'lm'")

    def dofit_lm(self):
        """
        Run the fit using LM
        """
        from scipy.optimize import leastsq

        if self.use_jacob:
            res = leastsq(self.ydiff,
                          self.guess,
                          full_output=1,
                          Dfun=self.jacob,
                          col_deriv=1)
        else:
            res = leastsq(self.ydiff,
                          self.guess,
                          full_output=1)

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
            w,=where(d < 0)

            if w.size == 0:
                # only do if non negative
                self.perr = sqrt(d)

        self.set_lm_flags()

        if self.flags == 0:
            cov_inv = numpy.linalg.inv(self.pcov)
            mcov_inv = cov_inv[2:2+3, 2:2+3]
            self.mcov_fix = numpy.linalg.inv(mcov_inv)
            

       
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
            #import images
            #print >>stderr,'pcov:'
            #images.imprint(self.pcov,stream=stderr)
            e,v = eig(self.pcov)
            weig,=where(e < 0)
            if weig.size > 0:
                if self.verbose:
                    import images
                    print >>stderr,'negative covariance eigenvalues'
                    print_pars(self.popt, front='popt: ')
                    print_pars(e,         front='eig:  ')
                    images.imprint(self.pcov,stream=stderr)
                flags += GMIXFIT_NEG_COV_EIG 

            wneg,=where(diag(self.pcov) < 0)
            if wneg.size > 0:
                if self.verbose:
                    import images
                    # only print if we didn't see negative eigenvalue
                    print >>stderr,'negative covariance diagonals'
                    #images.imprint(self.pcov,stream=stderr)
                flags += GMIXFIT_NEG_COV_DIAG 

            if self.ptype == 'cov':
                mcov = self.pcov[2:2+3,2:2+3].copy()
                me,v = eig(mcov)
                weig,=where(me < 0)
                if weig.size > 0:
                    if self.verbose:
                        import images
                        print >>stderr,'negative M covariance eigenvalues'
                        print >>stderr,me
                        #images.imprint(mcov,stream=stderr)
                    flags += GMIXFIT_NEG_MCOV_DIAG

                # this has more strict checks it seems
                try:
                    MM = numpy.linalg.cholesky(mcov)
                except numpy.linalg.LinAlgError as e:
                    if self.verbose:
                        import images
                        print >>stderr,e
                        #images.imprint(mcov,stream=stderr)
                    flags += GMIXFIT_MCOV_NOTPOSDEF

                if ( ((flags & GMIXFIT_MCOV_NOTPOSDEF) != 0)
                        and ((flags & GMIXFIT_NEG_MCOV_DIAG) == 0) ):
                    import images
                    print >>stderr,'found one:'
                    print >>stderr,me
                    #images.imprint(mcov,stream=stderr,fmt='%.16e')
                    #print >>stderr,mcov.dtype.descr
                    #print >>stderr,mcov.flags
                    #print >>stderr,mcov.shape
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

    def get_gmix(self):
        return pars2gmix_coellip(self.popt, ptype=self.ptype)
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
        if self.ptype == 'cov':
            # make sure p and f values are > 0
            vals = pars[5:]
        elif self.ptype=='eta':

            eta = pars[2]
            ellip = eta2ellip(eta)
            if ellip == 0 or ellip > 1:
                return False

            vals=pars[4:]
        elif self.ptype == 'e1e2':
            e1=pars[2]
            e2=pars[3]
            e = sqrt(e1**2 + e2**2)
            if (abs(e1) >= 1) or (abs(e2) >= 1) or (e >= 1):
                if self.verbose:
                    print >>stderr,'ellip >= 1'
                return False

            vals=pars[4:]

            tvals = pars[4+self.ngauss:]
            w,=where(tvals < self.Tmin)
            if w.size > 0:
                if self.verbose:
                    print >>stderr,'Found T < Tmin:',tvals[w[0]],self.Tmin
                    print vals
                return False

        else:
            raise ValueError("bad ptype: %s" % self.ptype)

        w,=where(vals <= 0)
        if w.size > 0:
            if self.verbose:
                print >>stderr,'bad p/T'
                print vals
            return False

        # check determinant for all images we might have
        # to create with psf convolutions

        gmix=pars2gmix_coellip(pars, ptype=self.ptype)
        if self.psf is not None:
            moms = total_moms_psf(gmix, self.psf)
            pdet = moms['irr']*moms['icc']-moms['irc']**2
            if pdet <= 0:
                if self.verbose:
                    print >>stderr,'bad p det'
                return False
            for g in gmix:
                for p in self.psf:
                    irr = g['irr'] + p['irr']
                    irc = g['irc'] + p['irc']
                    icc = g['icc'] + p['icc']
                    det = irr*icc-irc**2
                    if det <= 0:
                        if self.verbose:
                            print >>stderr,'bad p+obj det'
                        return False

        # overall determinant and centroid
        g0 = gmix[0]
        det = g0['irr']*g0['icc'] - g0['irc']**2
        if (det <= 0 
                or pars[0] < 0 or pars[0] > (self.image.shape[0]-1)
                or pars[1] < 0 or pars[1] > (self.image.shape[1]-1)):
            if self.verbose:
                print >>stderr,'bad det or centroid'
            return False

        return True

    def make_model(self, pars, aslist=False):
        gmix = pars2gmix_coellip(pars, ptype=self.ptype)
        return gmix2image(gmix, 
                          self.image.shape, 
                          psf=self.psf,
                          aslist=aslist, 
                          nsub=self.nsub,
                          renorm=False)

    def jacob(self, pars):
        if self.ptype == 'eta':
            return self.jacob_eta(pars)
        elif self.ptype == 'cov':
            return self.jacob_cov(pars)
        elif self.ptype == 'e1e2':
            return self.jacob_e1e2(pars)
        else:
            raise ValueError("ptype must be 'e1e2','cov','eta'")

    def jacob_e1e2(self, pars):
        """
        Calculate the jacobian of the function for each parameter
        using the eta parametrization
        """
        if self.psf is not None:
            return self.jacob_e1e2_psf(pars)

        ngauss=self.ngauss
        y = self.row-pars[0]
        x = self.col-pars[1]
        flist = self.make_model(pars, aslist=True)

        e1    = pars[2]
        e2    = pars[3]
        ellip = sqrt(e1**2 + e2**2)

        x2my2 = x**2 - y**2
        xy2 = 2*x*y
        x2=x**2
        y2=y**2

        jacob = []

        jy0 = zeros(self.image.shape)
        jx0 = zeros(self.image.shape)
        je1 = zeros(self.image.shape)
        je2 = zeros(self.image.shape)

        jp_list = []
        jT_list = []

        # 
        # for cen we sum up contributions from each gauss
        # 

        for i,Fi in enumerate(flist):
            gp = pars[4+i]
            T = pars[4+ngauss+i]

            # centroid
            y0fac = y*(1.+e1) - x*e2
            x0fac = x*(1.-e1) - y*e2

            y0fac *= 2./T/(1-ellip**2)
            x0fac *= 2./T/(1-ellip**2)

            jy0 += Fi*y0fac
            jx0 += Fi*x0fac

            #
            # e1
            #
            e1fac1 = e1/(1-ellip**2)
            e1fac2 = x2my2*(1-ellip**2 + 2*e1**2)
            e1fac2 *= 1./T/(1.-ellip**2)**2

            je1 += Fi*(e1fac1+e1fac2)

            #
            # e2
            #

            e2fac1 = e2/(1-ellip**2)
            e2fac2 = xy2*(1-ellip**2 + 2*e2**2)
            e2fac2 *= 1./T/(1.-ellip**2)**2

            je2 += Fi*(e2fac1+e2fac2)

            #
            # p
            #
            jp = Fi/gp
            jp_list.append(jp)

            #
            # T
            #
            Tfac1 = -1./T
            arg = (x2*(1.-e1) - xy2*e2 + y2*(1.+e1))/T/(1.-ellip**2)
            Tfac2 = arg/T
            jT = Fi*(Tfac1+Tfac2)
            jT_list.append(jT)

        jacob.append(jy0)
        jacob.append(jx0)
        jacob.append(je1)
        jacob.append(je2)
        jacob += jp_list
        jacob += jT_list


        for i in xrange(len(jacob)):
            #print >>stderr,i
            #print >>stderr,jacob[i]
            jacob[i] = jacob[i].reshape(self.image.size)

        return jacob

    def jacob_e1e2_psf(self, pars):

        flist = self.make_model(pars, aslist=True)

        ngauss=self.ngauss
        y = self.row-pars[0]
        x = self.col-pars[1]

        e1    = pars[2]
        e2    = pars[3]

        x2=x**2
        y2=y**2

        x2my2 = x2 - y2
        r2 = x2 + y2
        xy2 = 2*x*y

        jacob = []

        jy0 = zeros(self.image.shape)
        jx0 = zeros(self.image.shape)
        je1 = zeros(self.image.shape)
        je2 = zeros(self.image.shape)

        jp_list = []
        jT_list = []
        for i,plist in enumerate(flist):
            T = pars[4+ngauss+i]
            gp = pars[4+i]

            # we have one of these for each gaussian
            jp = zeros(self.image.shape)
            jT = zeros(self.image.shape)
            for j in xrange(len(plist)):
                p = self.psf[j]
                pim = plist[j] # convolved image

                Tpsf = p['irr']+p['icc']
                To = T + Tpsf

                e1psf = (p['icc']-p['irr'])/Tpsf
                e2psf = 2*p['irc']/Tpsf

                R = T/To
                s2 = Tpsf/T

                e1o = R*(e1 + e1psf*s2)
                e2o = R*(e2 + e2psf*s2)
                ellipo_2 = e1o**2 + e2o**2

                #
                # centroid
                #
                y0fac = y*(1.+e1o) - x*e2o
                y0fac *= 2./To/(1-ellipo_2)

                x0fac = x*(1.-e1o) - y*e2o
                x0fac *= 2./To/(1-ellipo_2)

                jy0 += pim*y0fac
                jx0 += pim*x0fac

                #
                # e1
                #

                e1fac1 = e1o/(1-ellipo_2)
                e1fac2 = x2my2*(1-ellipo_2 + 2*e1o**2)
                e1fac2 *= 1./To/(1.-ellipo_2)**2

                e1fac = R*(e1fac1 + e1fac2)
                je1 += pim*e1fac

                #
                # e2
                #

                e2fac1 = e2o/(1-ellipo_2)
                e2fac2 = xy2*(1-ellipo_2 + 2*e2o**2)
                e2fac2 *= 1./To/(1.-ellipo_2)**2
                
                e2fac = R*(e2fac1 + e2fac2)
                je2 += pim*e2fac


                #
                # counts
                #
                jp += pim

                #
                # T
                #
                Tfac1 = -1./To
                arg = (x2*(1.-e1o) - xy2*e2o + y2*(1.+e1o))/To/(1.-ellipo_2)
                Tfac2 = arg/To
                jT += pim*(Tfac1+Tfac2)


            jp /= gp
            jp_list.append(jp)
            jT_list.append(jT)


        jacob.append(jy0)
        jacob.append(jx0)
        jacob.append(je1)
        jacob.append(je2)
        jacob += jp_list
        jacob += jT_list

        for i in xrange(len(jacob)):
            jacob[i] = jacob[i].reshape(self.image.size)
        return jacob



    def jacob_eta(self, pars):
        """
        Calculate the jacobian of the function for each parameter
        using the eta parametrization
        """
        if self.psf is not None:
            return self.jacob_eta_psf(pars)

        ngauss=self.ngauss
        y = self.row-pars[0]
        x = self.col-pars[1]
        flist = self.make_model(pars, aslist=True)

        eta   = pars[2]
        de_deta = 0.5*(2./(exp(eta)+exp(-eta)))**2

        ellip = eta2ellip(eta)
        theta = pars[3]
        cos2theta = cos(2*theta)
        sin2theta = sin(2*theta)
        e1    = ellip*cos2theta
        e2    = ellip*sin2theta
        overe2 = 1./(1.-ellip**2)

        x2my2 = x**2 - y**2
        xy2 = 2*x*y
        x2=x**2
        y2=y**2

        jacob = []

        # 
        # for cen we sum up contributions from each gauss
        # 

        # y0,x0
        jy0 = zeros(self.image.shape)
        jx0 = zeros(self.image.shape)
        for i,Fi in enumerate(flist):
            T = pars[4+ngauss+i]
            yfac = y*(1.+e1) - x*e2
            xfac = x*(1.-e1) - y*e2

            yfac *= 2./T/(1-ellip**2)
            xfac *= 2./T/(1-ellip**2)

            jy0 += Fi*yfac
            jx0 += Fi*xfac

        jacob.append(jy0)
        jacob.append(jx0)

        # eta
        
        # just the chain rule
        # we calculate dF/de*de/d(eta)  and use
        #   ellip=(1+tanh(eta))/2
        # this is the derivative of ellip with respect
        # to eta, using d(tanh)/d(eta) = sech^2(eta)
        # = ( 2/(e^(\eta) + e^(-\eta)) )^2

        jtmp = zeros(self.image.shape)
        for i,Fi in enumerate(flist):
            T = pars[4+ngauss+i]
            # this is from dF/de
            fac1 = ellip/(1-ellip**2)
            fac2 = 2.*ellip*(x2+y2) - (1+ellip**2)*(x2my2*cos2theta + xy2*sin2theta)
            fac2 *= -1./T/(1.-ellip**2)**2
            # now multiply by de/deta and F
            jtmp += Fi*(fac1+fac2)*de_deta
        jacob.append(jtmp)

        # theta
        jtmp = zeros(self.image.shape)
        for i,Fi in enumerate(flist):
            T = pars[4+ngauss+i]
            fac = (x2my2*e2 - xy2*e1)
            fac *= -2./T/(1.-ellip**2)
            jtmp += Fi*fac
        jacob.append(jtmp)

        # p
        # have an entry for *each* gauss rather than summed
        for i,Fi in enumerate(flist):
            pi = pars[4+i]
            jtmp = Fi/pi
            jacob.append(jtmp)


        # T
        for i,Fi in enumerate(flist):
            T = pars[4+ngauss+i]
            fac1 = -1./T
            ch = (x2*(1.-e1) - xy2*e2 + y2*(1.+e1))/T/(1.-ellip**2)
            fac2 = ch/T
            jtmp = Fi*(fac1+fac2)
            jacob.append(jtmp)


        for i in xrange(len(jacob)):
            #print >>stderr,i
            #print >>stderr,jacob[i]
            jacob[i] = jacob[i].reshape(self.image.size)

        return jacob

    def jacob_eta_psf(self, pars):

        flist = self.make_model(pars, aslist=True)

        ngauss=self.ngauss
        y = self.row-pars[0]
        x = self.col-pars[1]

        x2=x**2
        y2=y**2

        x2my2 = x2 - y2
        r2 = x2 + y2
        xy2 = 2*x*y

        eta   = pars[2]
        de_deta = 0.5*(2./(exp(eta)+exp(-eta)))**2

        ellip = eta2ellip(eta)
        theta = pars[3]
        cos2theta = cos(2*theta)
        sin2theta = sin(2*theta)
        e1    = ellip*cos2theta
        e2    = ellip*sin2theta

        jacob = []

        jy0 = zeros(self.image.shape)
        jx0 = zeros(self.image.shape)
        jeta = zeros(self.image.shape)
        jtheta = zeros(self.image.shape)

        jp_list = []
        jT_list = []
        for i,plist in enumerate(flist):
            T = pars[4+ngauss+i]
            gp = pars[4+i]

            # we have one of these for each gaussian
            jp = zeros(self.image.shape)
            jT = zeros(self.image.shape)
            for j in xrange(len(plist)):
                p = self.psf[j]
                pim = plist[j] # convolved image

                Tpsf = p['irr']+p['icc']
                To = T + Tpsf

                e1psf = (p['icc']-p['irr'])/Tpsf
                e2psf = 2*p['irc']/Tpsf

                R = T/To
                s2 = Tpsf/T

                e1o = R*(e1 + e1psf*s2)
                e2o = R*(e2 + e2psf*s2)
                ellipo_2 = e1o**2 + e2o**2

                ellipo = sqrt(ellipo_2)
                # if ellipticity is zero, we don't care
                # about the angle!
                if ellipo_2 > 0:
                    cos2thetao = e1o/ellipo
                    sin2thetao = e2o/ellipo
                else:
                    cos2thetao = 1.0
                    sin2thetao = 0.0


                #
                # centroid
                #
                y0fac = y*(1.+e1o) - x*e2o
                y0fac *= 2./To/(1-ellipo_2)

                x0fac = x*(1.-e1o) - y*e2o
                x0fac *= 2./To/(1-ellipo_2)

                jy0 += pim*y0fac
                jx0 += pim*x0fac

                #
                # eta
                #
                eta_fac1 = ellipo/(1-ellipo_2)
                eta_fac2 = \
                    2.*ellipo*r2 \
                    - (1+ellipo_2)*(x2my2*cos2thetao + xy2*sin2thetao)
                eta_fac2 *= -1./To/(1.-ellipo_2)**2

                # derivitive of observed ellipticity with respect to 
                # object e
                deo_de = R*(cos2thetao*cos2theta + sin2thetao*sin2theta)

                # now multiply by F * deo/de * de/deta
                jeta += pim*(eta_fac1+eta_fac2)*deo_de*de_deta

                #
                # theta
                #
                om = 1./(1.-ellipo_2)
                Tom2 = om**2/To

                theta_fac1 = \
                    -2*R*e2*(e1o*om + x2my2*Tom2*(1-ellipo_2+2*e1o**2) )
                theta_fac2 = \
                     2*R*e1*(e2o*om +   xy2*Tom2*(1-ellipo_2+2*e2o**2) )

                # this gives the exact same answer as above
                #fac1 = -2*R*e2*( ((e1o**2-e2o**2+1)*(x**2-y**2)-To*e1o*(e1o**2+e2o**2-1)))/(To*(-e1o**2-e2o**2+1)**2)
                #fac2 =  2*R*e1*( (2*x*y*(e2o**2-e1o**2+1)-To*e2o*(e2o**2+e1o**2-1)))/(To*(-e2o**2-e1o**2+1)**2)

                jtheta += pim*(theta_fac1+theta_fac2)

                #
                # counts
                #
                jp += pim

                #
                # T
                #
                Tfac1 = -1./To
                ch = (x2*(1.-e1o) - xy2*e2o + y2*(1.+e1o))/To/(1.-ellipo_2)
                Tfac2 = ch/To
                jT += pim*(Tfac1+Tfac2)


            jp /= gp
            jp_list.append(jp)
            jT_list.append(jT)


        jacob.append(jy0)
        jacob.append(jx0)
        jacob.append(jeta)
        jacob.append(jtheta)
        jacob += jp_list
        jacob += jT_list

        for i in xrange(len(jacob)):
            jacob[i] = jacob[i].reshape(self.image.size)
        return jacob



    def jacob_cov(self, pars):
        """
        Calculate the jacobian of the function for each parameter
        using the covariance parametrization
        """
        if self.psf is not None:
            return self.jacob_cov_psf(pars)

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

    def jacob_cov_psf(self, pars):
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


class GMixFitCoellipFix:
    """
    This one is for testing particular parts of the jacobian: keep
    all parameters except one, or one type, fixed.

    Perform a non-linear fit of the image to a gaussian mixture model.  
    
    The gaussians are forced to be co-elliptical.  Image is assumed to be
    sky-subtracted

    parameters
    ----------
    image: numpy array, dim=2
        A background-subtracted image.
    guess:
        A starting guess for the parameters.  If ptype=='ellip'
        then this array is
            [cen0,cen1,eta,theta,pi,Ti]
        Where i runs over all gaussians. Eta is defined by
            ellip=(1+tanh(eta))/2.
        Which allows the bounds of eta to b -inf,inf.

        Note ngauss=(len(guess)-4)/2
        
        If ptype=='cov' then the parameters are
            [cen0,cen1,cov00,cov01,cov11,pi,fi]

    ptype: string
        Either 'e1e2' or 'cov' or 'eta'
    error: float
        The error per pixel.  Note currently used.
    psf: list of dictionaries
        A gaussian mixture model specified as a list of
        dictionaries.  Each dict has these entries
            p: A normalization
            row: center of the gaussian in the first dimension
            col: center of the gaussian in the second dimension
            irr: Covariance matrix element for row*row
            irc: Covariance matrix element for row*col
            icc: Covariance matrix element for col*col

    method: string
        method for the fit.  Only 'lm' is useful currently.
    verbose: bool
    """
    def __init__(self, image, guess, imove,
                 use_jacob=True,
                 ptype='cov',
                 error=None, 
                 psf=None, 
                 method='lm', 
                 verbose=False):

        self.imove = imove
        self.use_jacob=use_jacob

        self.image=image
        self.error=error # per pixel
        self.guess=guess
        self.ptype=ptype
        self.ngauss=(len(guess)-4)/2
        self.nsub=1
        self.verbose=verbose

        # can enter the psf model as a mixture model or
        # a coelliptical psf model; gmix is more flexible..
        # we keep the gmix version since we use gmix2image to
        # make models
        if isinstance(psf,numpy.ndarray):
            self.psf = pars2gmix_coellip(psf,ptype=self.ptype)
        else:
            self.psf = psf

        self.row,self.col=ogrid[0:image.shape[0], 0:image.shape[1]]

        if method=='lm':
            self.dofit_lm()
        else:
            raise ValueError("expected method 'lm'")

    def get_full_pars(self, pars1):
        pars = self.guess.copy()
        pars[self.imove] = pars1[0]
        return pars

    def dofit_lm(self):
        """
        Run the fit using LM
        """
        from scipy.optimize import leastsq
        guess = array([self.guess[self.imove]])

        if self.use_jacob:
            res = leastsq(self.ydiff,
                          guess,
                          full_output=1,
                          Dfun=self.jacob,
                          col_deriv=1)
        else:
            res = leastsq(self.ydiff,
                          guess,
                          full_output=1)

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
            w,=where(d < 0)

            if w.size == 0:
                # only do if non negative
                self.perr = sqrt(d)

        self.set_lm_flags()

       
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
            #import images
            #print >>stderr,'pcov:'
            #images.imprint(self.pcov,stream=stderr)
            e,v = eig(self.pcov)
            weig,=where(e <= 0)
            if weig.size > 0:
                if self.verbose:
                    import images
                    print >>stderr,'negative covariance eigenvalues'
                    #images.imprint(self.pcov,stream=stderr)
                flags += GMIXFIT_NEG_COV_EIG 

            wneg,=where(diag(self.pcov) <= 0)
            if wneg.size > 0:
                if self.verbose:
                    import images
                    # only print if we didn't see negative eigenvalue
                    print >>stderr,'negative covariance diagonals'
                    #images.imprint(self.pcov,stream=stderr)
                flags += GMIXFIT_NEG_COV_DIAG 

            if self.ptype == 'cov':
                mcov = self.pcov[2:2+3,2:2+3].copy()
                me,v = eig(mcov)
                weig,=where(me <= 0)
                if weig.size > 0:
                    if self.verbose:
                        import images
                        print >>stderr,'negative M covariance eigenvalues'
                        print >>stderr,me
                        #images.imprint(mcov,stream=stderr)
                    flags += GMIXFIT_NEG_MCOV_DIAG

                # this has more strict checks it seems
                try:
                    MM = numpy.linalg.cholesky(mcov)
                except numpy.linalg.LinAlgError as e:
                    if self.verbose:
                        import images
                        print >>stderr,e
                        #images.imprint(mcov,stream=stderr)
                    flags += GMIXFIT_MCOV_NOTPOSDEF

                if ( ((flags & GMIXFIT_MCOV_NOTPOSDEF) != 0)
                        and ((flags & GMIXFIT_NEG_MCOV_DIAG) == 0) ):
                    import images
                    print >>stderr,'found one:'
                    print >>stderr,me
                    #images.imprint(mcov,stream=stderr,fmt='%.16e')
                    #print >>stderr,mcov.dtype.descr
                    #print >>stderr,mcov.flags
                    #print >>stderr,mcov.shape
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

    def get_gmix(self):
        return pars2gmix_coellip(self.popt, ptype=self.ptype)
    gmix = property(get_gmix)

    def ydiff(self, pars1):
        """
        Also apply hard priors on centroid range and determinant(s)
        """
        import images

        if self.ptype == 'e1e2':
            if not self.check_hard_priors_e1e2(pars1):
                return zeros(self.image.size) + numpy.inf

        model = self.make_model(pars1)
        #images.multiview(model)
        #stop
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

    def check_hard_priors_e1e2(self, pars1):
        pars = self.get_full_pars(pars1)
        vals=pars[4:]
        w,=where(vals <= 0)
        if w.size > 0:
            print >>stderr,'bad p/T'
            print vals
            return False

        # check determinant for all images we might have
        # to create with psf convolutions

        gmix=pars2gmix_coellip(pars, ptype=self.ptype)
        if self.psf is not None:
            moms = total_moms_psf(gmix, self.psf)
            pdet = moms['irr']*moms['icc']-moms['irc']**2
            if pdet <= 0:
                print >>stderr,'bad p det'
                return False
            for g in gmix:
                for p in self.psf:
                    irr = g['irr'] + p['irr']
                    irc = g['irc'] + p['irc']
                    icc = g['icc'] + p['icc']
                    det = irr*icc-irc**2
                    if det <= 0:
                        print >>stderr,'bad p+obj det'
                        return False

        # overall determinant and centroid
        g0 = gmix[0]
        det = g0['irr']*g0['icc'] - g0['irc']**2
        if (det <= 0 
                or pars[0] < 0 or pars[0] > (self.image.shape[0]-1)
                or pars[1] < 0 or pars[1] > (self.image.shape[1]-1)):
            print >>stderr,'bad det or centroid'
            return False

        return True

    def make_model(self, pars1, aslist=False):
        pars = self.get_full_pars(pars1)
        #print 'full pars:',pars
        gmix = pars2gmix_coellip(pars, ptype=self.ptype)
        return gmix2image(gmix, 
                          self.image.shape, 
                          psf=self.psf,
                          aslist=aslist, 
                          nsub=self.nsub,
                          renorm=False)

    def jacob(self, pars):
        if self.ptype == 'eta':
            return self.jacob_eta(pars)
        elif self.ptype == 'cov':
            return self.jacob_cov(pars)
        elif self.ptype == 'e1e2':
            return self.jacob_e1e2(pars)
        else:
            raise ValueError("ptype must be 'e1e2','cov','eta'")

    def jacob_eta_psf(self, pars1):

        flist = self.make_model(pars1, aslist=True)
        pars=self.get_full_pars(pars1)

        ngauss=self.ngauss
        y = self.row-pars[0]
        x = self.col-pars[1]

        x2my2 = x**2 - y**2
        xy2 = 2*x*y
        x2=x**2
        y2=y**2

        eta   = pars[2]
        de_deta = 0.5*(2./(exp(eta)+exp(-eta)))**2

        ellip = eta2ellip(eta)
        theta = pars[3]
        cos2theta = cos(2*theta)
        sin2theta = sin(2*theta)
        e1    = ellip*cos2theta
        e2    = ellip*sin2theta

        jacob = []
        if self.imove == 0:
            print >>stderr,"doing psf y0"
            jy0 = zeros(self.image.shape)

            for i,plist in enumerate(flist):
                T = pars[4+ngauss+i]
                for j in xrange(len(plist)):
                    p = self.psf[j]
                    pim = plist[j] # convolved image

                    Tpsf = p['irr']+p['icc']
                    To = T + Tpsf

                    e1psf = (p['icc']-p['irr'])/Tpsf
                    e2psf = 2*p['irc']/Tpsf

                    R = T/To
                    s2 = Tpsf/T

                    e1o = R*(e1 + e1psf*s2)
                    e2o = R*(e2 + e2psf*s2)
                    ellipo_2 = e1o**2 + e2o**2

                    yfac = y*(1.+e1o) - x*e2o
                    yfac *= 2./To/(1-ellipo_2)

                    jy0 += pim*yfac

            jacob.append(jy0)


        if self.imove == 1:
            print >>stderr,"doing psf x0"
            jx0 = zeros(self.image.shape)

            for i,plist in enumerate(flist):
                T = pars[4+ngauss+i]
                for j in xrange(len(plist)):
                    p = self.psf[j]
                    pim = plist[j] # convolved image

                    Tpsf = p['irr']+p['icc']
                    To = T + Tpsf

                    e1psf = (p['icc']-p['irr'])/Tpsf
                    e2psf = 2*p['irc']/Tpsf

                    R = T/To
                    s2 = Tpsf/T

                    e1o = R*(e1 + e1psf*s2)
                    e2o = R*(e2 + e2psf*s2)
                    ellipo_2 = e1o**2 + e2o**2

                    xfac = x*(1.-e1o) - y*e2o
                    xfac *= 2./To/(1-ellipo_2)

                    jx0 += pim*xfac

            jacob.append(jx0)

        if self.imove == 2:
            print >>stderr,"doing psf eta"
            jeta = zeros(self.image.shape)

            for i,plist in enumerate(flist):
                T = pars[4+ngauss+i]
                for j in xrange(len(plist)):
                    p = self.psf[j]
                    pim = plist[j] # convolved image

                    Tpsf = p['irr']+p['icc']
                    To = T + Tpsf

                    e1psf = (p['icc']-p['irr'])/Tpsf
                    e2psf = 2*p['irc']/Tpsf

                    R = T/To
                    s2 = Tpsf/T

                    e1o = R*(e1 + e1psf*s2)
                    e2o = R*(e2 + e2psf*s2)
                    ellipo_2 = e1o**2 + e2o**2
                    ellipo = sqrt(ellipo_2)
                    # if ellipticity is zero, we don't care
                    # about the angle!
                    if ellipo_2 > 0:
                        cos2thetao = e1o/ellipo
                        sin2thetao = e2o/ellipo
                    else:
                        cos2thetao = 1.0
                        sin2thetao = 0.0

                    fac1 = ellipo/(1-ellipo_2)
                    fac2 = 2.*ellipo*(x2+y2) \
                            - (1+ellipo_2)*(x2my2*cos2thetao + xy2*sin2thetao)
                    fac2 *= -1./To/(1.-ellipo_2)**2

                    # derivitive of observed ellipticity with respect to 
                    # object e
                    deo_de = R*(cos2thetao*cos2theta + sin2thetao*sin2theta)

                    # now multiply by F * deo/de * de/deta
                    jeta += pim*(fac1+fac2)*deo_de*de_deta

            jacob.append(jeta)

        if self.imove == 3:
            print >>stderr,"doing psf theta"
            jtheta = zeros(self.image.shape)

            for i,plist in enumerate(flist):
                T = pars[4+ngauss+i]
                for j in xrange(len(plist)):
                    p = self.psf[j]
                    pim = plist[j] # convolved image

                    Tpsf = p['irr']+p['icc']
                    To = T + Tpsf

                    e1psf = (p['icc']-p['irr'])/Tpsf
                    e2psf = 2*p['irc']/Tpsf

                    R = T/To
                    s2 = Tpsf/T

                    e1o = R*(e1 + e1psf*s2)
                    e2o = R*(e2 + e2psf*s2)
                    ellipo_2 = e1o**2 + e2o**2
                    ellipo = sqrt(ellipo_2)

                    om = 1./(1.-ellipo_2)
                    Tom2 = om**2/To

                    fac1 = \
                        -2*R*e2*(e1o*om + x2my2*Tom2*(1-ellipo_2+2*e1o**2) )
                    fac2 = \
                         2*R*e1*(e2o*om +   xy2*Tom2*(1-ellipo_2+2*e2o**2) )

                    # this gives the exact same answer as above
                    #fac1 = -2*R*e2*( ((e1o**2-e2o**2+1)*(x**2-y**2)-To*e1o*(e1o**2+e2o**2-1)))/(To*(-e1o**2-e2o**2+1)**2)
                    #fac2 =  2*R*e1*( (2*x*y*(e2o**2-e1o**2+1)-To*e2o*(e2o**2+e1o**2-1)))/(To*(-e2o**2-e1o**2+1)**2)

                    jtheta += pim*(fac1+fac2)
            jacob.append(jtheta)

        # p
        # have an entry for *each* gauss rather than summed
        for i,plist in enumerate(flist):
            if self.imove == (4+i):
                print >>stderr,"doing psf p%d" % i
                gp = pars[4+i]
                jp = zeros(self.image.shape)
                for j in xrange(len(plist)):
                    p = self.psf[j]
                    pim = plist[j] # convolved image

                    jp += pim
                jp /= gp
                jacob.append(jp)

        for i,plist in enumerate(flist):
            if self.imove == (4+ngauss+i):
                print >>stderr,"doing psf T%d" % i
                T = pars[4+ngauss+i]
                jT = zeros(self.image.shape)
                for j in xrange(len(plist)):
                    p = self.psf[j]
                    pim = plist[j] # convolved image

                    Tpsf = p['irr']+p['icc']
                    To = T + Tpsf

                    e1psf = (p['icc']-p['irr'])/Tpsf
                    e2psf = 2*p['irc']/Tpsf

                    R = T/To
                    s2 = Tpsf/T

                    e1o = R*(e1 + e1psf*s2)
                    e2o = R*(e2 + e2psf*s2)
                    ellipo_2 = e1o**2 + e2o**2

                    fac1 = -1./To
                    ch = (x2*(1.-e1o) - xy2*e2o + y2*(1.+e1o))/To/(1.-ellipo_2)
                    fac2 = ch/To
                    jT += pim*(fac1+fac2)

                jacob.append(jT)


        for i in xrange(len(jacob)):
            jacob[i] = jacob[i].reshape(self.image.size)
        return jacob


    def jacob_eta(self, pars1):
        """
        Calculate the jacobian of the function for each parameter
        using the eta parametrization
        """
        import images
        if self.psf is not None:
            return self.jacob_eta_psf(pars1)

        pars=self.get_full_pars(pars1)
        #print >>stderr,"pars:",pars
        ngauss=self.ngauss
        y = self.row-pars[0]
        x = self.col-pars[1]

        flist = self.make_model(pars1, aslist=True)

        eta   = pars[2]
        de_deta = 0.5*(2./(exp(eta)+exp(-eta)))**2

        ellip = eta2ellip(eta)
        theta = pars[3]
        cos2theta = cos(2*theta)
        sin2theta = sin(2*theta)
        e1    = ellip*cos2theta
        e2    = ellip*sin2theta
        overe2 = 1./(1.-ellip**2)

        x2my2 = x**2 - y**2
        xy2 = 2*x*y
        x2=x**2
        y2=y**2

        jacob = []

        # 
        # for cen we sum up contributions from each gauss
        # 

        # y0,x0
        if self.imove == 0:
            print >>stderr,"doing y0"
            jy0 = zeros(self.image.shape)
            #jx0 = zeros(self.image.shape)
            for i,Fi in enumerate(flist):
                T = pars[4+ngauss+i]

                yfac = y*(1.+e1) - x*e2
                #yfac = y*(1.-e1) - x*e2
                yfac *= 2./T/(1-ellip**2)

                jy0 += Fi*yfac

            #images.multiview(jy0,title='y0')
            #stop
            jacob.append(jy0)

        if self.imove == 1:
            print >>stderr,"doing x0"
            jx0 = zeros(self.image.shape)
            for i,Fi in enumerate(flist):
                T = pars[4+ngauss+i]

                xfac = x*(1.-e1) - y*e2
                xfac *= 2./T/(1-ellip**2)

                jx0 += Fi*xfac

            #images.multiview(jx0,title='x0')
            #stop
            #print >>stderr,"jx0:",jx0
            jacob.append(jx0)


        # eta
        
        # just the chain rule
        # we calculate dF/de*de/d(eta)  and use
        #   ellip=(1+tanh(eta))/2
        # this is the derivative of ellip with respect
        # to eta, using d(tanh)/d(eta) = sech^2(eta)
        # = ( 2/(e^(\eta) + e^(-\eta)) )^2

        if self.imove == 2:
            print >>stderr,"doing eta"
            jtmp = zeros(self.image.shape)
            for i,Fi in enumerate(flist):
                T = pars[4+ngauss+i]
                # this is from dF/de
                fac1 = ellip/(1-ellip**2)
                fac2 = 2.*ellip*(x2+y2) - (1+ellip**2)*(x2my2*cos2theta + xy2*sin2theta)
                fac2 *= -1./T/(1.-ellip**2)**2
                # now multiply by de/deta and F
                jtmp += Fi*(fac1+fac2)*de_deta
            jacob.append(jtmp)

        if self.imove == 3:
            # theta
            print >>stderr,"doing theta"
            jtmp = zeros(self.image.shape)
            for i,Fi in enumerate(flist):
                T = pars[4+ngauss+i]
                fac = (x2my2*e2 - xy2*e1)
                fac *= -2./T/(1.-ellip**2)
                jtmp += Fi*fac
            jacob.append(jtmp)

        # p
        # have an entry for *each* gauss rather than summed
        for i,Fi in enumerate(flist):
            if self.imove == (4+i):
                print >>stderr,"doing p%d" % i
                pi = pars[4+i]
                jtmp = Fi/pi
                jacob.append(jtmp)


        # T
        for i,Fi in enumerate(flist):
            if self.imove == (4+ngauss+i):
                print >>stderr,"doing T%d" % i
                T = pars[4+ngauss+i]
                fac1 = -1./T
                ch = (x2*(1.-e1) - xy2*e2 + y2*(1.+e1))/T/(1.-ellip**2)
                fac2 = ch/T
                jtmp = Fi*(fac1+fac2)
                jacob.append(jtmp)


        for i in xrange(len(jacob)):
            #print >>stderr,i
            #print >>stderr,jacob[i]
            jacob[i] = jacob[i].reshape(self.image.size)
            """
            w,=where(isfinite(jacob[i]) == False)
            if w.size != 0:
                print i
                print jacob[i]
                stop
            """

        return jacob

    def jacob_e1e2_psf(self, pars1):

        flist = self.make_model(pars1, aslist=True)
        pars=self.get_full_pars(pars1)

        ngauss=self.ngauss
        y = self.row-pars[0]
        x = self.col-pars[1]

        x2my2 = x**2 - y**2
        xy2 = 2*x*y
        x2=x**2
        y2=y**2

        e1    = pars[2]
        e2    = pars[3]

        jacob = []
        if self.imove == 0:
            print >>stderr,"doing psf y0"
            jy0 = zeros(self.image.shape)

            for i,plist in enumerate(flist):
                T = pars[4+ngauss+i]
                for j in xrange(len(plist)):
                    p = self.psf[j]
                    pim = plist[j] # convolved image

                    Tpsf = p['irr']+p['icc']
                    To = T + Tpsf

                    e1psf = (p['icc']-p['irr'])/Tpsf
                    e2psf = 2*p['irc']/Tpsf

                    R = T/To
                    s2 = Tpsf/T

                    e1o = R*(e1 + e1psf*s2)
                    e2o = R*(e2 + e2psf*s2)
                    ellipo_2 = e1o**2 + e2o**2

                    yfac = y*(1.+e1o) - x*e2o
                    yfac *= 2./To/(1-ellipo_2)

                    jy0 += pim*yfac

            jacob.append(jy0)


        if self.imove == 1:
            print >>stderr,"doing psf x0"
            jx0 = zeros(self.image.shape)

            for i,plist in enumerate(flist):
                T = pars[4+ngauss+i]
                for j in xrange(len(plist)):
                    p = self.psf[j]
                    pim = plist[j] # convolved image

                    Tpsf = p['irr']+p['icc']
                    To = T + Tpsf

                    e1psf = (p['icc']-p['irr'])/Tpsf
                    e2psf = 2*p['irc']/Tpsf

                    R = T/To
                    s2 = Tpsf/T

                    e1o = R*(e1 + e1psf*s2)
                    e2o = R*(e2 + e2psf*s2)
                    ellipo_2 = e1o**2 + e2o**2

                    xfac = x*(1.-e1o) - y*e2o
                    xfac *= 2./To/(1-ellipo_2)

                    jx0 += pim*xfac

            jacob.append(jx0)

        if self.imove == 2:
            print >>stderr,"doing psf e1"
            je1 = zeros(self.image.shape)

            for i,plist in enumerate(flist):
                T = pars[4+ngauss+i]
                for j in xrange(len(plist)):
                    p = self.psf[j]
                    pim = plist[j] # convolved image

                    Tpsf = p['irr']+p['icc']
                    To = T + Tpsf

                    e1psf = (p['icc']-p['irr'])/Tpsf
                    e2psf = 2*p['irc']/Tpsf

                    R = T/To
                    s2 = Tpsf/T

                    e1o = R*(e1 + e1psf*s2)
                    e2o = R*(e2 + e2psf*s2)
                    ellipo_2 = e1o**2 + e2o**2

                    e1fac1 = e1o/(1-ellipo_2)

                    e1fac2 = x2my2
                    e1fac2 *= (1-ellipo_2+2*e1o**2)/To/(1.-ellipo_2)**2

                    de1o_de1 = R

                    e1fac = de1o_de1*(e1fac1 + e1fac2)
                    je1 += pim*e1fac

            jacob.append(je1)

        if self.imove == 3:
            print >>stderr,"doing psf e2"
            je2 = zeros(self.image.shape)

            for i,plist in enumerate(flist):
                T = pars[4+ngauss+i]
                for j in xrange(len(plist)):
                    p = self.psf[j]
                    pim = plist[j] # convolved image

                    Tpsf = p['irr']+p['icc']
                    To = T + Tpsf

                    e1psf = (p['icc']-p['irr'])/Tpsf
                    e2psf = 2*p['irc']/Tpsf

                    R = T/To
                    s2 = Tpsf/T

                    e1o = R*(e1 + e1psf*s2)
                    e2o = R*(e2 + e2psf*s2)
                    ellipo_2 = e1o**2 + e2o**2

                    e2fac1 = e2o/(1-ellipo_2)
                    e2fac2 = xy2*(1-ellipo_2 + 2*e2o**2)
                    e2fac2 *= 1./T/(1.-ellipo_2)**2
                    
                    de2o_de2 = R

                    e2fac = de2o_de2*(e2fac1 + e2fac2)
                    je2 += pim*e2fac


            jacob.append(je2)




        # p
        # have an entry for *each* gauss rather than summed
        for i,plist in enumerate(flist):
            if self.imove == (4+i):
                print >>stderr,"doing psf p%d" % i
                gp = pars[4+i]
                jp = zeros(self.image.shape)
                for j in xrange(len(plist)):
                    p = self.psf[j]
                    pim = plist[j] # convolved image

                    jp += pim
                jp /= gp
                jacob.append(jp)

        for i,plist in enumerate(flist):
            if self.imove == (4+ngauss+i):
                print >>stderr,"doing psf T%d" % i
                T = pars[4+ngauss+i]
                jT = zeros(self.image.shape)
                for j in xrange(len(plist)):
                    p = self.psf[j]
                    pim = plist[j] # convolved image

                    Tpsf = p['irr']+p['icc']
                    To = T + Tpsf

                    e1psf = (p['icc']-p['irr'])/Tpsf
                    e2psf = 2*p['irc']/Tpsf

                    R = T/To
                    s2 = Tpsf/T

                    e1o = R*(e1 + e1psf*s2)
                    e2o = R*(e2 + e2psf*s2)
                    ellipo_2 = e1o**2 + e2o**2

                    fac1 = -1./To
                    ch = (x2*(1.-e1o) - xy2*e2o + y2*(1.+e1o))/To/(1.-ellipo_2)
                    fac2 = ch/To
                    jT += pim*(fac1+fac2)

                jacob.append(jT)


        for i in xrange(len(jacob)):
            jacob[i] = jacob[i].reshape(self.image.size)
        return jacob


    def jacob_e1e2(self, pars1):
        """
        Calculate the jacobian of the function for each parameter
        using the eta parametrization
        """
        import images
        if self.psf is not None:
            return self.jacob_e1e2_psf(pars1)

        pars=self.get_full_pars(pars1)
        #print >>stderr,"pars:",pars
        ngauss=self.ngauss
        y = self.row-pars[0]
        x = self.col-pars[1]

        flist = self.make_model(pars1, aslist=True)

        e1    = pars[2]
        e2    = pars[3]
        ellip = sqrt(e1**2 + e2**2)
        overe2 = 1./(1.-ellip**2)

        x2my2 = x**2 - y**2
        xy2 = 2*x*y
        x2=x**2
        y2=y**2

        jacob = []

        # 
        # for cen we sum up contributions from each gauss
        # 

        # y0,x0
        if self.imove == 0:
            print >>stderr,"doing y0"
            jy0 = zeros(self.image.shape)
            #jx0 = zeros(self.image.shape)
            for i,Fi in enumerate(flist):
                T = pars[4+ngauss+i]

                yfac = y*(1.+e1) - x*e2
                yfac *= 2./T/(1-ellip**2)

                jy0 += Fi*yfac

            jacob.append(jy0)

        if self.imove == 1:
            print >>stderr,"doing x0"
            jx0 = zeros(self.image.shape)
            for i,Fi in enumerate(flist):
                T = pars[4+ngauss+i]

                xfac = x*(1.-e1) - y*e2
                xfac *= 2./T/(1-ellip**2)

                jx0 += Fi*xfac

            jacob.append(jx0)


        if self.imove == 2:
            print >>stderr,"doing e1"
            jtmp = zeros(self.image.shape)
            for i,Fi in enumerate(flist):
                T = pars[4+ngauss+i]

                fac1 = e1/(1-ellip**2)
                fac2 = x2my2*(1-ellip**2 + 2*e1**2)
                fac2 *= 1./T/(1.-ellip**2)**2
                # now multiply by de/deta and F
                jtmp += Fi*(fac1+fac2)
            jacob.append(jtmp)

        if self.imove == 3:
            print >>stderr,"doing e2"
            jtmp = zeros(self.image.shape)
            for i,Fi in enumerate(flist):
                T = pars[4+ngauss+i]

                fac1 = e2/(1-ellip**2)
                fac2 = xy2*(1-ellip**2 + 2*e2**2)
                fac2 *= 1./T/(1.-ellip**2)**2
                # now multiply by de/deta and F
                jtmp += Fi*(fac1+fac2)
            jacob.append(jtmp)



        # p
        # have an entry for *each* gauss rather than summed
        for i,Fi in enumerate(flist):
            if self.imove == (4+i):
                print >>stderr,"doing p%d" % i
                pi = pars[4+i]
                jtmp = Fi/pi
                jacob.append(jtmp)


        # T
        for i,Fi in enumerate(flist):
            if self.imove == (4+ngauss+i):
                print >>stderr,"doing T%d" % i
                T = pars[4+ngauss+i]
                fac1 = -1./T
                ch = (x2*(1.-e1) - xy2*e2 + y2*(1.+e1))/T/(1.-ellip**2)
                fac2 = ch/T
                jtmp = Fi*(fac1+fac2)
                jacob.append(jtmp)


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



def pars2gmix_coellip(pars, ptype='cov'):
    """
    Convert a parameter array as used for the LM code into a gaussian mixture
    model.  This is for the case of co-elliptical gaussians.
    """
    if ptype=='cov':
        return pars2gmix_coellip_cov(pars)
    elif ptype=='e1e2':
        return pars2gmix_coellip_e1e2(pars)
    elif ptype=='eta':
        return pars2gmix_coellip_eta(pars)
    else:
        raise ValueError("ptype should be in ['cov','eta']")

def pars2gmix_coellip_cov(pars):
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

def eta2ellip(eta):
    """
    This is not eta from BJ02
    """
    return (1.+tanh(eta))/2.
def ellip2eta(ellip):
    """
    This is not eta from BJ02
    """
    return arctanh( 2*ellip-1 )

def pars2gmix_coellip_eta(pars):
    """
    Convert a parameter array as used for the LM code into a gaussian mixture
    model.  This is for the case of co-elliptical gaussians.
    """
    #print 'making gmix by eta'
    ngauss = (len(pars)-4)/2
    gmix=[]

    eta = pars[2]
    ellip = eta2ellip(eta)
    theta=pars[3]
    e1 = ellip*cos(2*theta)
    e2 = ellip*sin(2*theta)
    
    for i in xrange(ngauss):
        d={}
        p = pars[4+i]
        T = pars[4+ngauss+i]
        d['p'] = p
        d['row'] = pars[0]
        d['col'] = pars[1]
        #d['irr'] = (T/2.)*(1+e1)
        d['irr'] = (T/2.)*(1-e1)
        d['irc'] = (T/2.)*e2
        #d['icc'] = (T/2.)*(1-e1)
        d['icc'] = (T/2.)*(1+e1)
        gmix.append(d)

    return gmix

def pars2gmix_coellip_e1e2(pars):
    """
    Convert a parameter array as used for the LM code into a gaussian mixture
    model.  This is for the case of co-elliptical gaussians.
    """
    #print 'making gmix by eta'
    ngauss = (len(pars)-4)/2
    gmix=[]

    e1 = pars[2]
    e2 = pars[3]
    
    for i in xrange(ngauss):
        d={}

        p = pars[4+i]
        T = pars[4+ngauss+i]

        d['p'] = p
        d['row'] = pars[0]
        d['col'] = pars[1]
        d['irr'] = (T/2.)*(1-e1)
        d['irc'] = (T/2.)*e2
        d['icc'] = (T/2.)*(1+e1)
        gmix.append(d)

    return gmix


def print_pars(pars, stream=stderr, front=None):
    """
    print the parameters with a uniform width
    """
    if front is not None:
        stream.write(front)
        stream.write(' ')
    if pars is None:
        stream.write('%s\n' % None)
    else:
        fmt = ' '.join( ['%10.6g ']*len(pars) )
        stream.write(fmt % tuple(pars))
        stream.write('\n')
