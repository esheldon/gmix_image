import numpy
from numpy import zeros, array, where, ogrid
from fimage import model_image

from gmix_image import gmix2image

class GMixFitCoellip:
    """
    Use levenberg marquardt to fit a gaussian mixture model.  The gaussians are
    forced to be co-elliptical.
    
    Image is assumed to be sky-subtracted

    When we get a mask image, we can use that too within ydiff
    """
    def __init__(self, image, ngauss, psf=None):
        self.image=image
        self.counts = image.sum()
        self.ngauss=ngauss
        self.nsub=1

        # can enter the psf model as a mixture model or
        # a coelliptical psf model; gmix is more flexible..
        # we keep the gmix version since we use gmix2image to
        # make models
        if isinstance(psf,numpy.ndarray):
            self.psf = pars2gmix_coellip(psf)
        else:
            self.psf = psf

        self.row,self.col=ogrid[0:image.shape[0], 0:image.shape[1]]

    def ydiff(self, pars):
        """
        Also apply hard priors on centroid range
        and determinant
        """

        # make sure p and f values are > 0
        vals = pars[5:]
        w,=where(vals <= 0)
        if w.size > 0:
            return zeros(self.image.size) + numpy.inf

        det = pars[2]*pars[4]-pars[3]**2
        if (det <= 0 
                or pars[0] < 0 or pars[0] > (self.image.shape[0]-1)
                or pars[1] < 0 or pars[1] > (self.image.shape[1]-1)):
            return zeros(self.image.size) + numpy.inf

        model = self.make_model(pars)
        return (model-self.image).reshape(self.image.size)

    def make_model(self, pars, aslist=False):
        """
        pars = [row,col,irr,irc,icc,pi...,fi...]
        """
        gmix = pars2gmix_coellip(pars)
        return gmix2image(gmix, self.image.shape, 
                          psf=self.psf,
                          aslist=aslist, 
                          order='f', renorm=False)

    def make_model_old(self, pars, aslist=False):
        """
        pars = [row,col,irr,irc,icc,pi...,fi...]
        """
        if aslist:
            modlist=[]
        else:
            self.model[:] = 0.0
        cen = pars[0:2]
        countsper = self.counts/self.ngauss
        for i in xrange(self.ngauss):
            pi = pars[5+i]
            if i > 0:
                fi = pars[5+self.ngauss+i-1]
                covari = fi*pars[2:2+3]
            else:
                covari = pars[2:2+3]
            model = model_image('gauss',
                                self.image.shape,
                                cen,covari,
                                counts=pi*countsper,
                                nsub=self.nsub,
                                order='c')
            if aslist:
                modlist.append(model)
            else:
                self.model += model

        if aslist:
            return modlist
        else:
            return self.model


    def jacob(self, pars):
        """
        Calculate the jacobian for each parameter
        """
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
        # [r0, c0, Irr, Irc, Icc, pi, fi]
        det = pars[2]*pars[4]-pars[3]**2
        y = self.row-pars[0]
        x = self.col-pars[1]

        #f = self.make_model(pars)
        flist = self.make_model(pars, aslist=True)

        Myy = pars[2]
        Mxy = pars[3]
        Mxx = pars[4]

        Myy_list = [(Myy+p['irr']) for p in self.psf]
        Mxy_list = [(Mxy+p['irc']) for p in self.psf]
        Mxx_list = [(Mxx+p['icc']) for p in self.psf]
        
        '''
        Myy_x = Myy*x
        Mxy_y = Mxy*y
        Mxy_x = Mxy*x
        Mxx_y = Mxx*y
        '''

        Myy_x_list = [(Myy+p['irr'])*x for p in self.psf]
        Mxy_y_list = [(Mxy+p['irc'])*y for p in self.psf]
        Mxy_x_list = [(Mxy+p['irc'])*x for p in self.psf]
        Mxx_y_list = [(Mxx+p['icc'])*y for p in self.psf]

        jacob = []

        #
        # for cen,cov we sum up contributions from each gauss
        #

        # y0
        jtmp = zeros(self.image.shape)
        #fac = (Mxx_y - Mxy_x)/det
        #for i,Fi in enumerate(flist):
        #    jtmp += Fi*fac

        for i,plist in enumerate(flist):
            fi = 1 if i == 0 else pars[5+self.ngauss+i-1]
            for j in xrange(len(plist)):
                p = self.psf[j]
                pim = plist[j]
                
                tMyy = Myy*fi + p['irr']
                tMxy = Mxy*fi + p['irc']
                tMxx = Mxx*fi + p['icc']

                det = (tMyy*tMxx - tMxy**2)
                fac = (tMxx*y - tMxy*x)/det

                jtmp += pim*fac
        jacob.append(jtmp)


        # x0
        jtmp = zeros(self.image.shape)
        #fac = (Myy_x - Mxy_y)/det
        #for i,Fi in enumerate(flist):
        #    jtmp += Fi*fac
        for plist in flist:
            for i in xrange(len(plist)):
                pim = plist[i]
                det = (Myy_list[i]*Mxx_list[i]-Mxy_list[i]**2)
                Myy_x = Myy_x_list[i]
                Mxy_y = Mxy_y_list[i]
                fac = (Myy_x - Mxy_y)/det

                jtmp += pim*fac

        jacob.append(jtmp)

        # Myy (a)
        jtmp = zeros(self.image.shape)
        fac = (-Mxx/det  + 0.5*((Mxx_y-Mxy_x)/det)**2)
        for i,Fi in enumerate(flist):
            if i > 0:
                fi = pars[5+self.ngauss+i-1]
            else:
                fi=1.
            jtmp += Fi*fac/fi**2

        for j,plist in enumerate(flist):
            if j > 0:
                fj = pars[5+self.ngauss+j-1]
            else:
                fj=1.
            for i in xrange(len(plist)):
                pim = plist[i]
                det = (Myy_list[i]*Mxx_list[i]-Mxy_list[i]**2)
                det *= fj**2

        jacob.append(jtmp)

        # Mxy
        jtmp = zeros(self.image.shape)
        fac = (2*Mxy/det + (Mxx_y-Mxy_x)*(Myy_x-Mxy_y)/det**2)
        for i,Fi in enumerate(flist):
            if i > 0:
                fi = pars[5+self.ngauss+i-1]
            else:
                fi=1.
            jtmp += Fi*fac/fi**2
        jacob.append(jtmp)

        # Mxx
        jtmp = zeros(self.image.shape)
        fac = (-Myy/det  + 0.5*( (Myy_x - Mxy_y)/det)**2 )
        for i,Fi in enumerate(flist):
            if i > 0:
                fi = pars[5+self.ngauss+i-1]
            else:
                fi=1.
            jtmp += Fi*fac/fi**2
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
                jtmp = Fi*fac/fi**2
                jacob.append(jtmp)

        for i in xrange(len(jacob)):
            jacob[i] = jacob[i].reshape(self.image.size)
        return jacob


    def scale_cov(self, popt, pcov):
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


