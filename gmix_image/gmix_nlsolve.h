#ifndef _gmix_nlsolve_h_guard
#define _gmix_nlsolve_h_guard

#include <iostream>
#include <vector>
#include <sstream>
#include <math.h>
#include "nlsolver.h"
#include "defs.h"

#include <Python.h>
#include "numpy/arrayobject.h" 

// simple image wrapper
struct simple_image {
    size_t size;
    size_t nrows;
    size_t ncols;

    double **rows;
};
struct gauss {
    double p;
    double row;
    double col;
    double irr;
    double irc;
    double icc;
    double det;
};


#define SIMPLE_IM_GET(im, row, col)                  \
    ( *((im)->rows[(row)] +  (col)) )

#define SIMPLE_IM_SET(im, row, col, val)                  \
    ( *((im)->rows[(row)] + (col)) = (val) )

class GMixCoellipSolver : public NLSolver { 

    public:
        GMixCoellipSolver(PyObject* image_obj, 
                          PyObject* guess_obj, 
                          double skysig,
                          int maxiter, 
                          PyObject* psf_obj, 
                          int verbose) throw (const char*) {
            this->image=NULL;
            this->image_obj=NULL;
            this->skysig=skysig;

            import_array();
            this->associate_image(image_obj);
            this->copy_guess(guess_obj);

            if (psf_obj != Py_None) {
                this->copy_psf(psf_obj);
            }

            tmv::Vector<double> tpars;
            tmv::Vector<double> tydiff;

            tpars.resize(this->guess.size());
            tpars = this->guess;
            tydiff.resize(this->image->size);

            this->useHybrid();

            this->setFTol(1.e-8);
            this->setGTol(1.e-8);
            this->setMinStep(1.e-8);
            this->setMaxIter(maxiter);
            //this->setTau(1.);

            if (verbose) {
                this->setOutput(std::cerr);
            }
            this->success = this->solve(tpars, tydiff);

            this->pars.resize(tpars.size());
            this->ydiff.resize(tydiff.size());
            this->pars = tpars;
            this->ydiff = tydiff;

            this->set_chi2per();

        }
        ~GMixCoellipSolver(){
            if (this->image) {
                free(this->image->rows);
                free(this->image);
            }
            this->image=NULL;

            Py_XDECREF(this->image_obj);
        }

        bool get_success() const {
            return this->success;
        }
        int get_flags() const {
            int flags=0;
            if (!this->success) {
                flags=1;
            }
            return flags;
        }
        double get_chi2per() const {
            return this->chi2per;
        }
        PyObject *get_pars() const {
            npy_intp dims[1];
            int fortran=0;
            dims[0] = this->pars.size();
            PyObject* apars = PyArray_ZEROS(1, dims, NPY_DOUBLE, fortran);
            double *data = (double*) PyArray_DATA(apars);
            for (int i=0; i<dims[0]; i++) {
                data[i] = this->pars(i);
            }

            return apars;
        }
        PyObject *get_perr() const {
            npy_intp dims[1];
            int fortran=0;

            int npars=this->pars.size();
            dims[0] = npars;

            PyObject* perr_obj = PyArray_ZEROS(1, dims, NPY_DOUBLE, fortran);

            if (this->success) {

                tmv::Matrix<double> cov(npars,npars);
                this->getCovariance(cov);

                for (int i=0; i<dims[0]; i++) {
                    double *ptr = (double*) PyArray_GETPTR1(perr_obj, i);
                    *ptr = sqrt(cov(i,i));
                }
            } else {
                for (int i=0; i<dims[0]; i++) {
                    double *ptr = (double*) PyArray_GETPTR1(perr_obj, i);
                    *ptr = .9999e10;
                }
            }

            return perr_obj;

        }
        PyObject *get_pcov() const {
            npy_intp dims[2];
            int fortran=0;

            int npars=this->pars.size();
            dims[0] = npars;
            dims[1] = npars;

            PyObject* acov = PyArray_ZEROS(2, dims, NPY_DOUBLE, fortran);

            double *ptr = NULL;
            if (this->success) {
                tmv::Matrix<double> cov(npars,npars);
                this->getCovariance(cov);
                for (int i=0; i<dims[0]; i++) {
                    for (int j=0; j<dims[0]; j++) {
                        ptr = (double*) PyArray_GETPTR2(acov, i, j);
                        *ptr = cov(i,j);
                    }
                }
            } else {
                for (int i=0; i<dims[0]; i++) {
                    for (int j=0; j<dims[0]; j++) {
                        ptr = (double*) PyArray_GETPTR2(acov, i, j);
                        *ptr = .999e10;
                    }
                }
            }

            return acov;
        }

        PyObject *get_model() const {
            npy_intp dims[2];
            int fortran=0;

            dims[0] = this->image->nrows;
            dims[1] = this->image->nrows;
            PyObject* model = PyArray_ZEROS(2, dims, NPY_DOUBLE, fortran);

            double *data=(double*) PyArray_DATA(model);
            this->_render_model(this->pars, data);

            return model;
        }
        double get_s2n() const {
            tmv::Vector<double> model;
            npy_intp nel=this->image->nrows*this->image->ncols;
            model.resize(nel);
            this->_render_model(this->pars, &model(0));

            double sum=0., s2n=0.;

            for (npy_intp i=0; i<model.size(); i++) {
                sum += model(i);
            }

            s2n = sum/(this->skysig*sqrt(model.size()));

            return s2n;
        }



        size_t get_nrows() {
            return (size_t) this->image->nrows;
        }
        size_t get_ncols() {
            return (size_t) this->image->ncols;
        }
        double get_val(size_t row, size_t col) throw (const char*) {
            if ( (row > (image->nrows-1)) || (col > (image->ncols-1)) ) {
                throw "out of bounds";
            }
            return SIMPLE_IM_GET(image, row, col);
        }
    private:

        void calculateF_old(const tmv::Vector<double>& pars, 
                        tmv::Vector<double>& ydiff) const {

            size_t nrows=this->image->nrows;
            size_t ncols=this->image->ncols;

            struct gauss *gauss=NULL;
            const struct gauss *pgauss=NULL;
            double u=0, v=0, uv=0, u2=0, v2=0;
            double chi2=0, b=0;
            size_t col=0, row=0;
            double irr=0, irc=0, icc=0, det=0, psum=0;

            double val=0, tval=0, image_val=0;

            int ngauss = (pars.size()-4)/2;
            std::vector<struct gauss> gvec(ngauss);
            this->set_gvec(pars, gvec);


            size_t ii=0;
            for (row=0; row<nrows; row++) {
                for (col=0; col<ncols; col++) {

                    val=0;
                    for (int i=0; i<ngauss; i++) {
                        gauss = &gvec[i];

                        if (gauss->det <= 0) {
                            DBG wlog("found det: %.16g\n",gauss->det);
                            tval = 1.e20;
                        } else {
                            u = row-gauss->row;
                            v = col-gauss->col;

                            u2 = u*u; v2 = v*v; uv = u*v;

                            tval=0.;
                            if (this->psf.size() > 0) {
                                psum=0;
                                for (size_t j=0; j<this->psf.size(); j++) {
                                    pgauss=&this->psf[j];
                                    irr = gauss->irr + pgauss->irr;
                                    irc = gauss->irc + pgauss->irc;
                                    icc = gauss->icc + pgauss->icc;
                                    det = irr*icc - irc*irc;
                                    if (det <= 0) {
                                        DBG wlog("found convolved det: %.16g\n", det);
                                        tval = 1.e20;
                                        psum=1.;
                                        break;
                                    } else {
                                        chi2=icc*u2 + irr*v2 - 2.0*irc*uv;
                                        chi2 /= det;

                                        b = M_TWO_PI*sqrt(det);
                                        tval += pgauss->p*exp( -0.5*chi2 )/b;
                                        psum += pgauss->p;
                                    }
                                }
                                // psf always normalized to unity
                                tval *= gauss->p/psum;

                            } else {
                                chi2=gauss->icc*u2 + gauss->irr*v2 - 2.0*gauss->irc*uv;
                                chi2 /= gauss->det;
                                b = M_TWO_PI*sqrt(gauss->det);
                                tval = gauss->p*exp( -0.5*chi2 )/b;
                            }
                        }
                        val += tval;
                    } // gvec

                    image_val = SIMPLE_IM_GET(image, row, col);
                    //ydiff(ii) = val-tval;
                    ydiff(ii) = (val-image_val)/this->skysig;
                    ii++;

                } // cols
            } // rows

        }


        void calculateF(const tmv::Vector<double>& pars, 
                        tmv::Vector<double>& ydiff) const {

            size_t nrows=this->image->nrows;
            size_t ncols=this->image->ncols;

            this->_render_model(pars, &ydiff(0));

            double *ptr=&ydiff(0);
            for (size_t row=0; row<nrows; row++) {
                for (size_t col=0; col<ncols; col++) {

                    double image_val = SIMPLE_IM_GET(image, row, col);
                    double model_val = *ptr;

                    *ptr = (model_val-image_val)/this->skysig;
                    ptr++;
                }
            }
        }


        void _render_model(const tmv::Vector<double>& pars, 
                           double *model) const {

            size_t nrows=this->image->nrows;
            size_t ncols=this->image->ncols;

            struct gauss *gauss=NULL;
            const struct gauss *pgauss=NULL;
            double u=0, v=0, uv=0, u2=0, v2=0;
            double chi2=0, b=0;
            size_t col=0, row=0;
            double irr=0, irc=0, icc=0, det=0, psum=0;

            double val=0, tval=0;

            int ngauss = (pars.size()-4)/2;
            std::vector<struct gauss> gvec(ngauss);
            this->set_gvec(pars, gvec);


            size_t ii=0;
            for (row=0; row<nrows; row++) {
                for (col=0; col<ncols; col++) {

                    val=0;
                    for (int i=0; i<ngauss; i++) {
                        gauss = &gvec[i];

                        if (gauss->det <= 0) {
                            DBG wlog("found det: %.16g\n",gauss->det);
                            tval = 1.e20;
                        } else {
                            u = row-gauss->row;
                            v = col-gauss->col;

                            u2 = u*u; v2 = v*v; uv = u*v;

                            tval=0.;
                            if (this->psf.size() > 0) {
                                psum=0;
                                for (size_t j=0; j<this->psf.size(); j++) {
                                    pgauss=&this->psf[j];
                                    irr = gauss->irr + pgauss->irr;
                                    irc = gauss->irc + pgauss->irc;
                                    icc = gauss->icc + pgauss->icc;
                                    det = irr*icc - irc*irc;
                                    if (det <= 0) {
                                        DBG wlog("found convolved det: %.16g\n", det);
                                        tval = 1.e20;
                                        psum=1.;
                                        break;
                                    } else {
                                        chi2=icc*u2 + irr*v2 - 2.0*irc*uv;
                                        chi2 /= det;

                                        b = M_TWO_PI*sqrt(det);
                                        tval += pgauss->p*exp( -0.5*chi2 )/b;
                                        psum += pgauss->p;
                                    }
                                }
                                // psf always normalized to unity
                                tval *= gauss->p/psum;

                            } else {
                                chi2=gauss->icc*u2 + gauss->irr*v2 - 2.0*gauss->irc*uv;
                                chi2 /= gauss->det;
                                b = M_TWO_PI*sqrt(gauss->det);
                                tval = gauss->p*exp( -0.5*chi2 )/b;
                            }
                        }
                        val += tval;
                    } // gvec

                    model[ii] = val;
                    ii++;

                } // cols
            } // rows

        }

        void set_gvec(const tmv::Vector<double>& pars, std::vector<struct gauss>& gvec) const
        {
            npy_intp ngauss=0;
            double row=0, col=0, e1=0, e2=0, Tmax=0, Ti=0, pi=0, Tfrac=0;
            struct gauss *gauss=NULL;

            int i=0;

            ngauss = (pars.size()-4)/2;

            row=pars(0);
            col=pars(1);
            e1 = pars(2);
            e2 = pars(3);
            Tmax = pars(4);

            for (i=0; i<ngauss; i++) {
                gauss = &gvec[i];

                if (i==0) {
                    Ti = Tmax;
                } else {
                    Tfrac = pars(4+i);
                    Ti = Tmax*Tfrac;
                }

                pi = pars(4+ngauss+i);

                gauss->p = pi;
                gauss->row = row;
                gauss->col = col;

                gauss->irr = (Ti/2.)*(1-e1);
                gauss->irc = (Ti/2.)*e2;
                gauss->icc = (Ti/2.)*(1+e1);
                gauss->det = gauss->irr*gauss->icc - gauss->irc*gauss->irc;
            }
        }


        void copy_guess(PyObject* guess_obj) throw (const char*) {
            check_numpy_arr(guess_obj);
            npy_intp npars = PyArray_SIZE(guess_obj);
            this->guess.resize(npars);

            double *data = (double*) PyArray_DATA(guess_obj);
            for (npy_intp i=0; i<npars; i++) {
                this->guess(i) = data[i];
            }
        }

        // using full because we might want to allow psf
        // to be co-centric but not co-elliptical
        void copy_psf(PyObject* psf_obj) throw (const char*) {
            npy_intp sz=0;
            npy_intp ngauss=0;
            double *pars=NULL;
            struct gauss *gauss=NULL;
            int i=0, beg=0;


            check_numpy_arr(psf_obj);

            pars = (double*) PyArray_DATA(psf_obj);
            sz = PyArray_SIZE(psf_obj);
            if ((sz % 6) != 0) {
                throw "psf must be a full gaussian mixture";
            }
            ngauss = sz/6;

            this->psf.resize(ngauss);

            for (i=0; i<ngauss; i++) {
                beg = i*6;

                gauss = &this->psf[i];

                gauss->p   = pars[beg+0];
                gauss->row = pars[beg+1];
                gauss->col = pars[beg+2];
                gauss->irr = pars[beg+3];
                gauss->irc = pars[beg+4];
                gauss->icc = pars[beg+5];

                gauss->det = gauss->irr*gauss->icc - gauss->irc*gauss->irc;
            }
        }

        // no copy made
        void associate_image(PyObject* image_obj) throw (const char*)
        {

            this->check_numpy_image(image_obj);
            npy_intp* dims = PyArray_DIMS(image_obj);
            npy_intp nrows=dims[0];
            npy_intp ncols=dims[1];

            check_numpy_image(image_obj);

            this->image = (struct simple_image*) calloc(1, sizeof(struct simple_image));
            this->image->nrows = nrows;
            this->image->ncols = ncols;
            this->image->size = nrows*ncols;
            this->image->rows = (double**) calloc(nrows,sizeof(double *));

            double* data = (double*) PyArray_DATA((PyArrayObject*)image_obj);

            this->image->rows[0] = data;
            for(npy_intp i = 1; i < nrows; i++) {
                this->image->rows[i] = this->image->rows[i-1] + ncols;
            }

            this->image_obj = image_obj;
            Py_XINCREF(this->image_obj);
        }

        void check_numpy_image(PyObject* obj) throw (const char*) {
            if (!PyArray_Check(obj) 
                    || NPY_DOUBLE != PyArray_TYPE(obj)
                    || 2 != PyArray_NDIM(obj)) {
                throw "image input must be a 2D double PyArrayObject";
            }
            
        }
        void check_numpy_arr(PyObject* obj) throw (const char*) {
            if (!PyArray_Check(obj) 
                    || NPY_DOUBLE != PyArray_TYPE(obj)
                    || 1 != PyArray_NDIM(obj)) {
                throw "array must be a 1D double PyArrayObject";
            }
        }

        void set_chi2per() {
            chi2per=0.;
            for (ssize_t i=0; i < ydiff.size(); i++) {
                chi2per += ydiff(i)*ydiff(i);
            }

            chi2per /= (ydiff.size()-pars.size());
        }


        long image_size;
        struct simple_image *image;
        tmv::Vector<double> guess;
        tmv::Vector<double> pars;

        tmv::Vector<double> ydiff;
        std::vector<gauss> gvec;

        // we always represent the psf with a full
        // gaussian mixture
        std::vector<gauss> psf;

        int ngauss;
        PyObject *image_obj;

        double skysig;

        double chi2per;

        bool success;
};

#endif
