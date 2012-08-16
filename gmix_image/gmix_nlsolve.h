#ifndef _gmix_nlsolve_h_guard
#define _gmix_nlsolve_h_guard

#include <iostream>
#include <vector>
#include <sstream>
#include <math.h>
#include "NLSolver.h"
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
        GMixCoellipSolver(PyObject* image_obj, PyObject* guess_obj, int maxiter) throw (const char*) {
            this->image=NULL;
            this->image_obj=NULL;

            import_array();
            this->associate_image(image_obj);
            this->copy_guess(guess_obj);


            tmv::Vector<double> tpars;
            tmv::Vector<double> tydiff;

            tpars.resize(this->guess.size());
            tpars = this->guess;
            tydiff.resize(this->image->size);

            this->setMaxIter(maxiter);
            this->useHybrid();
            this->success = this->solve(tpars, tydiff);

            this->pars.resize(tpars.size());
            this->ydiff.resize(tydiff.size());
            this->pars = tpars;
            this->ydiff = tydiff;

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
        PyObject *get_cov() const {
            npy_intp dims[2];
            int fortran=0;

            int npars=this->pars.size();
            dims[0] = npars;
            dims[1] = npars;

            PyObject* acov = PyArray_ZEROS(2, dims, NPY_DOUBLE, fortran);

            tmv::Matrix<double> cov(npars,npars);
            this->getCovariance(cov);

            double *ptr = NULL;
            for (int i=0; i<dims[0]; i++) {
                for (int j=0; j<dims[0]; j++) {
                    ptr = (double*) PyArray_GETPTR2(acov, i, j);
                    *ptr = cov(i,j);
                }
            }

            return acov;
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
        /* calculate model-image, no psf yet */
        void calculateF(const tmv::Vector<double>& pars, 
                        tmv::Vector<double>& ydiff) const {

            size_t nrows=this->image->nrows;
            size_t ncols=this->image->ncols;

            struct gauss *gauss=NULL;
            //struct gauss *pgauss=NULL;
            double u=0, v=0, uv=0, u2=0, v2=0;
            double chi2=0, b=0;
            size_t col=0, row=0;
            //size_t j=0;
            //double irr=0, irc=0, icc=0, det=0, psum=0;

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

                            chi2=gauss->icc*u2 + gauss->irr*v2 - 2.0*gauss->irc*uv;
                            chi2 /= gauss->det;
                            b = M_TWO_PI*sqrt(gauss->det);
                            tval = gauss->p*exp( -0.5*chi2 )/b;
                        }
                        val += tval;
                    } // gvec

                    tval = SIMPLE_IM_GET(image, row, col);
                    ydiff(ii) = val-tval;
                    //ydiff(ii) = (val-tval)/1.e-5;
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



        long image_size;
        struct simple_image *image;
        tmv::Vector<double> guess;
        tmv::Vector<double> pars;
        tmv::Vector<double> ydiff;
        std::vector<gauss> gvec;

        int ngauss;
        PyObject *image_obj;

        bool success;
};

#endif
