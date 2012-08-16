#ifndef NLSolver_H
#define NLSolver_H

// The class NLSolver is designed as a base class.  Therefore, you should
// define a class with your particular function as a derived class of
// NLSolver:
//
// class MyNLFunction : public NLSolver { ... }
//
// You need to define the virtual functions calculateF and calculateJ.
// J = dF/dx is a MxN matrix where M is the number of elements
// in the F vector and N is the number of elements in the x vector.
// This allows you to put any kind of global information that
// might be needed for evaluating the function into the class.
//
// Then:
//
// MyNLSolver nls;
// tmv::Vector<double> x(n);
// tmv::Vector<double> f(m);
// x = [initial guess]
// bool success = nls.solve(x,f);
// [ if success, x is now a solution to the function. ]
// [ f is returned as F(x), regardless of success. ]
//
// Note: for some functions, the calculation of J is 
// partially degenerate with the calculations needed for F.
// Therefore, we do two things to help out.
// First, the function call for calculateJ includes the value for
// f that has previously been calculated for that x.
// Second, we guarantee that every call to calculateJ will be for the 
// same x argument as the previous call to calculateF.  This allows
// the class to potentially store useful auxiliary information
// which might make the J calculation easier.
//
// If m > n, then the solution sought is the 
// least squares solution which minimizes 
// Q = 1/2 Sum |F_i|^2
//
// If you want the covariance matrix of the variables at the 
// solution location, you can do:
//
// tmv::Matrix<double> cov(n,n);
// nls.getCovariance(cov);
//
// This needs to be done _after_ the solve function returns a success.
// 
// Sometimes it is more useful to have the inverse covariance matrix rather
// than the covariance.  It turns out that it is more efficient to 
// calculate the inverse covariance matrix directly here, rather than 
// the covariance matrix and then invert it using a normal inversion algorithm.
// So we provide the option of getting this instead:
//
// tmv::Matrix<double> invCov(n,n);
// nls.getInverseCovariance(invCov);
//
// There are 5 methods implemented here, all of which are based on
// "Methods for Non-Linear Least Squares Problems", 2nd edition,
// April, 2004, K. Madsen, H.B. Nielsen, O. Tingleff,
// Informatics and Mathematical Modelling, Technical University of Denmark.
//
// The Newton method is a simple method which can be very fast for cases
// where the solution is at f=0, but it is also more liable to fail 
// than the below methods.
//
// The Hybrid method is usually the best choice when the number of 
// equations (m) > the number of variables (n).
//
// The Dogleg method is usually best when n == m and you are expecting an
// exact solution.
//
// The LM method is often good in both cases, but sometimes has more trouble 
// converging than the above choices.  For some situations, though, it may
// be faster than these.
//
// The SecantDogleg method may be best when m == n and a direct calculation of 
// J is not possible.
//
// The SecantLM method may be best when m > n and a direct calculation of 
// J is not possible.
//
// To set which method to use for a particular solver objects type:
// solver.useNewton();
// solver.useHybrid();
// solver.useDogleg();
// solver.useLM();
// solver.useSecantDogleg();
// solver.useSecantLM();
//
//
// There are two ways that the algorithms can seccessfully converge:
//
// success if NormInf(f) < ftol
// success if NormInf(grad(1/2 NormSq(f)) < gtol
//
// There are also a number of ways they can fail.  
// Two important ones which you should set appropriately are:
//
// fail if Norm2(delta x) < minstep * (Norm2(x) + epsilon2)
// fail if number of iterations > max_iter
//
// The defaults are:
//
// ftol = 1.e-8
// gtol = 1.e-8
// minstep = 1.e-8
// maxiter = 200
//
// These are set with the methods:
// solver.setFTol(ftol)
// solver.setGTol(gtol
// solver.setTol(ftol,gtol)
// solver.setMinStep(minstep)
// solver.setMaxIter(maxiter)
//
// There are other failure modes for the various algorithms which use
// the above parameters for their tests, so no need to set anything else.
// However, there are a couple more parameters which affect how 
// various algorithms are initialized:
// 
// solver.setTau(tau)
// tau is a parameter used in LM, Hybrid, and SecantLM.  
// If the initial guess is thought to be very good, use tau = 1.e-6. 
// If it's somewhat reasonable, use tau = 1.e-3. (default)
// If it's bad, use tau = 1.
//
// solver.setDelta0(delta0)
// delta0 is a parameter used in Dogleg and SecantDogleg.
// It gives the initial scale size allowed for steps in x.
// The algorithm will increase or decrease this as appropriate.
// The default value is 1.
//
// solver.setOuput(os)
// You can have some information about the solver progress sent to 
// an ostream, os.  e.g. solver.setOutput(std::cout);
// This will print basic information about each iteration.
// If you want much more information about what's going on, 
// you can get verbose output with:
// solver.useVerboseOutput();
//
// solver.useDirectH()
// solver.noUseDirectH()
// This tells the solver whether you have defined the calculateH 
// function, which may make the solver a bit faster.
// This is only used for the Hybrid method.
// If you don't define the calculateH function, then the algorithm
// builds up successive approximations to H with each pass of the 
// calculation.  This is often good enough and is the default behavior.  
// However, if the H matrix happens to be particularly easy to calculate 
// directly, then it can be worth it to do so.
//
// solver.useCholesky()
// solver.noUseCholesky()
// This tells the solver whether you want to start out using Cholesky 
// decomposition for the matrix solvers in LM, Hybrid and SecantLM.
// This is the default, and is usually faster.  The algorithm automatically
// detects if the matrix becomes non-positive-definite, in which case,
// the solver switches over to Bunch-Kauffman decomposition instead.
// A (very) slight improvement in speed could be obtained if you know that
// your matrix is not going to remain positive definite very long,
// in which case setting startwithch=false will use Bunch-Kauffman from
// the beginning.
//
// solver.useSVD()
// solver.noUseSVD()
// This tells the sovler whether you want to use singular value decomposition
// for the division.  The default is not to, since it is slow, and it doesn't
// usually produce much improvement in the results.
// But in some cases, SVD can find a valid step for a singular (or nearly
// singular) jacobian matrix, which moves you closer to the correct solution 
// (and often away from singularity).
// So it may be worth giving it a try if the solver gets stuck at some 
// point.  Call useSVD(), and then rerun solve.
// Also, if the hessian is singular at the solution, then the routines may
// produce a good result, but the covariance matrix will be garbage.  So
// in that case, it may be worth calling useSVD() _after_ solve is done,
// but before calling getCovariance.  

#include <stdexcept>
#include <memory>

// There are too many changes required for this class, so here
// and in NLSolver.cpp, I do a full separate definition for Eigen.
// In part, this is because I didn't want to bother going through and
// making the changes for all of the methods, so I dropped down to 
// just Hybrid and Dogleg, the ones I use most often.
// And second, Eigen isn't as flexible about its division method, so 
// I removed the options of using SVD or Choleskey, and only use LU.

#ifdef USE_TMV

#ifdef MEM_TEST
#define SAVE_MEM_TEST
#undef MEM_TEST
#endif

#include "TMV.h"
#include "TMV_Sym.h"

#ifdef SAVE_MEM_TEST
#define MEM_TEST
#endif

class NLSolver 
{
public :

    NLSolver();
    virtual ~NLSolver() {}

    // This is the basic function that needs to be overridden.
    virtual void calculateF(
        const tmv::Vector<double>& x, tmv::Vector<double>& f) const =0;

    // J(i,j) = df_i/dx_j
    // If you don't overload the J function, then a finite
    // difference calculation will be performed.
    virtual void calculateJ(
        const tmv::Vector<double>& x, const tmv::Vector<double>& f, 
        tmv::Matrix<double>& j) const;

    // Try to solve for F(x) = 0.
    // Returns whether it succeeded.
    // On exit, x is the best solution found.
    // Also, f is returned as the value of F(x) for the best x,
    // regardless of whether the fit succeeded or not.
    virtual bool solve(tmv::Vector<double>& x, tmv::Vector<double>& f) const;

    // Get the covariance matrix of the solution.
    // This only works if solve() returns true.
    // So it should be called after a successful solution is returned.
    virtual void getCovariance(tmv::Matrix<double>& cov) const;

    // You can also get the inverse covariance matrix if you prefer.
    virtual void getInverseCovariance(tmv::Matrix<double>& invcov) const;

    // H(i,j) = d^2 Q / dx_i dx_j
    // where Q = 1/2 Sum_k |f_k|^2
    // H = JT J + Sum_k f_k d^2f_k/(dx_i dx_j)
    //
    // This is only used for the Hybrid method, and if it is not
    // overloaded, then an approximation is calculated on the fly.
    // It's not really important to overload this, but in case 
    // the calculation is very easy, a direct calculation would
    // be faster, so we allow for that possibility.
    virtual void calculateH(
        const tmv::Vector<double>& x, const tmv::Vector<double>& f, 
        const tmv::Matrix<double>& j, tmv::SymMatrix<double>& h) const;

    // This tests whether the direct calculation of J matches
    // the numerical approximation of J calculated from finite differences.
    // It is useful as a test that your analytic formula was coded correctly.
    //
    // If you pass it an ostream (e.g. os = &cout), then debugging
    // info will be sent to that stream.
    //
    // The second optional parameter, relerr, is for functions which 
    // are merely good approximations of the correct J, rather than
    // exact.  Normally, the routine tests whether the J function
    // calculates the same jacobian as a numerical approximation to 
    // within the numerical precision possible for doubles.
    // If J is only approximate, you can set relerr as the relative
    // error in J to be tested for.
    // If relerr == 0 then sqrt(numeric_limits<double>::epsilon())
    // = 1.56e-7 is used.  This is the default.
    //
    // Parameters are x, f, os, relerr.
    virtual bool testJ(
        const tmv::Vector<double>& , tmv::Vector<double>& ,
        std::ostream* os=0, double relerr=0.) const;

    // H(i,j) = d^2 Q / dx_i dx_j
    // where Q = 1/2 Sum_k |f_k|^2
    // H = JT J + Sum_k f_k d^2f_k/(dx_i dx_j)
    virtual void calculateNumericH(
        const tmv::Vector<double>& x, const tmv::Vector<double>& f, 
        tmv::SymMatrix<double>& h) const;

    virtual void useNewton() { _method = NEWTON; }
    virtual void useHybrid() { _method = HYBRID; }
    virtual void useLM() { _method = LM; }
    virtual void useDogleg() { _method = DOGLEG; }
    virtual void useSecantLM() { _method = SECANT_LM; }
    virtual void useSecantDogleg() { _method = SECANT_DOGLEG; }

    virtual void setFTol(double ftol) { _ftol = ftol; }
    virtual void setGTol(double gtol) { _gtol = gtol; }
    virtual void setTol(double ftol, double gtol) 
    { _ftol = ftol; _gtol = gtol; }

    virtual void setMinStep(double minstep) { _minstep = minstep; }
    virtual void setMaxIter(int maxiter) { _maxiter = maxiter; }
    virtual void setTau(double tau) { _tau = tau; }
    virtual void setDelta0(double delta0) { _delta0 = delta0; }

    virtual double getFTol() { return _ftol; }
    virtual double getGTol() { return _gtol; }
    virtual double getMinStep() { return _minstep; }
    virtual int getMaxIter() { return _maxiter; }
    virtual double getTau() { return _tau; }
    virtual double getDelta0() { return _delta0; }

    virtual void setOutput(std::ostream& os) { _nlout = &os; }
    virtual void useVerboseOutput() { _verbose = 1; }
    virtual void useExtraVerboseOutput() { _verbose = 2; }
    virtual void noUseVerboseOutput() { _verbose = 0; }

    virtual void useDirectH() { _directh = true; }
    virtual void useSVD() { _usesvd = true; }
    virtual void useCholesky() { _usech = true; }
    virtual void noUseDirectH() { _directh = false; }
    virtual void noUseSVD() { _usesvd = false; }
    virtual void noUseCholesky() { _usech = false; }

protected:

    enum Method { NEWTON, HYBRID, DOGLEG, LM, SECANT_DOGLEG, SECANT_LM };

    Method _method;
    double _ftol;
    double _gtol;
    double _minstep;
    int _maxiter;
    double _tau;
    double _delta0;
    std::ostream* _nlout;
    int _verbose;
    bool _directh;
    bool _usech;
    bool _usesvd;

    bool solveNewton(
        tmv::Vector<double>& x, tmv::Vector<double>& f) const;
    bool solveLM(
        tmv::Vector<double>& x, tmv::Vector<double>& f) const;
    bool solveDogleg(
        tmv::Vector<double>& x, tmv::Vector<double>& f) const;
    bool solveHybrid(
        tmv::Vector<double>& x, tmv::Vector<double>& f) const;
    bool solveSecantLM(
        tmv::Vector<double>& x, tmv::Vector<double>& f) const;
    bool solveSecantDogleg(
        tmv::Vector<double>& x, tmv::Vector<double>& f) const;

    mutable std::auto_ptr<tmv::Matrix<double> > _pJ;

};

#else

#include "MyMatrix.h"

class NLSolver 
{
public :

    NLSolver();
    virtual ~NLSolver() {}

    // This is the basic function that needs to be overridden.
    virtual void calculateF(const DVector& x, DVector& f) const =0;

    // J(i,j) = df_i/dx_j
    // If you don't overload the J function, then a finite
    // difference calculation will be performed.
    virtual void calculateJ(
        const DVector& x, const DVector& f, DMatrix& j) const;

    // Try to solve for F(x) = 0.
    // Returns whether it succeeded.
    // On exit, x is the best solution found.
    // Also, f is returned as the value of F(x) for the best x,
    // regardless of whether the fit succeeded or not.
    virtual bool solve(DVector& x, DVector& f) const;

    // Get the covariance matrix of the solution.
    // This only works if solve() returns true.
    // So it should be called after a successful solution is returned.
    virtual void getCovariance(DMatrix& cov) const;

    // You can also get the inverse covariance matrix if you prefer.
    virtual void getInverseCovariance(DMatrix& invcov) const;

    virtual bool testJ(const DVector& , DVector& ,
                       std::ostream* os=0, double relerr=0.) const;

    virtual void useHybrid() { _method = HYBRID; }
    virtual void useDogleg() { _method = DOGLEG; }

    virtual void setFTol(double ftol) { _ftol = ftol; }
    virtual void setGTol(double gtol) { _gtol = gtol; }
    virtual void setTol(double ftol, double gtol) 
    { _ftol = ftol; _gtol = gtol; }

    virtual void setMinStep(double minstep) { _minstep = minstep; }
    virtual void setMaxIter(int maxiter) { _maxiter = maxiter; }
    virtual void setTau(double tau) { _tau = tau; }
    virtual void setDelta0(double delta0) { _delta0 = delta0; }

    virtual double getFTol() { return _ftol; }
    virtual double getGTol() { return _gtol; }
    virtual double getMinStep() { return _minstep; }
    virtual int getMaxIter() { return _maxiter; }
    virtual double getTau() { return _tau; }
    virtual double getDelta0() { return _delta0; }

    virtual void setOutput(std::ostream& os) { _nlout = &os; }
    virtual void useVerboseOutput() { _verbose = 1; }
    virtual void useExtraVerboseOutput() { _verbose = 2; }
    virtual void noUseVerboseOutput() { _verbose = 0; }

    virtual void useDirectH() {}
    virtual void useSVD() { _usesvd = true; }
    virtual void useCholesky() {}
    virtual void noUseDirectH() {}
    virtual void noUseSVD() { _usesvd = false; }
    virtual void noUseCholesky() {}

protected:

    enum Method { HYBRID, DOGLEG };

    Method _method;
    double _ftol;
    double _gtol;
    double _minstep;
    int _maxiter;
    double _tau;
    double _delta0;
    std::ostream* _nlout;
    int _verbose;
    bool _directh;
    bool _usech;
    bool _usesvd;

    bool solveDogleg(DVector& x, DVector& f) const;
    bool solveHybrid(DVector& x, DVector& f) const;

    mutable std::auto_ptr<DMatrix> _pJ;

};

#endif

#endif
