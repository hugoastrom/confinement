#include "GeneralHIPBasis.h"
#include "lobatto.h"

namespace helfem {
  namespace polynomial_basis {

    static void print_test(arma::mat f, const std::string & msg, double thr=1e-9) {
      // Set small elements to zero
      f.elem(arma::find(arma::abs(f)<thr)).zeros();
      // Print out with message
      f.print(msg);
    }

    // Dummy length
    static const double dummy_length = 1.0;

    GeneralHIPBasis::GeneralHIPBasis(const arma::vec & x, int id_, int nder_) {
      id=id_;
      nder=nder_;

      // Number of overlapping functions
      noverlap=nder+1;
      // The number of functions we need is
      int nfuncs=(nder+1)*x.n_elem;
      // All functions are enabled
      enabled=arma::linspace<arma::uvec>(0,nfuncs-1,nfuncs);
      // Number of nodes is
      nnodes=x.n_elem;

      // Construct the necessary LIP basis
      arma::vec xlip, wlip;
      ::lobatto_compute(nfuncs,xlip,wlip);
      lip = polynomial_basis::LIPBasis(xlip);

      printf("Setting up %i-node %i:th order HIPs from a %i-node LIP basis.\n", nnodes, nder, nfuncs);

      // Evaluate the values of the LIPs and their derivatives
      arma::mat fval = lip.eval_f(x, dummy_length).t();
      arma::mat dfval;
      if(nder>0)
        dfval = lip.eval_df(x, dummy_length).t();
      arma::mat d2fval;
      if(nder>1)
        d2fval = lip.eval_d2f(x, dummy_length).t();
      arma::mat d3fval;
      if(nder>2)
        d3fval = lip.eval_d3f(x, dummy_length).t();
      arma::mat d4fval;
      if(nder>3)
        d4fval = lip.eval_d4f(x, dummy_length).t();
      arma::mat d5fval;
      if(nder>4)
        d5fval = lip.eval_d5f(x, dummy_length).t();
      if(nder>5)
        throw std::logic_error("General HIP functions have not been implemented beyond 5th order!\n");

      /*
       * Construct the equation for the transformation matrix.
       *
       *           T X = 1
       *
       * where T is the transformation matrix and X are the values of
       * LIPs and their derivatives at the nodes, so T = X^-1.
       *
       * We can easily build the matrix in our target basis by looping
       * over the nodes. The first nder+1 functions are the value of
       * the LIP and its nder derivaties at the first node. The next
       * nder+1 functions are the values at the second node. Etc.
       */
      arma::mat X(nfuncs, nfuncs, arma::fill::zeros);
      for(int ifun=0;ifun < nnodes;ifun++) {
        X.col((nder+1)*ifun) = fval.col(ifun);
        if(nder>0)
          X.col((nder+1)*ifun+1) = dfval.col(ifun);
        if(nder>1)
          X.col((nder+1)*ifun+2) = d2fval.col(ifun);
        if(nder>2)
          X.col((nder+1)*ifun+3) = d3fval.col(ifun);
        if(nder>3)
          X.col((nder+1)*ifun+4) = d4fval.col(ifun);
        if(nder>4)
          X.col((nder+1)*ifun+5) = d5fval.col(ifun);
      }

      // X has our target functions in its columns so X^-1 has the
      // target in its rows; if we take the transpose then we get the
      // target functions in columns in T
      printf("Transformation matrix reciprocal condition number %e\n",arma::rcond(X));
      T = arma::inv(X.t());

      // Test the conditions
      print_test(eval_f(x, dummy_length),  "Function   value at nodes");
      if(nder>0)
        print_test(eval_df(x, dummy_length), "Derivative value at nodes");
      if(nder>1)
        print_test(eval_d2f(x, dummy_length), "Second derivative value at nodes");
      if(nder>2)
        print_test(eval_d3f(x, dummy_length), "Third derivative value at nodes");
      if(nder>3)
        print_test(eval_d4f(x, dummy_length), "Fourth derivative value at nodes");
      if(nder>4)
        print_test(eval_d5f(x, dummy_length), "Fifth derivative value at nodes");
    }

    GeneralHIPBasis::~GeneralHIPBasis() {
    }

    GeneralHIPBasis * GeneralHIPBasis::copy() const {
      return new GeneralHIPBasis(*this);
    }

    void GeneralHIPBasis::scale_derivatives(arma::mat & f, double element_length) const {
      for(int ifun=0; ifun<nnodes; ifun++) {
        for(int ider=1;ider<nder;ider++)
          f.col((nder+1)*ifun+ider) *= std::pow(element_length,ider);
      }
    }

    void GeneralHIPBasis::eval_prim_f(const arma::vec & x, arma::mat & f, double element_length) const {
      // Evaluate the primitive LIP polynomials
      f=lip.eval_f(x, dummy_length)*T;
      // and scale the derivative functions
      scale_derivatives(f, element_length);
    }

    void GeneralHIPBasis::eval_prim_df(const arma::vec & x, arma::mat & df, double element_length) const {
      // Evaluate the primitive LIP polynomials
      df=lip.eval_df(x, dummy_length)*T;
      // and scale the derivative functions
      scale_derivatives(df, element_length);
    }

    void GeneralHIPBasis::eval_prim_d2f(const arma::vec & x, arma::mat & d2f, double element_length) const {
      // Evaluate the primitive LIP polynomials
      d2f=lip.eval_d2f(x, dummy_length)*T;
      // and scale the derivative functions
      scale_derivatives(d2f, element_length);
    }

    void GeneralHIPBasis::eval_prim_d3f(const arma::vec & x, arma::mat & d3f, double element_length) const {
      // Evaluate the primitive LIP polynomials
      d3f=lip.eval_d3f(x, dummy_length)*T;
      // and scale the derivative functions
      scale_derivatives(d3f, element_length);
    }

    void GeneralHIPBasis::eval_prim_d4f(const arma::vec & x, arma::mat & d4f, double element_length) const {
      // Evaluate the primitive LIP polynomials
      d4f=lip.eval_d4f(x, dummy_length)*T;
      // and scale the derivative functions
      scale_derivatives(d4f, element_length);
    }

    void GeneralHIPBasis::eval_prim_d5f(const arma::vec & x, arma::mat & d5f, double element_length) const {
      // Evaluate the primitive LIP polynomials
      d5f=lip.eval_d5f(x, dummy_length)*T;
      // and scale the derivative functions
      scale_derivatives(d5f, element_length);
    }

    void GeneralHIPBasis::drop_first(bool func, bool deriv) {
      // Subset of functions in the first
      int nfuncs=nder+1;
      arma::uvec same_funcs(enabled.subvec(nfuncs,enabled.n_elem-1));
      arma::uvec first_funcs(enabled.subvec(0, nfuncs-1));

      // Number of new functions
      size_t nnew = same_funcs.n_elem;
      if(!func) nnew++;
      if(!deriv) nnew+=nder;
      arma::uvec new_funcs(nnew);

      int idx = 0;
      if(!func)
        new_funcs(idx++) = first_funcs(0);
      if(!deriv)
        for(int ider=0;ider<nder;ider++)
          new_funcs(idx++) = first_funcs(1+ider);
      new_funcs.subvec(idx,new_funcs.n_elem-1) = same_funcs;

      enabled = new_funcs;
    }

    void GeneralHIPBasis::drop_last(bool func, bool deriv) {
      // Subset of functions in the last element
      int nfuncs=nder+1;
      arma::uvec same_funcs(enabled.subvec(0,enabled.n_elem-nfuncs-1));
      arma::uvec last_funcs(enabled.subvec(enabled.n_elem-nfuncs, enabled.n_elem-1));

      // Number of new functions
      size_t nnew = same_funcs.n_elem;
      if(!func) nnew++;
      if(!deriv) nnew+=nder;
      arma::uvec new_funcs(nnew);
      new_funcs.subvec(0,same_funcs.n_elem-1) = same_funcs;

      int idx = same_funcs.n_elem;
      if(!func)
        new_funcs(idx++) = last_funcs(0);
      if(!deriv)
        for(int ider=0;ider<nder;ider++)
          new_funcs(idx++) = last_funcs(1+ider);

      enabled = new_funcs;
    }
  }
}
