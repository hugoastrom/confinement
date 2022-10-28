/*
 *                This source code is part of
 *
 *                          HelFEM
 *                             -
 * Finite element methods for electronic structure calculations on small systems
 *
 * Written by Susi Lehtola, 2018-
 * Copyright (c) 2018- Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "basis.h"
#include "solver.h"
#include "../general/dftfuncs.h"
#include "../general/scf_helpers.h"
#include "../general/diis.h"

// Shell types
static const char shtype[]="spdfgh";

namespace helfem {
  namespace sadatom {
    namespace solver {

      bool operator<(const shell_occupation_t & lh, const shell_occupation_t & rh) {
        return lh.E < rh.E;
      }

      arma::sword OrbitalChannel::ShellCapacity(arma::sword l) const {
        return restr ? (4*l+2) : (2*l+1);
      }

      OrbitalChannel::OrbitalChannel() {
      }

      OrbitalChannel::OrbitalChannel(bool restr_) : restr(restr_) {
        lmax=-1;
      }

      OrbitalChannel::~OrbitalChannel() {
      }

      bool OrbitalChannel::Restricted() const {
        return restr;
      }

      void OrbitalChannel::SetRestricted(bool restr_) {
        restr=restr_;
      }

      bool OrbitalChannel::OrbitalsInitialized() const {
        if(!C.n_elem)
          return false;
        return true;
      }

      bool OrbitalChannel::OccupationsInitialized() const {
        return Nel() != 0;
      }

      int OrbitalChannel::Lmax() const {
        return lmax;
      }

      arma::cube OrbitalChannel::Coeffs() const {
        return C;
      }

      void OrbitalChannel::SetLmax(int lmax_) {
        lmax=lmax_;
      }

      arma::sword OrbitalChannel::Nel() const {
        return arma::sum(occs);
      }

      arma::ivec OrbitalChannel::Occs() const {
        return occs;
      }

      void OrbitalChannel::SetOccs(const arma::ivec & occs_) {
        occs=occs_;
      }

      std::vector<shell_occupation_t> OrbitalChannel::GetOccupied() const {
        // Form list of shells
        std::vector<shell_occupation_t> occlist;
        for(size_t l=0;l<E.n_cols;l++) {
          // Number of electrons to put in
          arma::sword numl = occs(l);
          for(size_t io=0;io<E.n_rows;io++) {
            arma::sword nocc = std::min(ShellCapacity(l), numl);
            numl-=nocc;
            if(nocc == 0)
              break;

            shell_occupation_t sh;
            sh.n = l+io+1;
            sh.l = l;
            sh.E = E(io,l);
            sh.nocc = nocc;
            occlist.push_back(sh);
          }
        }
        std::sort(occlist.begin(),occlist.end());
        return occlist;
      }

      arma::vec OrbitalChannel::GetGap() const {
        arma::vec gap(E.n_cols);
        for(size_t l=0;l<E.n_cols;l++) {
          // Number of electrons to put in
          arma::sword numl = occs(l);
          for(size_t io=0;io<E.n_rows;io++) {
            arma::sword nocc = std::min(ShellCapacity(l), numl);
            numl-=nocc;
            if(nocc == 0) {
              if(io==0)
                // Gap is just orbital energy
                gap(l)=E(io,l);
              else
                // Gap is orbital energy difference
                gap(l)=E(io,l)-E(io-1,l);
              break;
            }
          }
        }

        return gap;
      }

      std::string OrbitalChannel::Characterize() const {
        std::vector<shell_occupation_t> occlist(GetOccupied());
        std::ostringstream oss;
        for(size_t i=0;i<occlist.size();i++) {
          if(i)
            oss << " ";
          oss << occlist[i].n << shtype[occlist[i].l] << "^{" << occlist[i].nocc << "}";
        }

        return oss.str();
      }

      void OrbitalChannel::Print(const basis::TwoDBasis & basis) const {
        std::vector<shell_occupation_t> occlist(GetOccupied());
        arma::vec r(basis.radii());

        // Get r matrices
        std::vector< std::pair<int, arma::mat> > rmat(basis.Rmatrices());

        // Legend
        printf("%3s %4s %16s","nl","nocc","E");
        for(size_t ir=0;ir<rmat.size();ir++) {
          std::ostringstream oss;
          oss << "<r>(" << rmat[ir].first << ")";
          printf(" %12s",oss.str().c_str());
        }
        printf(" %12s\n","r(max)");

        // Orbital info
        for(size_t io=0;io<occlist.size();io++) {
          // Orbital coefficients
          arma::vec orb(C.slice(occlist[io].l).col(occlist[io].n-occlist[io].l-1));
          // Orbital density matrix
          arma::mat P(orb*orb.t());

          printf("%2i%c %4i % 16.9f",occlist[io].n, shtype[occlist[io].l], occlist[io].nocc, occlist[io].E);

          // loop over r matrices
          arma::vec rpos(rmat.size());
          for(size_t ir=0;ir<rmat.size();ir++) {
            rpos(ir)=std::pow(arma::trace(P*rmat[ir].second),1.0/rmat[ir].first);
            printf(" %e",rpos(ir));
          }

          // Electron density maximum
          printf(" %e\n",basis.electron_density_maximum(P));
        }
      }

      void OrbitalChannel::Save(const basis::TwoDBasis & basis, const std::string & symbol) const {
        std::vector<shell_occupation_t> occlist(GetOccupied());

        // Collect the occupied orbitals
        std::vector< std::vector<int> > iocc(lmax+1);
        std::vector< std::vector<int> > occnum(lmax+1);
        std::vector< std::vector<int> > lval(lmax+1);
        std::vector< std::vector<double> > Eorb(lmax+1);
        size_t norb=0;
        for(int l=0;l<=lmax;l++) {
          for(size_t io=0;io<occlist.size();io++) {
            if(occlist[io].l != l)
              continue;
            iocc[l].push_back(occlist[io].n-l-1);
            occnum[l].push_back(occlist[io].nocc);
            Eorb[l].push_back(occlist[io].E);
          }
          norb+=iocc[l].size();
        }

        // Evaluate the orbitals
        std::vector<arma::mat> orbval(lmax+1);
        for(int l=0;l<=lmax;l++) {
          arma::uvec oidx(arma::conv_to<arma::uvec>::from(iocc[l]));
          if(!oidx.n_elem)
            continue;

          // Orbital vector
          arma::mat Cl(C.slice(l).cols(oidx));
          orbval[l]=basis.orbitals(Cl);
        }

        // Save the results
        arma::vec r(basis.radii());

        std::ostringstream oss;
        oss << symbol << "_orbs.dat";
        FILE *out = fopen(oss.str().c_str(),"w");

        // Header: number of radial points and orbitals
        fprintf(out,"%i %i\n",(int) orbval[0].n_rows,(int) norb);

        // Orbital angular momenta
        for(int l=0;l<=lmax;l++) {
          for(size_t io=0;io<Eorb[l].size();io++)
            fprintf(out," %i",l);
        }
        fprintf(out,"\n");
        // Orbital occupations
        for(int l=0;l<=lmax;l++) {
          for(size_t io=0;io<occnum[l].size();io++)
            fprintf(out," %i",occnum[l][io]);
        }
        fprintf(out,"\n");
        // Orbital energies
        for(int l=0;l<=lmax;l++) {
          for(size_t io=0;io<Eorb[l].size();io++)
            fprintf(out," %e",Eorb[l][io]);
        }
        fprintf(out,"\n");
        // Orbital values
        for(size_t ir=0;ir<orbval[0].n_rows;ir++) {
          fprintf(out,"%e",r(ir));
          for(int l=0;l<=lmax;l++)
            for(size_t ic=0;ic<orbval[l].n_cols;ic++) {
              fprintf(out," % e",orbval[l](ir,ic));
            }
          fprintf(out,"\n");
        }
        fclose(out);
      }

      bool OrbitalChannel::operator==(const OrbitalChannel & rh) const {
        if(occs.n_elem != rh.occs.n_elem)
          return false;
        for(size_t i=0;i<occs.n_elem;i++) {
          if(occs(i) != rh.occs(i))
            return false;
        }
        return true;
      }

      size_t OrbitalChannel::CountOccupied(int l) const {
        // Count occupied shells
        arma::sword numl = occs(l);
        size_t nsh;
        for(nsh=0;nsh<C.n_cols;nsh++) {
          arma::sword nocc = std::min(ShellCapacity(l), numl);
          numl -= nocc;
          if(nocc==0)
            break;
        }
        return nsh;
      }

      void OrbitalChannel::UpdateOrbitals(const arma::cube & F, const arma::mat & Sinvh) {
        E.resize(F.n_rows,lmax+1);
        C.resize(F.n_rows,F.n_rows,lmax+1);
        for(int l=0;l<=lmax;l++) {
          arma::vec El;
          helfem::scf::eig_gsym(El,C.slice(l),F.slice(l),Sinvh);
          E.col(l)=El;
        }
      }

      void OrbitalChannel::UpdateOrbitalsDamped(const arma::cube & F, const arma::mat & Sinvh, const arma::mat & S, double dampov) {
        E.resize(F.n_rows,lmax+1);
        C.resize(F.n_rows,F.n_rows,lmax+1);
        for(int l=0;l<=lmax;l++) {
          // Fock matrix
          arma::mat Fl(F.slice(l));

          size_t nsh(CountOccupied(l));
          if(nsh) {
            // Go to MO basis
            arma::mat Fmo(C.slice(l).t()*S*Fl*S*C.slice(l));
            size_t nmo=C.n_cols;

            arma::uvec oidx(arma::linspace<arma::uvec>(0,nsh-1,nsh));
            arma::uvec vidx(arma::linspace<arma::uvec>(nsh,nmo-1,nmo-nsh));
            // Damp OV blocks
            Fmo(oidx,vidx)*=dampov;
            Fmo(vidx,oidx)*=dampov;
            // Recreate Fock matrix
            Fl=C.slice(l)*Fmo*C.slice(l).t();
          }

          arma::vec El;
          helfem::scf::eig_gsym(El,C.slice(l),Fl,Sinvh);
          E.col(l)=El;
        }
      }

      void OrbitalChannel::UpdateOrbitalsShifted(const arma::cube & F, const arma::mat & Sinvh, const arma::mat & S, double shift) {
        E.resize(F.n_rows,lmax+1);
        C.resize(F.n_rows,F.n_rows,lmax+1);
        for(int l=0;l<=lmax;l++) {
          // Fock matrix
          arma::mat Fl(F.slice(l));

          // Count occupied shells
          size_t nsh(CountOccupied(l));
          arma::mat Cv;
          if(nsh) {
            // Apply level shift to virtual orbitals
            Cv=C.slice(l).cols(nsh,C.n_cols-1);
            // Shift matrix
            arma::mat shmat(shift*S*Cv*Cv.t()*S);
            // Update orbitals
            arma::vec El;
            helfem::scf::eig_gsym(El,C.slice(l),Fl+shmat,Sinvh);
            E.col(l)=El;
          } else {
            arma::vec El;
            helfem::scf::eig_gsym(El,C.slice(l),Fl,Sinvh);
            E.col(l)=El;
          }

          /*
            if(nsh) {
            arma::mat proj(C.slice(l).cols(0,nsh-1).t()*S*Co);
            printf("Projection of occupied am = %i subspace with shift %e is %e\n",(int) l, shift, arma::sum(arma::sum(arma::square(proj)))/nsh);
            E[l].subvec(0,nsh-1).t().print("Occupied eigenvalues");
            }
          */
        }
      }

      void OrbitalChannel::UpdateDensity(arma::cube & Pl) const {
        Pl.zeros(C.n_rows,C.n_rows,lmax+1);
        for(int l=0;l<=lmax;l++) {
          // Number of electrons to put in
          arma::sword numl = occs(l);
          for(size_t io=0;io<C.n_cols;io++) {
            arma::sword nocc = std::min(ShellCapacity(l), numl);
            numl -= nocc;
            Pl.slice(l) += nocc * C.slice(l).col(io) * C.slice(l).col(io).t();
            if(nocc == 0)
              break;
          }
        }
      }

      static arma::mat full_density(const arma::cube & input) {
        // Get the angular basis
        arma::ivec lval, mval;
        atomic::basis::angular_basis(input.n_slices-1,input.n_slices-1,lval,mval);
        size_t Nrad=input.n_rows;

        // Initialize
        arma::mat output;
        output.zeros(Nrad*lval.n_elem,Nrad*lval.n_elem);

        for(int l=0;l<(int) input.n_slices;l++) {
          // Indices
          arma::uvec lidx(arma::find(lval==l));
          arma::ivec msub(mval(lidx));
          // Loop over subchannels
          for(int m=-l;m<=l;m++) {
            // Find the correct angular shell
            arma::uvec midx(arma::find(msub==m));
            if(midx.n_elem != 1)
              throw std::logic_error("Shell not found!\n");
            // so the index of the angular shell is
            arma::uword angidx(lidx(midx(0)));

            // Radial density matrix already has 2l+1 factor
            output.submat(angidx*Nrad,angidx*Nrad,(angidx+1)*Nrad-1,(angidx+1)*Nrad-1) = input.slice(l)/(2*l+1);
          }
        }

        return output;
      }

      static arma::mat full_overlap(const arma::mat & S, int lmax) {
        // Get the angular basis
        arma::ivec lval, mval;
        atomic::basis::angular_basis(lmax,lmax,lval,mval);
        size_t Nrad=S.n_rows;

        // Initialize
        arma::mat output;
        output.zeros(Nrad*lval.n_elem,Nrad*lval.n_elem);
        for(size_t il=0;il<lval.size();il++) {
          output.submat(il*Nrad,il*Nrad,(il+1)*Nrad-1,(il+1)*Nrad-1) = S;
        }

        return output;
      }

      static arma::mat full_orbs(const arma::cube & C) {
        // Get the angular basis
        arma::ivec lval, mval;
        atomic::basis::angular_basis(C.n_slices-1,C.n_slices-1,lval,mval);
        size_t Nrad=C.n_rows;

        // Initialize
        arma::mat output;
        output.zeros(Nrad*lval.n_elem,Nrad*lval.n_elem);
        for(size_t il=0;il<lval.n_elem;il++) {
          output.submat(il*Nrad,il*Nrad,(il+1)*Nrad-1,(il+1)*Nrad-1) = C.slice(lval(il));
        }

        return output;
      }

      static arma::cube make_m_average(const arma::mat & input, size_t Nrad, const arma::ivec & lval, const arma::ivec & mval) {
        // Initialize
        arma::cube output;
        output.zeros(Nrad,Nrad,arma::max(lval)+1);

        for(int l=0;l<(int) output.n_slices;l++) {
          // Indices
          arma::uvec lidx(arma::find(lval==l));
          arma::ivec msub(mval(lidx));
          // Loop over subchannels
          for(int m=-l;m<=l;m++) {
            // Find the correct angular shell
            arma::uvec midx(arma::find(msub==m));
            if(midx.n_elem != 1)
              throw std::logic_error("Shell not found!\n");
            // so the index of the angular shell is
            arma::uword angidx(lidx(midx(0)));
            arma::mat subm(input.submat(angidx*Nrad,angidx*Nrad,(angidx+1)*Nrad-1,(angidx+1)*Nrad-1));
            output.slice(l) += subm;
          }
          // Average
          output.slice(l) /= 2*l+1;
        }

        return output;
      }

      static arma::mat slice_average(const arma::cube & input) {
        // Initialize
        arma::mat out(input.slice(0));
        for(int l=1;l<(int) input.n_slices;l++) {
          out += input.slice(l);
        }
        return out/input.n_slices;
      }

      arma::mat OrbitalChannel::FullDensity() const {
        // Return full matrix
        return full_density(AngularDensity());
      }

      arma::cube OrbitalChannel::AngularDensity() const {
        size_t Nrad=C.n_rows;

        // Initialize
        arma::cube P;
        P.zeros(Nrad,Nrad,lmax+1);
        for(int l=0;l<=lmax;l++) {
          // Number of electrons to put in
          arma::sword numl = occs(l);
          // Fill shells
          for(size_t io=0;io<C.n_cols;io++) {
            arma::sword nocc = std::min(ShellCapacity(l), numl);
            if(nocc == 0)
              break;

            // Fractional occupation is
            double fracocc = nocc*1.0/ShellCapacity(l);
            P.slice(l) += fracocc * C.slice(l).col(io) * C.slice(l).col(io).t();
            numl -= nocc;
          }
        }

        return P;
      }

      void OrbitalChannel::AufbauOccupations(arma::sword numel) {
        // Number of radial solutions
        size_t Nrad=E.n_rows;

        // Collect energies
        arma::vec El(E.n_elem);
        arma::ivec lval(E.n_elem);
        for(size_t l=0;l<E.n_cols;l++) {
          El.subvec(l*Nrad,(l+1)*Nrad-1)=E.col(l);
          lval.subvec(l*Nrad,(l+1)*Nrad-1)=l*arma::ones<arma::ivec>(Nrad);
        }

        // Sort in increasing energy
        arma::uvec idx(arma::sort_index(El,"ascend"));
        El=El(idx);
        lval=lval(idx);

        // Fill in electrons to shells
        occs.zeros(lmax+1);
        for(size_t i=0;i<El.n_elem;i++) {
          // Shell angular momentum is
          arma::sword l=lval(i);

          // Number of electrons to occupy shell with
          arma::sword nocc = std::min(ShellCapacity(l), numel);
          occs(l) += nocc;
          numel -= nocc;

          if(numel == 0)
            break;
        }
      }

      std::vector<OrbitalChannel> OrbitalChannel::MoveElectrons() const {
        std::vector<OrbitalChannel> ret;
        for(int shell_from=0;shell_from<=lmax;shell_from++) {
          for(int shell_to=0;shell_to<=lmax;shell_to++) {
            // Try moving up to a whole shell at a time
            for(int nmove=1;nmove<=std::min(ShellCapacity(shell_from),ShellCapacity(shell_to));nmove++) {
              // Check that we have electrons we can move
              if(occs(shell_from)<nmove)
                continue;

              // We include the identity, since otherwise fully
              // spin-polarized calculations don't work (empty list for beta
              // moves)

              // New channel
              OrbitalChannel newch(*this);
              newch.occs(shell_from)-=nmove;
              newch.occs(shell_to)+=nmove;

              ret.push_back(newch);
            }
          }
        }

        if(ret.size() == 0) {
          // Dummy list
          ret.resize(1);
          ret[0]=*this;
          ret[0].occs.zeros(lmax+1);
        }

        return ret;
      }

      bool operator==(const rconf_t & lh, const rconf_t & rh) {
        return lh.orbs == rh.orbs;
      }
      bool operator<(const rconf_t & lh, const rconf_t & rh) {
        // Sort first by convergence
        if(lh.converged && !rh.converged)
          return true;
        if(rh.converged && !lh.converged)
          return false;

        return lh.Econf < rh.Econf;
      }

      bool operator==(const uconf_t & lh, const uconf_t & rh) {
        return (lh.orbsa == rh.orbsa) && (lh.orbsb == rh.orbsb);
      }
      bool operator<(const uconf_t & lh, const uconf_t & rh) {
        // Sort first by convergence
        if(lh.converged && !rh.converged)
          return true;
        if(rh.converged && !lh.converged)
          return false;

        return lh.Econf < rh.Econf;
      }

      SCFSolver::SCFSolver(int Z, int finitenuc, double Rrms, int lmax_, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int Nquad, const arma::vec & bval, int x_func_, int c_func_, int maxit_, double shift_, double convthr_, double dftthr_, double diiseps_, double diisthr_, int diisorder_) : lmax(lmax_), maxit(maxit_), shift(shift_), convthr(convthr_), dftthr(dftthr_), diiseps(diiseps_), diisthr(diisthr_), diisorder(diisorder_) {

        // Construct the angular basis
        arma::ivec lval, mval;
        atomic::basis::angular_basis(lmax,lmax,lval,mval);

        basis=sadatom::basis::TwoDBasis(Z, (modelpotential::nuclear_model_t) (finitenuc), Rrms, poly, Nquad, bval, lmax);
        atbasis=atomic::basis::TwoDBasis(Z, (modelpotential::nuclear_model_t) (finitenuc), Rrms, poly, Nquad, bval, lval, mval, 0, 0, 0.0);
        printf("Basis set has %i radial functions\n",(int) basis.Nbf());

        // Form overlap matrix
        S=basis.overlap();
        // Get half-inverse
        Sinvh=basis.Sinvh();
        // Form kinetic energy matrix
        T=basis.kinetic();
        // Form kinetic energy matrix
        Tl=basis.kinetic_l();
        // Form nuclear attraction energy matrix
        Vnuc=basis.nuclear();
        // Form core Hamiltonian
        H0=T+Vnuc;

        // Form DFT grid
        grid=helfem::sadatom::dftgrid::DFTGrid(&basis);

        // Default values
        int ldft=4*lmax+10;
        int mdft=4*lmax+5;
        atgrid=helfem::atomic::dftgrid::DFTGrid(&atbasis,ldft,mdft);

        // Compute two-electron integrals
        basis.compute_tei();
        // Range separation?
        set_func(x_func_, c_func_);
      }

      SCFSolver::~SCFSolver() {
      }

      void SCFSolver::set_func(int x_func_, int c_func_) {
        x_func=x_func_;
        c_func=c_func_;

        bool erfc, yukawa;
        is_range_separated(x_func, erfc, yukawa);
        // Fraction of exact exchange
        double kfrac, kshort, omega;
        range_separation(x_func, omega, kfrac, kshort);
        if(omega!=0.0) {
          printf("\nUsing range-separated exchange with range-separation constant omega = % .3f.\n",omega);
          printf("Using % .3f %% short-range and % .3f %% long-range exchange.\n",(kfrac+kshort)*100,kfrac*100);
        } else if(kfrac!=0.0)
          printf("\nUsing hybrid exchange with % .3f %% of exact exchange.\n",kfrac*100);
        else
          printf("\nA pure exchange functional used, no exact exchange.\n");

        if(yukawa)
          basis.compute_yukawa(omega);
        else if(erfc)
          basis.compute_erfc(omega);
      }

      void SCFSolver::set_params(const arma::vec & px, const arma::vec & pc) {
        x_pars = px;
        c_pars = pc;
      }

      arma::mat SCFSolver::TotalDensity(const arma::cube & Pl) const {
        arma::mat P(Pl.slice(0));
        for(size_t l=1;l<Pl.n_slices;l++)
          P+=Pl.slice(l);
        return P;
      }

      void SCFSolver::Initialize(OrbitalChannel & orbs) const {
        orbs.SetLmax(lmax);
        orbs.UpdateOrbitals(ReplicateCube(H0)+KineticCube(),Sinvh);
      }

      bool is_meta(int x_func, int c_func) {
        bool ggax, mggatx, mggalx;
        is_gga_mgga(x_func, ggax, mggatx, mggalx);
        bool ggac, mggatc, mggalc;
        is_gga_mgga(c_func, ggac, mggatc, mggalc);
        return mggatx || mggatc || mggalx || mggalc;
      }

      double SCFSolver::FockBuild(rconf_t & conf) {
        // Form density
        conf.orbs.UpdateDensity(conf.Pl);
        arma::mat P(TotalDensity(conf.Pl));
        if(verbose) {
          printf("Tr P = %f\n",arma::trace(P*S));
          fflush(stdout);
        }

        // Angular factor
        double angfac(4.0*M_PI);
        // Kinetic energy
        arma::cube kc(KineticCube());

        // Compute energy
        conf.Ekin=arma::trace(P*T);
        for(arma::sword l=0;l<=lmax;l++)
          conf.Ekin+=arma::trace(conf.Pl.slice(l)*kc.slice(l));
        conf.Epot=arma::trace(P*Vnuc);
        if(verbose) {
          printf("Kinetic energy %.10e\n",conf.Ekin);
          printf("Nuclear attraction energy %.10e\n",conf.Epot);
          fflush(stdout);
        }

        // Form Coulomb matrix
        arma::mat J(basis.coulomb(P/angfac));
        conf.Ecoul=0.5*arma::trace(P*J);
        if(verbose) {
          printf("Coulomb energy %.10e\n",conf.Ecoul);
          fflush(stdout);
        }

        // Exchange-correlation
        conf.Exc=0.0;

        arma::cube XC;
        double nelnum;
        if(x_func > 0 || c_func > 0) {
          if(is_meta(x_func,c_func)) {
            arma::mat XCfull;
            double Ekin;
            atgrid.eval_Fxc(x_func, x_pars, c_func, c_pars, full_density(conf.Pl), XCfull, conf.Exc, nelnum, Ekin, dftthr);
            XC=make_m_average(XCfull,atbasis.Nrad(),atbasis.get_lval(),atbasis.get_mval());
          } else {
            arma::mat XCm;
            grid.eval_Fxc(x_func, x_pars, c_func, c_pars, P/angfac, XCm, conf.Exc, nelnum, dftthr);
            // Potential needs to be divided as well
            XCm/=angfac;
            XC=ReplicateCube(XCm);
          }
          if(verbose) {
            printf("DFT energy %.10e\n",conf.Exc);
            printf("Error in integrated number of electrons % e\n",nelnum-conf.orbs.Nel());
            fflush(stdout);
          }
        }

        // Fraction of exact exchange
        arma::cube K;
        // Fraction of exact exchange
        double kfrac, kshort, omega;
        range_separation(x_func, omega, kfrac, kshort);
        if(kfrac!=0.0 || kshort!=0.0) {
          K.zeros(P.n_rows,P.n_rows,lmax+1);
          if(kfrac!=0.0)
            K+=kfrac*basis.exchange(conf.orbs.AngularDensity());
          if(kshort!=0.0)
            K+=kshort*basis.rs_exchange(conf.orbs.AngularDensity());

          double Exx=0.0;
          for(int l=0;l<=lmax;l++)
            Exx += 0.5*arma::trace(K.slice(l)*conf.Pl.slice(l));
          if(verbose) {
            printf("Exact exchange energy %.10e\n",Exx);
            fflush(stdout);
          }
          conf.Exc += Exx;
        }

        // Fock matrices
        conf.Fl=ReplicateCube(H0+J)+kc;
        if(kfrac!=0.0 || kshort!=0.0)
          conf.Fl+=K;
        if(x_func>0 || c_func>0)
          conf.Fl+=XC;

        // Update energy
        conf.Econf=conf.Ekin+conf.Epot+conf.Ecoul+conf.Exc;

        return conf.Econf;
      }

      double SCFSolver::FockBuild(uconf_t & conf) {
        // Form density
        conf.orbsa.UpdateDensity(conf.Pal);
        conf.orbsb.UpdateDensity(conf.Pbl);

        arma::cube Pl(conf.Pal+conf.Pbl);
        arma::mat Pa(TotalDensity(conf.Pal));
        arma::mat Pb(TotalDensity(conf.Pbl));
        arma::mat P(Pa+Pb);

        // Angular factor
        double angfac(4.0*M_PI);

        // Kinetic energy
        arma::cube kc(KineticCube());

        // Compute energy
        conf.Ekin=arma::trace(P*T);
        for(arma::sword l=0;l<=lmax;l++)
          conf.Ekin+=arma::trace(Pl.slice(l)*kc.slice(l));
        conf.Epot=arma::trace(P*Vnuc);

        // Form Coulomb matrix
        arma::mat J(basis.coulomb(P/angfac));
        conf.Ecoul=0.5*arma::trace(P*J);
        if(verbose) {
          printf("Coulomb energy %.10e\n",conf.Ecoul);
          fflush(stdout);
        }

        // Exchange-correlation
        conf.Exc=0.0;
        arma::cube XCa, XCb;
        double nelnum;
        if(is_meta(x_func,c_func)) {
          arma::mat XCafull, XCbfull;
          double Ekin;
          atgrid.eval_Fxc(x_func, x_pars, c_func, c_pars, full_density(conf.Pal), full_density(conf.Pbl), XCafull, XCbfull, conf.Exc, nelnum, Ekin, true, dftthr);
          XCa=make_m_average(XCafull,atbasis.Nrad(),atbasis.get_lval(),atbasis.get_mval());
          XCb=make_m_average(XCbfull,atbasis.Nrad(),atbasis.get_lval(),atbasis.get_mval());
        } else {
          arma::mat XCam, XCbm;
          grid.eval_Fxc(x_func, x_pars, c_func, c_pars, Pa/angfac, Pb/angfac, XCam, XCbm, conf.Exc, nelnum, true, dftthr);
          // Potential needs to be divided as well
          XCam/=angfac;
          XCbm/=angfac;
          XCa=ReplicateCube(XCam);
          XCb=ReplicateCube(XCbm);
        }
        if(verbose) {
          printf("DFT energy %.10e\n",conf.Exc);
          printf("Error in integrated number of electrons % e\n",nelnum-conf.orbsa.Nel()-conf.orbsb.Nel());
          fflush(stdout);
        }

        // Fraction of exact exchange
        arma::cube Ka, Kb;
        // Fraction of exact exchange
        double kfrac, kshort, omega;
        range_separation(x_func, omega, kfrac, kshort);
        if(kfrac!=0.0 || kshort!=0.0) {
          Ka.zeros(P.n_rows,P.n_rows,lmax+1);
          Kb.zeros(P.n_rows,P.n_rows,lmax+1);
          if(kfrac!=0.0) {
            Ka+=kfrac*basis.exchange(conf.orbsa.AngularDensity());
            Kb+=kfrac*basis.exchange(conf.orbsb.AngularDensity());
          }
          if(kshort!=0.0) {
            Ka+=kshort*basis.rs_exchange(conf.orbsa.AngularDensity());
            Kb+=kshort*basis.rs_exchange(conf.orbsb.AngularDensity());
          }

          double Exx=0.0;
          for(int l=0;l<=lmax;l++)
            Exx += 0.5*arma::trace(Ka.slice(l)*conf.Pal.slice(l)) + 0.5*arma::trace(Kb.slice(l)*conf.Pbl.slice(l));
          if(verbose) {
            printf("Exact exchange energy %.10e\n",Exx);
            fflush(stdout);
          }
          conf.Exc += Exx;
        }

        // Fock matrices
        conf.Fal=ReplicateCube(H0+J)+kc;
        conf.Fbl=conf.Fal;
        if(kfrac!=0.0 || kshort!=0.0) {
          conf.Fal+=Ka;
          conf.Fbl+=Kb;
        }
        if(x_func>0 || c_func>0) {
          conf.Fal+=XCa;
          conf.Fbl+=XCb;
        }

        // Update energy
        conf.Econf=conf.Ekin+conf.Epot+conf.Ecoul+conf.Exc;

        return conf.Econf;
      }

      arma::mat SCFSolver::SuperMat(const arma::mat & M) const {
        arma::mat superM(M.n_rows*(lmax+1),M.n_cols*(lmax+1));
        superM.zeros();
        for(int l=0;l<=lmax;l++) {
          superM.submat(l*M.n_rows,l*M.n_cols,(l+1)*M.n_rows-1,(l+1)*M.n_cols-1)=M;
        }
        return superM;
      }

      arma::cube SCFSolver::ReplicateCube(const arma::mat & M) const {
        arma::cube Msuper(M.n_rows,M.n_cols,lmax+1);
        Msuper.zeros();
        for(int l=0;l<=lmax;l++) {
          Msuper.slice(l)=M;
        }
        return Msuper;
      }

      arma::cube SCFSolver::KineticCube() const {
        // Kinetic energy l factors
        arma::cube Tc(T.n_rows,T.n_cols,lmax+1);
        Tc.zeros();
        for(arma::sword l=0;l<=lmax;l++)
          Tc.slice(l)=l*(l+1)*Tl;
        return Tc;
      }

      arma::mat SCFSolver::SuperCube(const arma::cube & M) const {
        arma::mat Msuper(M.n_rows*(lmax+1),M.n_cols*(lmax+1));
        Msuper.zeros();
        for(int l=0;l<=lmax;l++) {
          Msuper.submat(l*M.n_rows,l*M.n_cols,(l+1)*M.n_rows-1,(l+1)*M.n_cols-1)=M.slice(l);
        }
        return Msuper;
      }

      arma::cube SCFSolver::MiniMat(const arma::mat & Msuper) const {
        arma::cube M(Msuper.n_rows/(lmax+1),Msuper.n_cols/(lmax+1),lmax+1);
        M.zeros();
        for(int l=0;l<=lmax;l++) {
          M.slice(l)=Msuper.submat(l*M.n_rows,l*M.n_cols,(l+1)*M.n_rows-1,(l+1)*M.n_cols-1);
        }
        return M;
      }

      double SCFSolver::Solve(rconf_t & conf) {
        if(!conf.orbs.OrbitalsInitialized())
          throw std::logic_error("Orbitals not initialized!\n");
        if(!conf.orbs.Restricted())
          throw std::logic_error("Running restricted calculation with unrestricted orbitals!\n");
        if(conf.orbs.Occs().n_elem != (arma::uword) (lmax+1))
          throw std::logic_error("Occupation vector is of wrong length!\n");

        verbose = false;

        if(verbose) {
          printf("Running SCF for orbital occupations\n");
          conf.orbs.Occs().t().print();
        }

        // DIIS object. ADIIS doesn't work for (significant) fractional occupation
        bool usediis=true, useadiis=true;
        ::rDIIS diis(SuperMat(S),SuperMat(Sinvh),usediis,diiseps,diisthr,useadiis,verbose,diisorder);
        double diiserr;

        double E=0.0, Eold;

        arma::sword iscf;
        for(iscf=1;iscf<=maxit;iscf++) {
          if(verbose) {
            printf("\n**** Iteration %i ****\n\n",(int) iscf);
          }

          // Form Fock matrix
          Eold=E;
          E=FockBuild(conf);

          double dE=E-Eold;
          if(verbose) {
            printf("Total energy is % .10f\n",E);
            if(iscf>1)
              printf("Energy changed by %e\n",dE);
            fflush(stdout);
          }

          // Since Fock operator depends on the l channel, we need to create
          // a supermatrix for DIIS.
          arma::mat Fsuper(SuperCube(conf.Fl));
          // Need density for DIIS as well
          arma::mat Psuper(SuperCube(conf.Pl));
          // Update DIIS
          diis.update(Fsuper,Psuper,E,diiserr);
          if(verbose) {
            printf("DIIS error is %e\n",diiserr);
            fflush(stdout);
          }
          // Have we converged? Note that DIIS error is still wrt full space, not active space.
          conf.converged=(diiserr<convthr) && (std::abs(dE)<convthr);

          // Solve DIIS to get Fock update
          diis.solve_F(Fsuper);
          conf.Fl=MiniMat(Fsuper);

          // Update orbitals and density
          if(diiserr > diisthr) {
            // Since ADIIS is unreliable, we also use a level shift.
            conf.orbs.UpdateOrbitalsShifted(conf.Fl,Sinvh,S,shift);
          } else {
            conf.orbs.UpdateOrbitals(conf.Fl,Sinvh);
          }

          if(conf.converged)
            break;
        }
        if(iscf > maxit) {
          printf("*** Not converged; DIIS error %e ***\n",diiserr);
          fflush(stdout);
        }

        if(verbose) {
          printf("%-21s energy: % .16f\n","Kinetic",conf.Ekin);
          printf("%-21s energy: % .16f\n","Nuclear attraction",conf.Epot);
          printf("%-21s energy: % .16f\n","Coulomb",conf.Ecoul);
          printf("%-21s energy: % .16f\n","Exchange-correlation",conf.Exc);
          printf("%-21s energy: % .16f\n","Total",conf.Econf);
          printf("%-21s energy: % .16f\n","Virial ratio",-conf.Econf/conf.Ekin);
          printf("\n");

          // Electron density at nucleus
          printf("Electron density at nucleus % .10e\n",basis.nuclear_density(TotalDensity(conf.Pl)));
        } else {
          printf("Evaluated energy % .16f for configuration ",conf.Econf);

          arma::ivec occs(conf.orbs.Occs());
          for(size_t i=0;i<occs.size();i++)
            printf(" %i",(int) occs(i));
          printf("\n");
          fflush(stdout);
        }

        return E;
      }

      double SCFSolver::Solve(uconf_t & conf) {
        if(!conf.orbsa.OrbitalsInitialized())
          throw std::logic_error("Orbitals not initialized!\n");
        if(!conf.orbsb.OrbitalsInitialized())
          throw std::logic_error("Orbitals not initialized!\n");
        if(conf.orbsa.Occs().n_elem != (arma::uword) (lmax+1))
          throw std::logic_error("Occupation vector is of wrong length!\n");
        if(conf.orbsb.Occs().n_elem != (arma::uword) (lmax+1))
          throw std::logic_error("Occupation vector is of wrong length!\n");

        if(conf.orbsa.Restricted())
          throw std::logic_error("Running unrestricted calculation with restricted orbitals!\n");
        if(conf.orbsb.Restricted())
          throw std::logic_error("Running unrestricted calculation with restricted orbitals!\n");

        double E=0.0, Eold;

        verbose = false;

        if(verbose) {
          printf("Running SCF for orbital occupations\n");
          conf.orbsa.Occs().t().print();
          conf.orbsb.Occs().t().print();
        }

        // S supermatrix
        bool combine=false, usediis=true, useadiis=true;
        uDIIS diis(SuperMat(S),SuperMat(Sinvh),combine, usediis,diiseps,diisthr,useadiis,verbose,diisorder);
        double diiserr;

        arma::sword iscf;
        for(iscf=1;iscf<=maxit;iscf++) {
          if(verbose) {
            printf("\n**** Iteration %i ****\n\n",(int) iscf);
          }

          Eold=E;
          E=FockBuild(conf);
          double dE=E-Eold;

          if(verbose) {
            printf("Total energy is % .10f\n",E);
            if(iscf>1)
              printf("Energy changed by %e\n",dE);
            fflush(stdout);
          }

          // Since Fock operator depends on the l channel, we need to create
          // a supermatrix for DIIS.
          arma::mat Fasuper(SuperCube(conf.Fal)), Fbsuper(SuperCube(conf.Fbl));
          arma::mat Pasuper(SuperCube(conf.Pal));
          arma::mat Pbsuper(SuperCube(conf.Pbl));
          // Update DIIS
          diis.update(Fasuper,Fbsuper,Pasuper,Pbsuper,E,diiserr);
          if(verbose) {
            printf("DIIS error is %e\n",diiserr);
            fflush(stdout);
          }

          // Have we converged? Note that DIIS error is still wrt full space, not active space.
          conf.converged=(diiserr<convthr) && (std::abs(dE)<convthr);

          // Solve DIIS to get Fock update
          diis.solve_F(Fasuper,Fbsuper);
          conf.Fal=MiniMat(Fasuper);
          conf.Fbl=MiniMat(Fbsuper);

          // Update orbitals and density
          if(diiserr > diisthr) {
            // Since ADIIS is unreliable, we also use a level shift
            conf.orbsa.UpdateOrbitalsShifted(conf.Fal,Sinvh,S,shift);
            conf.orbsb.UpdateOrbitalsShifted(conf.Fbl,Sinvh,S,shift);
          } else {
            conf.orbsa.UpdateOrbitals(conf.Fal,Sinvh);
            conf.orbsb.UpdateOrbitals(conf.Fbl,Sinvh);
          }
          if(conf.converged)
            break;
        }
        if(iscf > maxit) {
          printf("*** Not converged; DIIS error %e ***\n",diiserr);
          fflush(stdout);
        }

        if(verbose) {
          printf("%-21s energy: % .16f\n","Kinetic",conf.Ekin);
          printf("%-21s energy: % .16f\n","Nuclear attraction",conf.Epot);
          printf("%-21s energy: % .16f\n","Coulomb",conf.Ecoul);
          printf("%-21s energy: % .16f\n","Exchange-correlation",conf.Exc);
          printf("%-21s energy: % .16f\n","Total",conf.Econf);
          printf("%-21s energy: % .16f\n","Virial ratio",-conf.Econf/conf.Ekin);
          printf("\n");

          // Electron density at nucleus
          printf("Electron density at nucleus % .10e % .10e\n",basis.nuclear_density(TotalDensity(conf.Pal)),basis.nuclear_density(TotalDensity(conf.Pbl)));
        } else {
          printf("Evaluated energy % .16f for configuration ",E);

          arma::ivec occa(conf.orbsa.Occs());
          for(size_t i=0;i<occa.size();i++)
            printf(" %i",(int) occa(i));
          arma::ivec occb(conf.orbsb.Occs());
          for(size_t i=0;i<occb.size();i++)
            printf(" %i",(int) occb(i));
          printf("\n");
          fflush(stdout);
        }

        return E;
      }

      arma::mat SCFSolver::RestrictedPotential(rconf_t & conf) {
        if(!conf.orbs.OrbitalsInitialized())
          throw std::logic_error("No orbitals!\n");

        arma::mat P=TotalDensity(conf.Pl);

        arma::vec r(basis.radii());
        arma::vec wt(basis.quadrature_weights());
        arma::mat vcoul(basis.coulomb_screening(P));
        arma::vec vxc(basis.xc_screening(P,x_func,c_func));
        arma::vec Zeff(vcoul+vxc);
        arma::vec rho(basis.electron_density(P));
        arma::vec grho(basis.electron_density_gradient(P));
        arma::vec lrho(basis.electron_density_laplacian(P));

        arma::mat result(Zeff.n_rows,8);
        result.col(0)=r;
        result.col(1)=rho;
        result.col(2)=grho;
        result.col(3)=lrho;
        result.col(4)=vcoul;
        result.col(5)=vxc;
        result.col(6)=wt;
        result.col(7)=arma::ones<arma::vec>(Zeff.n_elem)*basis.charge()-Zeff;

        return result;
      }

      arma::mat SCFSolver::UnrestrictedPotential(uconf_t & conf) {
        if(!conf.orbsa.OrbitalsInitialized())
          throw std::logic_error("No orbitals!\n");
        if(!conf.orbsb.OrbitalsInitialized())
          throw std::logic_error("No orbitals!\n");

        arma::mat Pa=TotalDensity(conf.Pal);
        arma::mat Pb=TotalDensity(conf.Pbl);
        arma::mat P(Pa+Pb);

        arma::vec r(basis.radii());
        arma::vec wt(basis.quadrature_weights());
        arma::vec vcoul(basis.coulomb_screening(P));
        arma::mat vxcm(basis.xc_screening(Pa,Pb,x_func,c_func));
        // Averaged potential
        arma::vec vxc=arma::mean(vxcm,1);
        arma::vec Zeff(vcoul+vxc);
        arma::vec rho(basis.electron_density(P));
        arma::vec grho(basis.electron_density_gradient(P));
        arma::vec lrho(basis.electron_density_laplacian(P));

        printf("Electron density by quadrature: %e\n",arma::sum(wt%rho%r%r));

        arma::mat result(r.n_elem,8);
        result.col(0)=r;
        result.col(1)=rho;
        result.col(2)=grho;
        result.col(3)=lrho;
        result.col(4)=vcoul;
        result.col(5)=vxc;
        result.col(6)=wt;
        result.col(7)=arma::ones<arma::vec>(Zeff.n_elem)*basis.charge()-Zeff;

        return result;
      }

      arma::mat SCFSolver::AveragePotential(uconf_t & conf) {
        if(!conf.orbsa.OrbitalsInitialized())
          throw std::logic_error("No orbitals!\n");
        if(!conf.orbsb.OrbitalsInitialized())
          throw std::logic_error("No orbitals!\n");

        arma::mat Pa=TotalDensity(conf.Pal);
        arma::mat Pb=TotalDensity(conf.Pbl);
        arma::mat P(Pa+Pb);

        arma::vec wt(basis.quadrature_weights());
        arma::vec vcoul(basis.coulomb_screening(P));
        arma::vec vxc(basis.xc_screening(P,x_func,c_func));
        arma::vec Zeff(vcoul+vxc);

        arma::vec r(basis.radii());
        arma::vec rho(basis.electron_density(P));
        arma::vec grho(basis.electron_density_gradient(P));
        arma::vec lrho(basis.electron_density_laplacian(P));

        arma::mat result(Zeff.n_rows,8);
        result.col(0)=r;
        result.col(1)=rho;
        result.col(2)=grho;
        result.col(3)=lrho;
        result.col(4)=vcoul;
        result.col(5)=vxc;
        result.col(6)=wt;
        result.col(7)=arma::ones<arma::vec>(Zeff.n_elem)*basis.charge()-Zeff;

        return result;
      }

      arma::mat SCFSolver::WeightedPotential(uconf_t & conf) {
        if(!conf.orbsa.OrbitalsInitialized())
          throw std::logic_error("No orbitals!\n");
        if(!conf.orbsb.OrbitalsInitialized())
          throw std::logic_error("No orbitals!\n");

        arma::mat Pa=TotalDensity(conf.Pal);
        arma::mat Pb=TotalDensity(conf.Pbl);
        arma::mat P(Pa+Pb);

        arma::vec r(basis.radii());
        arma::vec wt(basis.quadrature_weights());
        arma::vec vcoul(basis.coulomb_screening(P));
        arma::mat vxcm(basis.xc_screening(Pa,Pb,x_func,c_func));
        arma::vec rhoa(basis.electron_density(Pa));
        arma::vec grhoa(basis.electron_density_gradient(Pa));
        arma::vec lrhoa(basis.electron_density_laplacian(Pa));
        arma::vec rhob(basis.electron_density(Pb));
        arma::vec grhob(basis.electron_density_gradient(Pb));
        arma::vec lrhob(basis.electron_density_laplacian(Pb));
        // Averaged potential
        arma::vec vxc((vxcm.col(0)%rhoa + vxcm.col(1)%rhob)/(rhoa+rhob));
        // Set areas of small electron density to zero
        arma::vec n(rhoa+rhob);
        vxc(arma::find(n<dftthr)).zeros();
        arma::vec Zeff(vcoul+vxc);

        arma::mat result(Zeff.n_rows,8);
        result.col(0)=r;
        result.col(1)=rhoa+rhob;
        result.col(2)=grhoa+grhob;;
        result.col(3)=lrhoa+lrhob;
        result.col(4)=vcoul;
        result.col(5)=vxc;
        result.col(6)=wt;
        result.col(7)=arma::ones<arma::vec>(Zeff.n_elem)*basis.charge()-Zeff;

        return result;
      }

      arma::mat SCFSolver::HighSpinPotential(uconf_t & conf) {
        if(!conf.orbsa.OrbitalsInitialized())
          throw std::logic_error("No orbitals!\n");
        if(!conf.orbsb.OrbitalsInitialized())
          throw std::logic_error("No orbitals!\n");

        arma::mat Pa=TotalDensity(conf.Pal);
        arma::mat Pb=TotalDensity(conf.Pbl);
        arma::mat Pcoul(Pa+Pb);
        arma::mat Pxc(2*Pa);

        arma::vec r(basis.radii());
        arma::vec wt(basis.quadrature_weights());
        arma::vec vcoul(basis.coulomb_screening(Pcoul));
        arma::mat vxc(basis.xc_screening(Pxc,x_func,c_func));
        arma::vec rhoa(basis.electron_density(Pa));
        arma::vec grhoa(basis.electron_density_gradient(Pa));
        arma::vec lrhoa(basis.electron_density_laplacian(Pa));
        arma::vec rhob(basis.electron_density(Pb));
        arma::vec grhob(basis.electron_density_gradient(Pb));
        arma::vec lrhob(basis.electron_density_laplacian(Pb));
        arma::vec Zeff(vcoul+vxc);

        arma::mat result(Zeff.n_rows,8);
        result.col(0)=r;
        result.col(1)=rhoa+rhob;
        result.col(2)=grhoa+grhob;
        result.col(3)=lrhoa+lrhob;
        result.col(4)=vcoul;
        result.col(5)=vxc;
        result.col(6)=wt;
        result.col(7)=arma::ones<arma::vec>(Zeff.n_elem)*basis.charge()-Zeff;

        return result;
      }

      arma::mat SCFSolver::LowSpinPotential(uconf_t & conf) {
        if(!conf.orbsa.OrbitalsInitialized())
          throw std::logic_error("No orbitals!\n");
        if(!conf.orbsb.OrbitalsInitialized())
          throw std::logic_error("No orbitals!\n");

        arma::mat Pa=TotalDensity(conf.Pal);
        arma::mat Pb=TotalDensity(conf.Pbl);
        arma::mat Pcoul(Pa+Pb);
        arma::mat Pxc(2*Pb);

        arma::vec r(basis.radii());
        arma::vec wt(basis.quadrature_weights());
        arma::vec vcoul(basis.coulomb_screening(Pcoul));
        arma::mat vxc(basis.xc_screening(Pxc,x_func,c_func));
        arma::vec rhoa(basis.electron_density(Pa));
        arma::vec grhoa(basis.electron_density_gradient(Pa));
        arma::vec lrhoa(basis.electron_density_laplacian(Pa));
        arma::vec rhob(basis.electron_density(Pb));
        arma::vec grhob(basis.electron_density_gradient(Pb));
        arma::vec lrhob(basis.electron_density_laplacian(Pb));
        arma::vec Zeff(vcoul+vxc);

        arma::mat result(Zeff.n_rows,8);
        result.col(0)=r;
        result.col(1)=rhoa+rhob;
        result.col(2)=grhoa+grhob;
        result.col(3)=lrhoa+lrhob;
        result.col(4)=vcoul;
        result.col(5)=vxc;
        result.col(6)=wt;
        result.col(7)=arma::ones<arma::vec>(Zeff.n_elem)*basis.charge()-Zeff;

        return result;
      }

      const sadatom::basis::TwoDBasis & solver::SCFSolver::Basis() const {
        return basis;
      }

      double SCFSolver::nuclear_density(const rconf_t & conf) const {
        return basis.nuclear_density(TotalDensity(conf.Pl));
      }

      double SCFSolver::nuclear_density(const uconf_t & conf) const {
        return basis.nuclear_density(TotalDensity(conf.Pal+conf.Pbl));
      }

      double SCFSolver::nuclear_density_gradient(const rconf_t & conf) const {
        return basis.nuclear_density_gradient(TotalDensity(conf.Pl));
      }

      double SCFSolver::nuclear_density_gradient(const uconf_t & conf) const {
        return basis.nuclear_density_gradient(TotalDensity(conf.Pal+conf.Pbl));
      }
    }
  }
}
