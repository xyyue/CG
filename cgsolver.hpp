#ifndef cgsolver_hpp
#define cgsolver_hpp

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <cstdlib>

#include "cgmapper.hpp"
#include "ell_sparsematrix.hpp"
#include "legionvector.hpp"

template<typename T>
class CGSolver{

	private:
	int niter;
	T L2normr;
	
	public:
	bool Solve(SpMatrix &A,
		         Array<T> &b,
		         Array<T> &x,
		         int nitermax,
		         T threshold,
		         Context ctx,
		         HighLevelRuntime *runtime);

	int GetNumberIterations(void) { return niter;}
	T GetL2Norm(void) { return L2normr;}
};

template<typename T>
bool CGSolver<T>::Solve(SpMatrix &A,
                   Array<T> &b,
                   Array<T> &x,
                   int nitermax,
                   T threshold,
		               Context ctx, 
                   HighLevelRuntime *runtime)
{
  bool converged = false;
	assert(A.nrows == b.size);
	assert(b.size == x.size);
		   
	if(nitermax == -1) nitermax = A.nrows;

  printf("the size is %d\n", (int) x.size);
  printf("the nparts is %d\n", (int) x.nparts);

	Array<T> r_old(x.size, x.nparts, ctx, runtime);
	Array<T> p(x.size, x.nparts, ctx, runtime);
	Array<T> A_p(x.size, x.nparts, ctx, runtime);
  A_p.Initialize(ctx, runtime);

  Predicate loop_pred = Predicate::TRUE_PRED;

	// Ap = A * x	
	//spmv(A, x, A_p, loop_pred, ctx, runtime);

  printf("Before the SPMV!!!\n");
  printf("Before the ADDED\n");
  //A.print_nodes(ctx, runtime);
  //IndexIterator itr_read(runtime, ctx, x.is);
	std::cout<<"Before SPMV~"<<std::endl;
  
  // FIXME: A.spmv(x, A_p, ctx, runtime);
  std::cout << "=====================================" << std::endl;
  std::cout << "This is the " << 0 << " call of spmv.." << std::endl;
  std::cout << "=====================================" << std::endl;

  std::cout << "=======================" << std::endl;
  std::cout << "Before the Multiplication, b's value:" << std::endl;
  b.PrintVals(ctx, runtime);
  std::cout << "Before the Multiplication, A_p's value:" << std::endl;
  A_p.PrintVals(ctx, runtime);
  std::cout << "=======================" << std::endl;
  A.Print(ctx, runtime);
  A.spmv(b, A_p, ctx, runtime);
  std::cout << "=======================" << std::endl;
  std::cout << "After the Multiplication:" << std::endl;
  A_p.PrintVals(ctx, runtime);
  std::cout << "=======================" << std::endl;

  exit(0);
	std::cout<<"Ax = A * x is done."<<std::endl;

  printf("After the SPMV!!!\n");
  //A.print_nodes(ctx, runtime);

	// r_old = b - Ap
	subtract(b, A_p, r_old, T(1.0), ctx, runtime);
	std::cout<<"r = b - Ax is done."<<std::endl;

	// Initial norm
	const T L2normr0 = L2norm(r_old, ctx, runtime);
	std::cout<<"L2normr0 is done."<<std::endl;
	L2normr = L2normr0;

	// p = r_old
	copy(r_old, p, ctx, runtime);
	std::cout<<"copy is done."<<std::endl;

	niter = 0;
	//std::cout<<"Iteration"<<"    "<<"L2norm"<<std::endl;
	//std::cout<<niter<<"            "<<std::setprecision(16)<<L2normr<<std::endl;

  Future r2_old, pAp, alpha, r2_new, beta; 
#ifdef PREDICATED_EXECUTION
  std::deque<Future>  pending_norms;
  const int max_norm_depth = runtime->get_tunable_value(ctx, PREDICATED_TUNABLE);
#endif

	std::cout<<"Iterating..."<<std::endl;

	while(niter < nitermax){
		
		std::cout<<niter<<"            "<<L2normr<<std::endl;
		niter++;

		// Ap = A * p
    std::cout << "AAA" << std::endl;
		//spmv(A, p, A_p, loop_pred, ctx, runtime);
    A.spmv(p, A_p, ctx, runtime);

		// r2 = r' * r
    std::cout << "BBB" << std::endl;
		r2_old = dot(r_old, r_old, loop_pred, r2_old, ctx, runtime);

		// pAp = p' * A * p
    std::cout << "CCC" << std::endl;
		pAp = dot(p, A_p, loop_pred, pAp, ctx, runtime);	

		// alpha = r2 / pAp
    std::cout << "DDD" << std::endl;
		alpha = compute_scalar<T>(r2_old, pAp, loop_pred, alpha, ctx, runtime);	
	
		// x = x + alpha * p
    std::cout << "EEE" << std::endl;
		add_inplace(x, p, alpha, loop_pred, ctx, runtime);
	
		// r_old = r_old - alpha * A_p
    std::cout << "FFF" << std::endl;
    subtract_inplace(r_old, A_p, alpha, loop_pred, ctx, runtime);

    std::cout << "GGG" << std::endl;
		r2_new = dot(r_old, r_old, loop_pred, r2_new, ctx, runtime);

    std::cout << "HHH" << std::endl;
		beta = compute_scalar<T>(r2_new, r2_old, loop_pred, beta, ctx, runtime);
	
		// p = r_old + beta*p
    std::cout << "III" << std::endl;
    axpy_inplace(r_old, p, beta, loop_pred, ctx, runtime);
#ifdef PREDICATED_EXECUTION
    Future norm = dot(r_old, r_old, loop_pred, 
                      pending_norms.empty() ? Future() : pending_norms.back(), ctx ,runtime);
    loop_pred = test_convergence(norm, L2normr0, threshold, 
        loop_pred, ctx, runtime);
    pending_norms.push_back(norm);
    if (pending_norms.size() == max_norm_depth) {
      // Pop the next future off the stack and wait for it
      norm = pending_norms.front();
      pending_norms.pop_front();
      L2normr = sqrt(norm.get_result<double>());
      converged = ((L2normr/L2normr0) < threshold);
      if (converged) {
        std::cout<<"Converged! :)"<<std::endl;
        break;
      }
    }
#else
		L2normr = L2norm(r_old, ctx, runtime);


		if(L2normr/ L2normr0 < threshold){
                  converged = true;
		  std::cout<<"Converged! :)"<<std::endl;
		  std::cout<<"Iteration"<<"    "<<"L2norm"<<std::endl;
		  std::cout<<niter<<"            "<<L2normr<<std::endl;		  
                  break;
                }

#endif
	}
	converged = true;
  
  cout << "The matrix is : " << endl;
  A.Print(ctx, runtime);

  cout << "The vector b is : " << endl;
  b.PrintVals(ctx, runtime);

  cout << "The result x is : " << endl;
  x.PrintVals(ctx, runtime);

	//T dummy = beta.get_result<T>();

	// destroy the objects
  r_old.DestroyArray(ctx, runtime);
  p.DestroyArray(ctx, runtime);
  A_p.DestroyArray(ctx, runtime);
	
  return converged;
} 	

#endif
