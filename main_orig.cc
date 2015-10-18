#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"

#include "params.hpp"
#include "legionvector.hpp"
#include "ell_sparsematrix.hpp"
#include "cgoperators.hpp"
#include "cgsolver.hpp"
#include "cgmapper.hpp"
#include <time.h>
using namespace LegionRuntime::HighLevel;

enum TaskIDs {
   TOP_LEVEL_TASK_ID = 0,
};

void top_level_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime){

	int64_t nx = 15; 
	int64_t nparts = 1;
        int iter_max = -1;
    	const InputArgs &command_args = HighLevelRuntime::get_input_args();
        // Parse command line arguments
        for (int i = 1; i < command_args.argc; i++)
        {
          if (!strcmp(command_args.argv[i], "-n"))
          {
            nx = atoi(command_args.argv[++i]);
            assert(nx > 0);
            continue;
          }
          if (!strcmp(command_args.argv[i], "-max"))
          {
            iter_max = atoi(command_args.argv[++i]);
            assert(iter_max >= 0);
            continue;
          }
        }
   
	
	// get naprts from the custom mapper
	nparts = runtime->get_tunable_value(ctx, SUBREGION_TUNABLE, 0);
	//std::cout<<nparts<<std::endl;
	//std::cin.get();

   	int64_t size = nx * nx;

	std::cout<<"generate problem..."<<std::endl;   	
   	Params<double> params(nx);
   	params.GenerateVals();
	std::cout<<"Problem generation done. Some properties are as follows:"<<std::endl;
	std::cout<<"*******************************************************"<<std::endl;

	// report the problem size and memory usage 
	std::cout<<"SPARSE MATRIX STORAGE FORMAT = ELL"<<std::endl;
	std::cout<<"MATRIX DIMENSIONS="<<params.nrows<<"x"<<params.nrows<<std::endl;
	std::cout<<"MEMORY SPENT ON  NONZERO VALUES = "<<params.max_nzeros * params.nrows * sizeof(double) / 1e6 <<
	" Mb"<<std::endl;
	std::cout<<"MEMORY SPENT ON COLUMN INDEX OF NONZERO VALUES  = "<<params.max_nzeros * params.nrows * 
	sizeof(int) / 1e6 <<" Mb"<<std::endl;
	std::cout<<"MEMORY SPENT ON X VECTOR = "<<params.nrows * sizeof(double) / 1e6 <<" Mb"<<std::endl;
	std::cout<<"MEMORY SPENT ON RHS VECTOR = "<<params.nrows * sizeof(double) / 1e6 <<" Mb"<<std::endl;
	std::cout<<"*******************************************************"<<std::endl;
	std::cout<<std::endl;
	
	// build unknown vector   
	std::cout<<"Make  unknown vector x..."<<std::endl;	
	Array<double> x(size, nparts, ctx, runtime);
	x.Initialize(ctx, runtime);

	// build rhs vector
	std::cout<<"Make rhs vector..."<<std::endl;
	Array<double> b(size, nparts, ctx, runtime);
	b.Initialize(params.rhs, ctx, runtime);	
	
	// build sparse matrix
	std::cout<<"Make sparse matrix..."<<std::endl;
	SpMatrix A(size, nparts, params.nonzeros, params.max_nzeros, ctx, runtime);
	A.BuildMatrix(params.vals, params.col_ind, params.nzeros_per_row, ctx, runtime);

	std::cout<<"Launche the CG solver..."<<std::endl;	
	std::cout<<std::endl;

	// run CG solver
	struct timespec t_start, t_end;
  	clock_gettime(CLOCK_MONOTONIC, &t_start);

	CGSolver<double> cgsolver;
	bool result = cgsolver.Solve(A, b, x, iter_max, 1e-4, ctx, runtime);

	clock_gettime(CLOCK_MONOTONIC, &t_end);

	if(result) {

		double time = ((1e3 * (t_end.tv_sec - t_start.tv_sec)) +
                       (1e-6 * (t_end.tv_nsec - t_start.tv_nsec)));

		std::cout<<"Elapsed time="<<std::setprecision(10)<<time<<" ms"<<std::endl;
	}
	else {
		std::cout<<"NO CONVERGENCE! :("<<std::endl;
	}
	std::cout<<std::endl;
	

	//print the solution
	//std::cout<<"SOLUTION:"<<std::endl;
	//x.PrintVals(ctx, runtime);
	//x.GiveNorm(params.exact, ctx, runtime);

	// destroy the objects
	x.DestroyArray(ctx, runtime);
	b.DestroyArray(ctx, runtime);
	A.DestroySpMatrix(ctx, runtime);
	
	return;
}

int main(int argc, char **argv){

	HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
   	
	HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
       	Processor::LOC_PROC, true/*single*/, false/*index*/,
        AUTO_GENERATE_ID, TaskConfigOptions(), "top_level_task");
       	
        // Register the callback function for creating custom mapper
  	HighLevelRuntime::set_registration_callback(mapper_registration);
	
	RegisterVectorTask<double>();

	RegisterOperatorTasks<double>();	
 
  return HighLevelRuntime::start(argc, argv);
}
