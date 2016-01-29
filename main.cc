#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <string>
#include "legion.h"

#include "params.hpp"
#include "legionvector.hpp"
#include "ell_sparsematrix.hpp"
#include "cgoperators.hpp"
#include "cgsolver.hpp"
#include "cgmapper.hpp"
#include <time.h>
#include "circuit.h"
using namespace LegionRuntime::HighLevel;

//enum TaskIDs {
//   TOP_LEVEL_TASK_ID = 0,
//};
void allocate_node_fields(Context ctx, HighLevelRuntime *runtime, FieldSpace node_space)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, node_space);
  allocator.allocate_field(sizeof(float), FID_NODE_CAP);
  runtime->attach_name(node_space, FID_NODE_CAP, "node capacitance");
  allocator.allocate_field(sizeof(float), FID_LEAKAGE);
  runtime->attach_name(node_space, FID_LEAKAGE, "leakage");
  allocator.allocate_field(sizeof(float), FID_CHARGE);
  runtime->attach_name(node_space, FID_CHARGE, "charge");
  allocator.allocate_field(sizeof(float), FID_NODE_VOLTAGE);
  runtime->attach_name(node_space, FID_NODE_VOLTAGE, "node voltage");

  allocator.allocate_field(sizeof(double), FID_NODE_VALUE);
  runtime->attach_name(node_space, FID_NODE_VALUE, "node value");
  allocator.allocate_field(sizeof(double), FID_NODE_RESULT);
  runtime->attach_name(node_space, FID_NODE_RESULT, "node result");
  allocator.allocate_field(sizeof(double), FID_NODE_OFFSET);
  runtime->attach_name(node_space, FID_NODE_OFFSET, "node offset");
}

void allocate_wire_fields(Context ctx, HighLevelRuntime *runtime, FieldSpace wire_space)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, wire_space);
  allocator.allocate_field(sizeof(ptr_t), FID_IN_PTR);
  runtime->attach_name(wire_space, FID_IN_PTR, "in_ptr");
  allocator.allocate_field(sizeof(ptr_t), FID_OUT_PTR);
  runtime->attach_name(wire_space, FID_OUT_PTR, "out_ptr");
  allocator.allocate_field(sizeof(PointerLocation), FID_IN_LOC);
  runtime->attach_name(wire_space, FID_IN_LOC, "in_loc");
  allocator.allocate_field(sizeof(PointerLocation), FID_OUT_LOC);
  runtime->attach_name(wire_space, FID_OUT_LOC, "out_loc");
  allocator.allocate_field(sizeof(float), FID_INDUCTANCE);
  runtime->attach_name(wire_space, FID_INDUCTANCE, "inductance");
  allocator.allocate_field(sizeof(float), FID_RESISTANCE);
  runtime->attach_name(wire_space, FID_RESISTANCE, "resistance");
  allocator.allocate_field(sizeof(float), FID_WIRE_CAP);
  runtime->attach_name(wire_space, FID_WIRE_CAP, "wire capacitance");
  for (int i = 0; i < WIRE_SEGMENTS; i++)
  {
    char field_name[10];
    allocator.allocate_field(sizeof(float), FID_CURRENT+i);
    sprintf(field_name, "current_%d", i);
    runtime->attach_name(wire_space, FID_CURRENT+i, field_name);
  }
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
  {
    char field_name[15];
    allocator.allocate_field(sizeof(float), FID_WIRE_VOLTAGE+i);
    sprintf(field_name, "wire_voltage_%d", i);
    runtime->attach_name(wire_space, FID_WIRE_VOLTAGE+i, field_name);
  }

  allocator.allocate_field(sizeof(double), FID_WIRE_VALUE);
  runtime->attach_name(wire_space, FID_WIRE_VALUE, "wire value");
  allocator.allocate_field(sizeof(int), FID_PIECE_NUM1);
  runtime->attach_name(wire_space, FID_PIECE_NUM1, "piece num1");
  allocator.allocate_field(sizeof(int), FID_PIECE_NUM2);
  runtime->attach_name(wire_space, FID_PIECE_NUM2, "piece num2");
}

void allocate_locator_fields(Context ctx, HighLevelRuntime *runtime, FieldSpace locator_space)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, locator_space);
  allocator.allocate_field(sizeof(PointerLocation), FID_LOCATOR);
  runtime->attach_name(locator_space, FID_LOCATOR, "locator");
}


void top_level_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime) {


  int64_t nx = 15; 
  int64_t nparts = 1;
  int iter_max = -1;
  std::string matrix_file;
  std::string rhs_file;
  bool inputmat = false;
  bool inputrhs = false;
  const InputArgs &command_args = HighLevelRuntime::get_input_args();

  int num_pieces = 4;
  
  // Parse command line arguments
  for (int i = 1; i < command_args.argc; i++)
  {
    if (!strcmp(command_args.argv[i], "-n"))
    {
      nx = atoi(command_args.argv[++i]);
      assert(nx > 0);
      continue;
    }
    if (!strcmp(command_args.argv[i], "-m"))
    {
      matrix_file = command_args.argv[++i];
      inputmat = true;
      continue;
    }
    if (!strcmp(command_args.argv[i], "-b"))
    {
      rhs_file = command_args.argv[++i];
      inputrhs = true;
      continue;
    }
    if (!strcmp(command_args.argv[i], "-max"))
    {
      iter_max = atoi(command_args.argv[++i]);
      assert(iter_max >= 0);
      continue;
    }
    if (!strcmp(command_args.argv[i], "-np"))
    {
      num_pieces = atoi(command_args.argv[++i]);
      assert(num_pieces >= 0);
      continue;
    }
  }
   	
	// get naprts from the custom mapper
	nparts = runtime->get_tunable_value(ctx, SUBREGION_TUNABLE, 0);
	
	Params<double> params;
	if(!inputmat && !inputrhs) {
		  params.Init(nx);
   		params.GenerateVals();
   	}
	else {	
	
		if(inputmat){
			assert(!matrix_file.empty());
			params.InitMat(matrix_file);
		}
		
		if(inputrhs){
			assert(!rhs_file.empty());
			params.InitRhs(rhs_file);
		}		
  }
		
	std::cout<<"Problem generation is done. Some properties are as follows:"<<std::endl;
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
	
	// build sparse matrix
	std::cout<<"Make sparse matrix..."<<std::endl;
	SpMatrix A(params.nrows, nparts, params.nonzeros, params.max_nzeros, ctx, runtime);
	A.BuildMatrix(params.vals, params.col_ind, params.nzeros_per_row, ctx, runtime);
  A.fill_dense_mat(ctx, runtime);
  A.dense_to_sparse();
  A.Print(ctx, runtime);
  //A.old_print();
  A.write_out();
  
  std::cout << "Executing the system call!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
  int r = system("mpirun -n 4 simpleGRAPH");
  std::cout << "After the system call!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
  if (r == 0);

  //int nodes_per_piece;   //Not useful actually
  Circuit &circuit = A.ckt;
  {
    int num_circuit_nodes = (int)A.mat.size();
    int num_circuit_wires = (int)A.sparse_mat.size();// This may make the code not working well
    //nodes_per_piece = (num_circuit_nodes % num_pieces == 0) ?  (num_circuit_nodes / num_pieces) : (num_circuit_nodes / num_pieces + 1);
    //printf("the wire number is %d\n",
    //calc_wire_num(sparse_mat));

    // Make index spaces
    IndexSpace node_index_space = runtime->create_index_space(ctx,num_circuit_nodes);
    runtime->attach_name(node_index_space, "node_index_space");
    IndexSpace wire_index_space = runtime->create_index_space(ctx,num_circuit_wires);
    runtime->attach_name(wire_index_space, "wire_index_space");
    // Make field spaces
    FieldSpace node_field_space = runtime->create_field_space(ctx);
    runtime->attach_name(node_field_space, "node_field_space");
    FieldSpace wire_field_space = runtime->create_field_space(ctx);
    runtime->attach_name(wire_field_space, "wire_field_space");
    FieldSpace locator_field_space = runtime->create_field_space(ctx);
    runtime->attach_name(locator_field_space, "locator_field_space");
    // Allocate fields
    allocate_node_fields(ctx, runtime, node_field_space);
    allocate_wire_fields(ctx, runtime, wire_field_space);
    allocate_locator_fields(ctx, runtime, locator_field_space);
    // Make logical regions
    circuit.all_nodes = runtime->create_logical_region(ctx,node_index_space,node_field_space);
    runtime->attach_name(circuit.all_nodes, "all_nodes");
    circuit.all_wires = runtime->create_logical_region(ctx,wire_index_space,wire_field_space);
    runtime->attach_name(circuit.all_wires, "all_wires");
    circuit.node_locator = runtime->create_logical_region(ctx,node_index_space,locator_field_space);
    runtime->attach_name(circuit.node_locator, "node_locator");

    std::cout << "the tree ids for circuit is: " << std::endl;
    std::cout << circuit.all_nodes.get_tree_id() << std::endl;
    std::cout << circuit.all_wires.get_tree_id() << std::endl;
    std::cout << circuit.node_locator.get_tree_id() << std::endl;

    std::cout << "the tree ids for A.ckt is: " << std::endl;

    std::cout << A.ckt.all_nodes.get_tree_id() << std::endl;
    std::cout << A.ckt.all_wires.get_tree_id() << std::endl;
    std::cout << A.ckt.node_locator.get_tree_id() << std::endl;
  }

  A.set_up_mat(ctx, runtime);

	// build unknown vector   
	std::cout<<"Make  unknown vector x..."<<std::endl;	
	Array<double> x(params.nrows, nparts, ctx, runtime); //TODO The partition here should be modified 

	//std::cout<<"Creating itr_read...."<<std::endl;	
  //IndexIterator itr_read(runtime, ctx, x.is);

	x.Initialize(ctx, runtime);
  //x.PrintVals(ctx, runtime);

	// build rhs vector
	std::cout<<"Make rhs vector..."<<std::endl;
	Array<double> b(params.nrows, nparts, ctx, runtime);
	std::cout<<"After making rhs vector..."<<std::endl;
	
	if(inputmat && !inputrhs) {	
		// fill rhs using random x vector
		Array<double> x_rand(params.nrows, nparts, ctx, runtime);
		x_rand.RandomInit(ctx, runtime);
		Predicate loop_pred = Predicate::TRUE_PRED;
		spmv(A, x_rand, b, loop_pred, ctx, runtime);
	}
	else {
	  std::cout<<" Inside the else..."<<std::endl;
	  // otherwise use the rhs array in params
		b.Initialize(params.rhs, ctx, runtime);	
	  std::cout<<"After Initializing b ..."<<std::endl;
	}
  //b.PrintVals(ctx, runtime);
		
	std::cout<<"Launch the CG solver..."<<std::endl;	
	std::cout<<std::endl;

	
	// run CG solver
	struct timespec t_start, t_end;
 	clock_gettime(CLOCK_MONOTONIC, &t_start);

	CGSolver<double> cgsolver;
  std::cout << "Start the CG Solver ..." << std::endl;
	bool result = cgsolver.Solve(A, b, x, iter_max, 1e-6, ctx, runtime);

	clock_gettime(CLOCK_MONOTONIC, &t_end);

	if(result) {

		double time = ((1e3 * (t_end.tv_sec - t_start.tv_sec)) + (1e-6 * (t_end.tv_nsec - t_start.tv_nsec)));

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

  std::cout<<"Exact solution:"<<std::endl;
  for(int i=0 ; i<params.nrows ; i++){
    std::cout<<params.exact[i]<<std::endl;
  }

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
  
  TaskHelper::register_cpu_variants<CalcNewCurrentsTask>();
       	
  // Register the callback function for creating custom mapper
  HighLevelRuntime::set_registration_callback(mapper_registration);
	
	RegisterVectorTask<double>();

	RegisterOperatorTasks<double>();	
 
  return HighLevelRuntime::start(argc, argv);
}

