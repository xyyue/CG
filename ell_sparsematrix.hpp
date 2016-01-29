#ifndef ell_sparsematrix_hpp
#define ell_sparsematrix_hpp

#include <iostream>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"
#include "circuit.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace std;

#ifndef OFFSETS_ARE_DENSE
#define OFFSETS_ARE_DENSE
template<unsigned DIM, typename T>
static inline bool offsets_are_dense(const Rect<DIM> &bounds, const ByteOffset *offset)
{
  off_t exp_offset = sizeof(T);
  for (unsigned i = 0; i < DIM; i++) {
    bool found = false;
    for (unsigned j = 0; j < DIM; j++) {
      if (offset[j].offset == exp_offset) {
        found = true;
        exp_offset *= (bounds.hi[j] - bounds.lo[j] + 1);
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}
#endif

struct SparseElem {
  int x;
  int y;
  double z;
};

enum SPTaskIDs{
	SP_INIT_TASK_ID = 2,
};

enum SPFieldIDs{
	FID_Vals = 0,
	FID_Col_Ind = 1,
	FID_NZEROS_PER_ROW = 2,
};

void BuildMatrix_Task(const Task *task,
                      const std::vector<PhysicalRegion> &region,
                      Context ctx,
                      HighLevelRuntime *runtime);

class InitTaskArgs{

	public:
	int64_t nrows;
	int64_t max_nzeros;
	double *vals;
	int *col_ind;
	int *nzeros_per_row;

	InitTaskArgs(int64_t nrows, int64_t max_nzeros, const double* vals, const int* col_ind, const int* nzeros_per_row){
		
		this->nrows = nrows;
		this->max_nzeros = max_nzeros;
		
		// allocate memory
		this->vals = new double [nrows*max_nzeros];
		this->col_ind = new int[nrows*max_nzeros];
		this->nzeros_per_row = new int[nrows];

		// import input data
		for(int i=0; i < nrows; i++){
			
			this->nzeros_per_row[i] = nzeros_per_row[i];

			for(int j=0; j < max_nzeros; j++){
				
				this->vals[i*max_nzeros+j] = vals[i*max_nzeros+j];
				this->col_ind[i*max_nzeros+j] = col_ind[i*max_nzeros+j];
			}
		}
	}

	~InitTaskArgs(){

		if(vals){
			delete [] vals;
			vals = NULL;
}
		if(col_ind) {
			delete [] col_ind;
			col_ind = NULL;
		}

		if(nzeros_per_row) {
			delete [] nzeros_per_row;
			nzeros_per_row = NULL;
		}
	}

};

class SpMatrix{

	private:
	int64_t nparts;

	public:
	int64_t nrows;
	int64_t ncols;
	int64_t nonzeros;
	int64_t max_nzeros;
	FieldID row_fid;
	FieldID val_fid;
	FieldID col_fid;
	Domain color_domain;
	Rect<1> row_rect;
	Rect<1> elem_rect;
	IndexSpace row_is;
	IndexSpace elem_is;
	IndexPartition row_ip;
	IndexPartition elem_ip;
	FieldSpace row_fs;
	FieldSpace elem_fs;
	LogicalRegion row_lr;
	LogicalRegion elem_lr;
	LogicalPartition row_lp;
 	LogicalPartition elem_lp;
  /***************newly added*********************/
  std::vector<SparseElem> sparse_mat;
  std::vector<std::vector<double> > mat;
  Circuit ckt;
  std::vector<CircuitPiece> pieces;
  int num_pieces;
  Partitions partitions;


  /***************newly added*********************/

	SpMatrix(void);
	SpMatrix(int64_t n, int64_t nparts, int64_t nonzeros, int64_t max_nzeros, Context ctx, HighLevelRuntime *runtime);
	void DestroySpMatrix(Context ctx, HighLevelRuntime *runtime);
	void BuildMatrix(double *vals, int *col_ind, int *nzeros_per_row, 
			         Context ctx, HighLevelRuntime *runtime);
  void Print(Context ctx, HighLevelRuntime *runtime);

  /***************newly added*********************/
  void fill_dense_mat(Context ctx, HighLevelRuntime *runtime);
  void dense_to_sparse(void);
  void write_out(void);
  void old_print(void);
  void set_up_mat(Context ctx, HighLevelRuntime *runtime);
  template <typename T>
  void spmv(Array<T> &x, Array<T> &A_p, Context ctx, HighLevelRuntime *runtime);
  void print_nodes(Context ctx, HighLevelRuntime *runtime);
  /***************newly added*********************/

};


void SpMatrix::print_nodes(Context ctx, HighLevelRuntime *runtime) {
  
  RegionRequirement nodes_req(ckt.all_nodes, READ_WRITE, EXCLUSIVE, ckt.all_nodes);
  nodes_req.add_field(FID_NODE_VALUE);
  nodes_req.add_field(FID_NODE_RESULT);

  InlineLauncher nodes_launcher(nodes_req);
  PhysicalRegion nodes = runtime->map_region(ctx, nodes_req);

  nodes.wait_until_valid();
  RegionAccessor<AccessorType::Generic, double> fa_node_value = 
    nodes.get_field_accessor(FID_NODE_VALUE).typeify<double>();

  IndexIterator itr(runtime, ctx, ckt.all_nodes.get_index_space());
  for (int n = 0; n < nrows; n++)
  {
    assert(itr.has_next());
    ptr_t node_ptr = itr.next();
    double val = fa_node_value.read(node_ptr);
    printf("%d: %f\n", n, val);
  }
  printf("\n");

  runtime->unmap_region(ctx, nodes);
 
}

template <typename T>
void SpMatrix::spmv(Array<T> &x, Array<T> &A_p, Context ctx, HighLevelRuntime *runtime) {
  printf("Starting Sparse Matrix SPMV...\n");
  // inline map physical instances for the nodes and wire regions
  RegionRequirement nodes_req(ckt.all_nodes, READ_WRITE, EXCLUSIVE, ckt.all_nodes);
  nodes_req.add_field(FID_NODE_VALUE);
  nodes_req.add_field(FID_NODE_RESULT);
  //nodes_req.add_field(FID_NODE_OFFSET);

  RegionRequirement locator_req(ckt.node_locator, READ_WRITE, EXCLUSIVE, ckt.node_locator);
  locator_req.add_field(FID_LOCATOR);

  InlineLauncher nodes_launcher(nodes_req);
  PhysicalRegion nodes = runtime->map_region(ctx, nodes_req);
  InlineLauncher locator_launcher(locator_req);
  PhysicalRegion locator = runtime->map_region(ctx, locator_req);


  nodes.wait_until_valid();
  RegionAccessor<AccessorType::Generic, double> fa_node_value = 
    nodes.get_field_accessor(FID_NODE_VALUE).typeify<double>();
  //TODO
  //RegionAccessor<AccessorType::Generic, double> fa_node_result = 
  //  nodes.get_field_accessor(FID_NODE_RESULT).typeify<double>();
  //RegionAccessor<AccessorType::Generic, double> fa_node_offset = 
  //  nodes.get_field_accessor(FID_NODE_OFFSET).typeify<double>();


	RegionRequirement req(x.lr, READ_ONLY, EXCLUSIVE, x.lr);
  req.add_field(FID_X); //x.fid

  InlineLauncher init_launcher(req);
  PhysicalRegion init_region = runtime->map_region(ctx, init_launcher);
  init_region.wait_until_valid();

  RegionAccessor<AccessorType::Generic, T> acc_x =
    init_region.get_field_accessor(FID_X).typeify<T>();

  // For unstructured index space
  IndexIterator itr(runtime, ctx, ckt.all_nodes.get_index_space());

  Rect<1> rect(Point<1>(0), Point<1> (nrows - 1));
  GenericPointInRectIterator<1> pir(rect);

  std::cout << "The values are:" << std::endl;
  for (int n = 0; n < nrows; n++)
  {
    assert(itr.has_next());
    ptr_t node_ptr = itr.next();

    double val = acc_x.read(DomainPoint::from_point<1>(pir.p));
    //printf("%d: The val is: %f \n", n, val);

    fa_node_value.write(node_ptr, val);
  }
  std::cout << "The process is over!" << std::endl;


  runtime->unmap_region(ctx, init_region);
  runtime->unmap_region(ctx, nodes);
  runtime->unmap_region(ctx, locator);
       

  ArgumentMap local_args;
  for (int idx = 0; idx < num_pieces; idx++)
  {
    DomainPoint point = DomainPoint::from_point<1>(Point<1>(idx));
    local_args.set_point(point, TaskArgument(&(pieces[idx]),sizeof(CircuitPiece)));
  }

  // Make the launchers
  Rect<1> launch_rect(Point<1>(0), Point<1>(num_pieces-1)); 
  Domain launch_domain = Domain::from_rect<1>(launch_rect);

  Partitions parts = partitions;

  CalcNewCurrentsTask cnc_launcher(parts.pvt_wires, parts.pvt_nodes, parts.shr_nodes, parts.ghost_nodes, parts.inside_nodes,
                                   ckt.all_wires, ckt.all_nodes, launch_domain, local_args);

  bool simulation_success = true;

  std::cout << "Before dispatching ..." << std::endl;

  TaskHelper::dispatch_task<CalcNewCurrentsTask>(cnc_launcher, ctx, runtime, false, simulation_success, true);
                                                   
  std::cout << "After dispatching ..." << std::endl;



  
  //{
  //  printf("num_pieces is %d !!!\n", num_pieces);
  //  for (int i = 0; i < nrows; i++)
  //  {
  //      int idx = partition[i][j];
  //      printf("idx is %d\n", idx);
  //      ptr_t node_ptr = get_ith_ptr(runtime, ctx, ckt.all_nodes.get_index_space(), idx);
  //      pvt_ptrs[i].push_back(node_ptr);

  //      //fa_node_value.write(node_ptr, vec[idx]);
  //      fa_node_result.write(node_ptr, 0.0);
  //      //fa_node_offset.write(node_ptr, b[idx]);

  //      // Just put everything in everyones private map at the moment       
  //      // We'll pull pointers out of here later as nodes get tied to 
  //      // wires that are non-local
  //      private_node_map[i].points.insert(node_ptr); // The private nodes in a piece
  //      privacy_map[0].points.insert(node_ptr);      // All the private nodes
  //      locator_node_map[i].points.insert(node_ptr);
  //      printf("i = %d\n", i);
  //      printf("the size is %d\n", (int)piece_node_ptrs[i].size());
	//      //piece_node_ptrs[i].push_back(node_ptr);
  //      inside_node_map[i].points.insert(node_ptr);  // The private and shared nodes in a piece
  //  }
  //}

  


}
void SpMatrix::old_print(void) {
  std::cout << "Inside old_print():" << std::endl;
  
  printf("The dense matrix is:\n");
  for (unsigned int i = 0; i < mat.size(); i++)
  {
    for (unsigned int j = 0; j < mat.size(); j++)
      printf("%f ", mat[i][j]);
      printf("\n");
  }
  printf("\n\n\nThe sparse matrix is:\n\n");

  for (unsigned int i = 0; i < sparse_mat.size(); i++)
    printf("%d, %d, %f\n", sparse_mat[i].x, sparse_mat[i].y, sparse_mat[i].z);

}

void SpMatrix::write_out(void) {

  FILE * fp = fopen("GRAPH.txt","w+");
  int size = (int) mat.size();
  fprintf(fp, "%d\n", size);
  std::vector<std::vector<int> > nbr(size, std::vector<int>());
  int total_neighbor = 0;
  for (int i = 0; i < (int)sparse_mat.size(); i++)
  {
    int x = sparse_mat[i].x;
    int y = sparse_mat[i].y;
    if (x != y)
    {
      nbr[x].push_back(y);
      nbr[y].push_back(x);
      total_neighbor += 2;
    }
  }
  fprintf(fp, "%d\n", total_neighbor);

  for (int i = 0; i < size; i++)
  {
    fprintf(fp, "%d ", i);
    fprintf(fp, "%d ", (int)nbr[i].size());
    for (int j = 0; j < (int)nbr[i].size(); j++)
      fprintf(fp, "%d ", nbr[i][j]);

    fprintf(fp, "\n");
  }
  fclose(fp);
}

void SpMatrix::dense_to_sparse(void) {

  int size = mat.size();
  for (int i = 0; i < size; i++)
    for (int j = i; j < size; j++)
      if (mat[i][j] > 1e-5)
      {
        SparseElem temp;
        temp.x = i;
        temp.y = j;
        temp.z = mat[i][j];
        sparse_mat.push_back(temp);
      }
}

void SpMatrix::fill_dense_mat(Context ctx, HighLevelRuntime *runtime) {

  std::cout << "Inside fill_dense_mat():" << std::endl;
  std::vector<std::vector<double> > a(nrows, std::vector<double>(nrows, 0));
  mat = a;

  RegionRequirement req1(row_lr, READ_ONLY, EXCLUSIVE, row_lr);
  req1.add_field(FID_NZEROS_PER_ROW);

  RegionRequirement req2(elem_lr, READ_ONLY, EXCLUSIVE, elem_lr);
  req2.add_field(FID_Col_Ind);
  req2.add_field(FID_Vals);

	InlineLauncher init_launcher1(req1);
	PhysicalRegion init_region1 = runtime->map_region(ctx, init_launcher1);
	init_region1.wait_until_valid();

	RegionAccessor<AccessorType::Generic, int64_t> acc_num_nzeros =
    init_region1.get_field_accessor(FID_NZEROS_PER_ROW).typeify<int64_t>();

	InlineLauncher init_launcher2(req2);
  PhysicalRegion init_region2 = runtime->map_region(ctx, init_launcher2);
  init_region2.wait_until_valid();

  RegionAccessor<AccessorType::Generic, int64_t> acc_col =
    init_region2.get_field_accessor(FID_Col_Ind).typeify<int64_t>();

  RegionAccessor<AccessorType::Generic, double> acc_vals =
    init_region2.get_field_accessor(FID_Vals).typeify<double>();
    
  
  GenericPointInRectIterator<1> itr_row(row_rect);
  GenericPointInRectIterator<1> itr_elem(elem_rect);

  cout << "The Matrix is :" << endl;

  for(int i = 0; i < nrows; i++) {
    int64_t num = acc_num_nzeros.read(DomainPoint::from_point<1>(itr_row.p));
    int64_t cnt = 0;

    int64_t col = acc_col.read(DomainPoint::from_point<1>(itr_elem.p));
    double val = acc_vals.read(DomainPoint::from_point<1>(itr_elem.p));
    //cout << "#############" << "ROW " << i << " ################" << endl;

    for (int j = 0; j < ncols; j++) {
      if (j == col) {
        mat[i][j] = val;
        cout << val << " ";
        cnt++;
        if (cnt == num) {
          for (int k = 0; k < ncols - j - 1; k++)
            mat[i][j + k + 1] = 0;
            cout << 0 << " ";
          for (int k = 0; k < max_nzeros - num + 1; k++)
            itr_elem++;
          break;
        }

        itr_elem++;

        col = acc_col.read(DomainPoint::from_point<1>(itr_elem.p));
        val = acc_vals.read(DomainPoint::from_point<1>(itr_elem.p));
      }
      else
        mat[i][j] = 0;
        cout << 0 << " ";
    }

    itr_row++;
    cout << endl;
  }
}

void SpMatrix::Print(Context ctx, HighLevelRuntime *runtime) {
  std::cout << "Inside Print():" << std::endl;

  RegionRequirement req1(row_lr, READ_ONLY, EXCLUSIVE, row_lr);
  req1.add_field(FID_NZEROS_PER_ROW);

  RegionRequirement req2(elem_lr, READ_ONLY, EXCLUSIVE, elem_lr);
  req2.add_field(FID_Col_Ind);
  req2.add_field(FID_Vals);

	InlineLauncher init_launcher1(req1);
	PhysicalRegion init_region1 = runtime->map_region(ctx, init_launcher1);
	init_region1.wait_until_valid();

	RegionAccessor<AccessorType::Generic, int64_t> acc_num_nzeros =
    init_region1.get_field_accessor(FID_NZEROS_PER_ROW).typeify<int64_t>();

	InlineLauncher init_launcher2(req2);
  PhysicalRegion init_region2 = runtime->map_region(ctx, init_launcher2);
  init_region2.wait_until_valid();

  RegionAccessor<AccessorType::Generic, int64_t> acc_col =
    init_region2.get_field_accessor(FID_Col_Ind).typeify<int64_t>();

  RegionAccessor<AccessorType::Generic, double> acc_vals =
    init_region2.get_field_accessor(FID_Vals).typeify<double>();
    
  
  GenericPointInRectIterator<1> itr_row(row_rect);
  GenericPointInRectIterator<1> itr_elem(elem_rect);

  cout << "The Matrix is :" << endl;

  for(int i = 0; i < nrows; i++) {
    int64_t num = acc_num_nzeros.read(DomainPoint::from_point<1>(itr_row.p));
    int64_t cnt = 0;

    int64_t col = acc_col.read(DomainPoint::from_point<1>(itr_elem.p));
    double val = acc_vals.read(DomainPoint::from_point<1>(itr_elem.p));
    //cout << "#############" << "ROW " << i << " ################" << endl;
    //cout << "\n NUM == " << num << endl;
    //cout << "\n COL == " << col << endl;
    //cout << "\n VAL == " << val << endl;

    for (int j = 0; j < ncols; j++) {
      if (j == col) {
        cout << val << " ";
        cnt++;
        if (cnt == num) {
          for (int k = 0; k < ncols - j - 1; k++)
            cout << 0 << " ";
          for (int k = 0; k < max_nzeros - num + 1; k++)
            itr_elem++;
          break;
        }

        itr_elem++;

        col = acc_col.read(DomainPoint::from_point<1>(itr_elem.p));
        val = acc_vals.read(DomainPoint::from_point<1>(itr_elem.p));
        //cout << "\n COL == " << col << endl;
        //cout << "\n VAL == " << val << endl;
      }
      else
        cout << 0 << " ";
    }

    itr_row++;
    cout << endl;
  }
}

SpMatrix::SpMatrix(int64_t n, int64_t nparts, int64_t nonzeros, int64_t max_nzeros, Context ctx, HighLevelRuntime *runtime){

	this-> row_fid = FID_NZEROS_PER_ROW;	
	this-> val_fid = FID_Vals;
	this-> col_fid = FID_Col_Ind;
	this-> nrows = n;
	this-> ncols = n;
	this-> nonzeros = nonzeros;
	this-> max_nzeros = max_nzeros;
	this-> nparts = nparts;
	
	// build logical region for row_ptr
	row_rect = Rect<1>(Point<1>(0), Point<1>(nrows-1));
	row_is = runtime->create_index_space(ctx,
			  Domain::from_rect<1>(row_rect));
	row_fs = runtime->create_field_space(ctx);
	{
		FieldAllocator allocator = runtime->create_field_allocator(ctx, row_fs);
		allocator.allocate_field(sizeof(int64_t), FID_NZEROS_PER_ROW);
	}
	row_lr = runtime->create_logical_region(ctx, row_is, row_fs);

	// build logical region for matrix nonzero values
	elem_rect = Rect<1>(Point<1>(0), Point<1>(max_nzeros*nrows-1));
	elem_is = runtime->create_index_space(ctx,
			   Domain::from_rect<1>(elem_rect));
	elem_fs = runtime->create_field_space(ctx);
	{
		FieldAllocator allocator = runtime->create_field_allocator(ctx, elem_fs);
		allocator.allocate_field(sizeof(double), FID_Vals);
		allocator.allocate_field(sizeof(int64_t), FID_Col_Ind);
	}
	elem_lr = runtime->create_logical_region(ctx, elem_is, elem_fs);

	Rect<1> color_bounds(Point<1>(0), Point<1>(nparts-1));
	color_domain = Domain::from_rect<1>(color_bounds);

	// partition the row logical region
	DomainColoring row_coloring;
	int index = 0;
	const int local_num_rows = (nrows + nparts - 1) / nparts;
	for(int color = 0; color < nparts - 1; color++){
		assert((index + local_num_rows) <= nrows);
		Rect<1> subrect(Point<1>(index), Point<1>(index + local_num_rows - 1));
		row_coloring[color] = Domain::from_rect<1>(subrect);
		index += local_num_rows;
	}
	Rect<1> subrect(Point<1>(index), Point<1>(nrows-1));
        row_coloring[nparts-1] = Domain::from_rect<1>(subrect);

	row_ip = runtime->create_index_partition(ctx, row_is, color_domain,
						 row_coloring, true/*disjoint*/);
	row_lp = runtime->get_logical_partition(ctx, row_lr, row_ip);

	// partition the nonzero values logical region
	index = 0;
	DomainColoring elem_coloring;
	const int local_num_nzeros = local_num_rows * max_nzeros;
	for(int color = 0; color < nparts-1; color++){
		Rect<1> subrect1(Point<1>(index), Point<1>(index + local_num_nzeros -1));
		elem_coloring[color] = Domain::from_rect<1>(subrect1);
		index += local_num_nzeros;
	}
	Rect<1> subrect1(Point<1>(index), Point<1>(nrows*max_nzeros - 1));
	elem_coloring[nparts-1] = Domain::from_rect<1>(subrect1);

	elem_ip = runtime->create_index_partition(ctx, elem_is, color_domain,
						  elem_coloring, true/*disjoint*/); 	
	elem_lp = runtime->get_logical_partition(ctx, elem_lr, elem_ip);

}

void SpMatrix::DestroySpMatrix(Context ctx, HighLevelRuntime *runtime){
	
	runtime->destroy_logical_region(ctx, row_lr);
	runtime->destroy_logical_region(ctx, elem_lr);
	runtime->destroy_field_space(ctx, row_fs);
	runtime->destroy_field_space(ctx, elem_fs);
	runtime->destroy_index_space(ctx, row_is);
	runtime->destroy_index_space(ctx, elem_is);
}

void SpMatrix::BuildMatrix(double *vals, int *col_ind, int *nzeros_per_row,
                          Context ctx, HighLevelRuntime *runtime){

  RegionRequirement req1(row_lr, WRITE_DISCARD, EXCLUSIVE, row_lr);
  req1.add_field(FID_NZEROS_PER_ROW);

  RegionRequirement req2(elem_lr, WRITE_DISCARD, EXCLUSIVE, elem_lr);
  req2.add_field(FID_Col_Ind);
  req2.add_field(FID_Vals);


	InlineLauncher init_launcher1(req1);
	PhysicalRegion init_region1 = runtime->map_region(ctx, init_launcher1);
	init_region1.wait_until_valid();

	RegionAccessor<AccessorType::Generic, int64_t> acc_num_nzeros =
    init_region1.get_field_accessor(FID_NZEROS_PER_ROW).typeify<int64_t>();

	InlineLauncher init_launcher2(req2);
  PhysicalRegion init_region2 = runtime->map_region(ctx, init_launcher2);
  init_region2.wait_until_valid();

  RegionAccessor<AccessorType::Generic, int64_t> acc_col =
    init_region2.get_field_accessor(FID_Col_Ind).typeify<int64_t>();

  RegionAccessor<AccessorType::Generic, double> acc_vals =
    init_region2.get_field_accessor(FID_Vals).typeify<double>();

  Rect<1> subrect;
  ByteOffset offsets[1];

	// number of nonzeros in each row
  int64_t *num_nzeros_ptr = acc_num_nzeros.raw_rect_ptr<1>(row_rect, subrect, offsets);  // What is this for?
  if (!num_nzeros_ptr || (subrect != row_rect) ||
      !offsets_are_dense<1,int64_t>(row_rect, offsets))
  {
    GenericPointInRectIterator<1> itr(row_rect);

    for(int i=0; i<nrows; i++) {
      acc_num_nzeros.write(DomainPoint::from_point<1>(itr.p), nzeros_per_row[i]);
      itr++;
    }
  } 
  else
  {
    // Do the fast case
    for (int i = 0; i < nrows; i++) {
      num_nzeros_ptr[i] = nzeros_per_row[i];
    }
  }

  // nonzero values and column index
  int64_t *col_ind_ptr = acc_col.raw_rect_ptr<1>(elem_rect, subrect, offsets);
  Rect<1> subrect2;
  ByteOffset offsets2[1];
  double *val_ptr = acc_vals.raw_rect_ptr<1>(elem_rect, subrect2, offsets2);
  if (!col_ind_ptr || !val_ptr || (subrect != elem_rect) || 
      (subrect2 != elem_rect) ||
      !offsets_are_dense<1,int64_t>(elem_rect, offsets) ||
      !offsets_are_dense<1,double>(elem_rect, offsets2))
  {
    GenericPointInRectIterator<1> itr(elem_rect);

    for(int i=0; i<nrows * max_nzeros; i++){

      acc_col.write(DomainPoint::from_point<1>(itr.p), col_ind[i]);
      acc_vals.write(DomainPoint::from_point<1>(itr.p), vals[i]);
      itr++;
    }
  }
  else
  {
    for (int i = 0; i < (nrows * max_nzeros); i++) {
      col_ind_ptr[i] = col_ind[i];
      val_ptr[i] = vals[i];
    }
  }

	runtime->unmap_region(ctx, init_region1);
	runtime->unmap_region(ctx, init_region2);

	/*RegionRequirement req3(elem_lr, READ_ONLY, EXCLUSIVE, elem_lr);
        req3.add_field(FID_Vals);

	InlineLauncher init_launcher(req3);
        PhysicalRegion init_region = runtime->map_region(ctx, init_launcher);
        init_region.wait_until_valid();

	RegionAccessor<AccessorType::Generic, double> acc_num =
        init_region.get_field_accessor(FID_Vals).typeify<double>();

	int counter = 1;
	for(GenericPointInRectIterator<1> pir(elem_rect); pir; pir++) {
		double out = acc_num.read(DomainPoint::from_point<1>(pir.p));
		std::cout<<counter<<"  "<<out<<std::endl;
		counter++;
	}
	runtime->unmap_region(ctx, init_region);*/

	return;
}

// Register the initialization task
void RegisterSpMatrixTask(void){
	HighLevelRuntime::register_legion_task<BuildMatrix_Task>(SP_INIT_TASK_ID, 
						 Processor::LOC_PROC, 
						 true /*single*/,
						 true /*index*/); 
	return;
}

// Tasks for initialization sparse matrix
void BuildMatrix_Task(const Task *task,
		      const std::vector<PhysicalRegion> &regions,
		      Context ctx,
		      HighLevelRuntime *runtime){

	assert(regions.size() == 2);
	assert(task->regions.size() == 2);
 	
	const InitTaskArgs init_args = *((const InitTaskArgs*)task->args);

	RegionAccessor<AccessorType::Generic, int64_t> acc_num_nzeros = 
	regions[0].get_field_accessor(FID_NZEROS_PER_ROW).typeify<int64_t>();

	RegionAccessor<AccessorType::Generic, int64_t> acc_col = 
	regions[1].get_field_accessor(FID_Col_Ind).typeify<int64_t>();

	RegionAccessor<AccessorType::Generic, double> acc_vals = 
	regions[1].get_field_accessor(FID_Vals).typeify<double>();

	Domain row_dom = runtime->get_index_space_domain(ctx,
			 task->regions[0].region.get_index_space());
	Rect<1> row_rect = row_dom.get_rect<1>();

	Domain elem_dom = runtime->get_index_space_domain(ctx,
			  task->regions[1].region.get_index_space());
	Rect<1> elem_rect = elem_dom.get_rect<1>();

	// number of nonzeros in each row
	{
		GenericPointInRectIterator<1> itr(row_rect);

		for(int i=0; i<init_args.nrows; i++) {
			acc_num_nzeros.write(DomainPoint::from_point<1>(itr.p), init_args.nzeros_per_row[i]);
			itr++;
		}
	}

	// nonzero values and column index
	{
		GenericPointInRectIterator<1> itr(elem_rect);

		for(int i=0; i<init_args.nrows * init_args.max_nzeros; i++){
			acc_col.write(DomainPoint::from_point<1>(itr.p), init_args.col_ind[i]);
			acc_vals.write(DomainPoint::from_point<1>(itr.p), init_args.vals[i]);
			itr++;
		}
	}

	return;
}

static int get_next_line(FILE *fp, char *buf, int bufsize)
{
int i, cval, len;
char *c;

  while (1)
  {
    c = fgets(buf, bufsize, fp);

    if (c == NULL)
      return 0;  /* end of file */

    len = strlen(c);

    for (i=0, c=buf; i < len; i++, c++)
    {
      cval = (int)*c;
      if (isspace(cval) == 0) break;
    }
    if (i == len) continue;   /* blank line */
    if (*c == '#') continue;  /* comment */ 
    if (c != buf)
    {
      strcpy(buf, c);
    }
    break;
  }
  return strlen(buf); /* number * of * characters * */
}


static int get_next_int(FILE* fp)
{
  int value;
  char buf[512];
  int bufsize = 512;

  int num = get_next_line(fp, buf, bufsize);
  if (num == 0)
    return -1;
  sscanf(buf, "%d", &value);
  return value;
}

ptr_t get_ith_ptr(HighLevelRuntime *runtime, Context ctx, IndexSpace index_space, int i)
{
  IndexIterator itr(runtime, ctx, index_space);
  i++;
  ptr_t node_ptr;
  for (int j = 0; j < i; j++)
  {
    assert(itr.has_next());
    node_ptr = itr.next();
  }
  return node_ptr;
}

int get_piece_num(std::vector<std::vector<int> > &partition, int num)
{
  for (int i = 0; i < (int)partition.size(); i++)
    for (int j = 0; j < (int) partition[i].size(); j++) // Have made the change
      if (num == partition[i][j])
        return i;
  return -1;
}

PointerLocation find_location(ptr_t ptr, const std::set<ptr_t> &private_nodes,
                              const std::set<ptr_t> &shared_nodes, const std::set<ptr_t> &ghost_nodes)
{
  if (private_nodes.find(ptr) != private_nodes.end())
  {
    return PRIVATE_PTR;
  }
  else if (shared_nodes.find(ptr) != shared_nodes.end())
  {
    return SHARED_PTR;
  }
  else if (ghost_nodes.find(ptr) != ghost_nodes.end())
  {
    return GHOST_PTR;
  }
  // Should never make it here, if we do something bad happened
  assert(false);
  return PRIVATE_PTR;
}



void SpMatrix::set_up_mat(Context ctx, HighLevelRuntime *runtime) 

//Partitions load_circuit(Circuit &ckt, std::vector<CircuitPiece> &pieces, Context ctx,
//                        HighLevelRuntime *runtime, int &num_pieces, int nodes_per_piece,
//                        int random_seed, std::vector<SparseElem>&sparse_mat, 
//                        std::vector<double> &vec, std::vector<double> &b)
{
  printf("Initializing matrix multiplication...");
  // inline map physical instances for the nodes and wire regions
  RegionRequirement wires_req(ckt.all_wires, READ_WRITE, EXCLUSIVE, ckt.all_wires);
  wires_req.add_field(FID_IN_PTR);
  wires_req.add_field(FID_OUT_PTR);
  wires_req.add_field(FID_IN_LOC);
  wires_req.add_field(FID_OUT_LOC);
  wires_req.add_field(FID_WIRE_VALUE);
  wires_req.add_field(FID_PIECE_NUM1);
  wires_req.add_field(FID_PIECE_NUM2);

  RegionRequirement nodes_req(ckt.all_nodes, READ_WRITE, EXCLUSIVE, ckt.all_nodes);
  nodes_req.add_field(FID_NODE_VALUE);
  nodes_req.add_field(FID_NODE_RESULT);
  nodes_req.add_field(FID_NODE_OFFSET);

  RegionRequirement locator_req(ckt.node_locator, READ_WRITE, EXCLUSIVE, ckt.node_locator);
  locator_req.add_field(FID_LOCATOR);

  PhysicalRegion wires = runtime->map_region(ctx, wires_req);
  PhysicalRegion nodes = runtime->map_region(ctx, nodes_req);
  PhysicalRegion locator = runtime->map_region(ctx, locator_req);


  Coloring wire_owner_map;
  Coloring private_node_map;
  Coloring shared_node_map;
  Coloring ghost_node_map;
  Coloring locator_node_map;

  Coloring privacy_map;
  Coloring inside_node_map;
  privacy_map[0];
  privacy_map[1];

  FILE *fp = fopen("partition.txt", "r");
  num_pieces = get_next_int(fp);
  pieces.resize(num_pieces);
  // keep a O(1) indexable list of nodes in each piece for connecting wires
  std::vector<std::vector<ptr_t> > piece_node_ptrs(num_pieces); // Node ptrs for each piece
  std::vector<int> piece_shared_nodes(num_pieces, 0); // num of shared nodes in each piece

  int random_seed = 12345;
  srand48(random_seed);

  nodes.wait_until_valid();
  //RegionAccessor<AccessorType::Generic, double> fa_node_value = 
  //  nodes.get_field_accessor(FID_NODE_VALUE).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> fa_node_result = 
    nodes.get_field_accessor(FID_NODE_RESULT).typeify<double>();
  //RegionAccessor<AccessorType::Generic, double> fa_node_offset = 
  //  nodes.get_field_accessor(FID_NODE_OFFSET).typeify<double>();

  locator.wait_until_valid();
  RegionAccessor<AccessorType::Generic, PointerLocation> locator_acc = 
    locator.get_field_accessor(FID_LOCATOR).typeify<PointerLocation>();

  int num_nodes = nrows;

  ptr_t *first_nodes = new ptr_t[num_pieces];
  {
    IndexAllocator node_allocator = runtime->create_index_allocator(ctx, ckt.all_nodes.get_index_space());
    node_allocator.alloc(num_nodes);
  }
  // Write the values of the nodes.
  std::vector<std::vector<int> > partition(num_pieces, std::vector<int>());
  std::vector<std::vector<ptr_t> > pvt_ptrs(num_pieces);
  for (int i = 0; i < num_pieces; i++)
  {
    int num = get_next_int(fp);
    for (int j = 0; j < num; j++)
    {
      partition[i].push_back(get_next_int(fp));
    }
  }

  //for (int i = 0; i < num_pieces; i++)
  //{
  //  printf("\nthe %d piece:\n", i);
  //  for (int j = 0; j < (int)partition[i].size(); j++)
  //    printf("%d ", partition[i][j]);

  //}

  //TODO:: the first_nodes are useless now!!!!
  {
    printf("num_pieces is %d !!!\n", num_pieces);
    for (int i = 0; i < num_pieces; i++)
    {
      for (int j = 0; j < (int)partition[i].size(); j++)
      {
        int idx = partition[i][j];
        //printf("idx is %d\n", idx);
        ptr_t node_ptr = get_ith_ptr(runtime, ctx, ckt.all_nodes.get_index_space(), idx);
        pvt_ptrs[i].push_back(node_ptr);

        //fa_node_value.write(node_ptr, vec[idx]);
        fa_node_result.write(node_ptr, 0.0);
        //fa_node_offset.write(node_ptr, b[idx]);

        // Just put everything in everyones private map at the moment       
        // We'll pull pointers out of here later as nodes get tied to 
        // wires that are non-local
        private_node_map[i].points.insert(node_ptr); // The private nodes in a piece
        privacy_map[0].points.insert(node_ptr);      // All the private nodes
        locator_node_map[i].points.insert(node_ptr);
        //printf("i = %d\n", i);
        //printf("the size is %d\n", (int)piece_node_ptrs[i].size());
	      //piece_node_ptrs[i].push_back(node_ptr);
        inside_node_map[i].points.insert(node_ptr);  // The private and shared nodes in a piece
      }
    }
  }
  // verify the previous implementation
  // Print the node values for each piece 
  //IndexIterator itr(runtime, ctx, ckt.all_nodes.get_index_space());
  //  for (int n = 0; n < num_pieces; n++)
  //  {
  //    printf("node values for the %d th piece:\n", n);
  //    for (int i = 0; i < nodes_per_piece; i++)
  //    {
  //      int current = n * nodes_per_piece + i;
  //      if (current >= num_nodes)
  //        break;
  //      assert(itr.has_next());
  //      ptr_t node_ptr = itr.next();
  //      printf("%f ", fa_node_value.read(node_ptr));
  //    }
  //    printf("There are %d nodes in this piece\n", (int)piece_node_ptrs[n].size());
  //    printf("\n");
  //  }
  //printf("\n");

  wires.wait_until_valid();
  RegionAccessor<AccessorType::Generic, ptr_t> fa_wire_in_ptr = 
    wires.get_field_accessor(FID_IN_PTR).typeify<ptr_t>();
  RegionAccessor<AccessorType::Generic, ptr_t> fa_wire_out_ptr = 
    wires.get_field_accessor(FID_OUT_PTR).typeify<ptr_t>();
  RegionAccessor<AccessorType::Generic, PointerLocation> fa_wire_in_loc = 
    wires.get_field_accessor(FID_IN_LOC).typeify<PointerLocation>();
  RegionAccessor<AccessorType::Generic, PointerLocation> fa_wire_out_loc = 
    wires.get_field_accessor(FID_OUT_LOC).typeify<PointerLocation>();

  RegionAccessor<AccessorType::Generic, double> fa_wire_value = 
    wires.get_field_accessor(FID_WIRE_VALUE).typeify<double>();
  RegionAccessor<AccessorType::Generic, int> fa_piece_num1 = 
    wires.get_field_accessor(FID_PIECE_NUM1).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> fa_piece_num2 = 
    wires.get_field_accessor(FID_PIECE_NUM2).typeify<int>();

  ptr_t *first_wires = new ptr_t[num_pieces];
  // Allocate all the wires
  int num_wires = (int)sparse_mat.size(); 
  {
    IndexAllocator wire_allocator = runtime->create_index_allocator(ctx, ckt.all_wires.get_index_space());
    wire_allocator.alloc(num_wires);
  }

  {
    IndexIterator itr(runtime, ctx, ckt.all_wires.get_index_space());
    for (int i = 0; i < num_wires; i++)
    {
      assert(itr.has_next());
      ptr_t wire_ptr = itr.next();
      // Record the first wire pointer for this piece


      /******************newly added****************/

      //int m1 = sparse_mat[i].x / nodes_per_piece;
      //int n1 = sparse_mat[i].x % nodes_per_piece;
      //ptr_t p1 = piece_node_ptrs[m1][n1];
      ptr_t p1 = get_ith_ptr(runtime, ctx, ckt.all_nodes.get_index_space(), sparse_mat[i].x);
      fa_wire_in_ptr.write(wire_ptr, p1);


      //int m2 = sparse_mat[i].y / nodes_per_piece;
      //int n2 = sparse_mat[i].y % nodes_per_piece;
      //ptr_t p2 = piece_node_ptrs[m2][n2];
      ptr_t p2 = get_ith_ptr(runtime, ctx, ckt.all_nodes.get_index_space(), sparse_mat[i].y);
      fa_wire_out_ptr.write(wire_ptr, p2);

      fa_wire_value.write(wire_ptr, sparse_mat[i].z); 

      int m1 = get_piece_num(partition, sparse_mat[i].x);
      int m2 = get_piece_num(partition, sparse_mat[i].y);
      
      fa_piece_num1.write(wire_ptr, m1); // corresponding to in_ptr
      fa_piece_num2.write(wire_ptr, m2); // corresponding to out_ptr
      // These nodes are no longer private
      if (m1 != m2) // If the two nodes are in different pieces
      {
        privacy_map[0].points.erase(p1);
        privacy_map[0].points.erase(p2);
        privacy_map[1].points.insert(p1);
        privacy_map[1].points.insert(p2);
        ghost_node_map[m1].points.insert(p2);
        ghost_node_map[m2].points.insert(p1);
        wire_owner_map[m1].points.insert(wire_ptr);
        wire_owner_map[m2].points.insert(wire_ptr);
      }
      else
        wire_owner_map[m1].points.insert(wire_ptr);

      /************newly added**********************/

    }
  }

  // Second pass: make some random fraction of the private nodes shared
  {
    IndexIterator itr(runtime, ctx, ckt.all_nodes.get_index_space()); 
    for (int i = 0; i < num_pieces; i++)
    {
      for (int j = 0; j < (int)partition[i].size(); j++)
      {
        int idx = partition[i][j];
        ptr_t node_ptr = get_ith_ptr(runtime, ctx, ckt.all_nodes.get_index_space(), idx);

        if (privacy_map[0].points.find(node_ptr) == privacy_map[0].points.end()) // if shared
        {
          private_node_map[i].points.erase(node_ptr);
          // node is now shared
          shared_node_map[i].points.insert(node_ptr);
          locator_acc.write(node_ptr,SHARED_PTR); // node is shared 
        }
        else
        {
          locator_acc.write(node_ptr,PRIVATE_PTR); // node is private 
        }
      }
    }
  }
  // Second pass (part 2): go through the wires and update the locations // ////////////////////////////// This part is useless
  {
    IndexIterator itr(runtime, ctx, ckt.all_wires.get_index_space());
    for (int i = 0; i < num_wires; i++)
    {
      assert(itr.has_next());
      ptr_t wire_ptr = itr.next();
      ptr_t in_ptr = fa_wire_in_ptr.read(wire_ptr);
      ptr_t out_ptr = fa_wire_out_ptr.read(wire_ptr);

      // Find out which piece does the wire belong to.
      int piece_num = 0;
      for (int m = 0; m < (int)wire_owner_map.size(); m++)
        if (wire_owner_map[m].points.find(wire_ptr) != wire_owner_map[m].points.end())
        {
          piece_num = m;
          break;
        }
     // printf("piece_num = %d\n\n", piece_num);      
      fa_wire_in_loc.write(wire_ptr, 
          find_location(in_ptr, private_node_map[piece_num].points, 
            shared_node_map[piece_num].points, ghost_node_map[piece_num].points));     
      fa_wire_out_loc.write(wire_ptr, 
          find_location(out_ptr, private_node_map[piece_num].points, 
            shared_node_map[piece_num].points, ghost_node_map[piece_num].points));
    }
  }

  runtime->unmap_region(ctx, wires);
  runtime->unmap_region(ctx, nodes);
  runtime->unmap_region(ctx, locator);

  // Now we can create our partitions and update the circuit pieces

  // first create the privacy partition that splits all the nodes into either shared or private
  IndexPartition privacy_part = runtime->create_index_partition(ctx, ckt.all_nodes.get_index_space(), privacy_map, true/*disjoint*/);
  runtime->attach_name(privacy_part, "is_private");

  
  IndexSpace all_private = runtime->get_index_subspace(ctx, privacy_part, 0);
  runtime->attach_name(all_private, "private");
  IndexSpace all_shared  = runtime->get_index_subspace(ctx, privacy_part, 1);
  runtime->attach_name(all_shared, "shared");

  // Now create partitions for each of the subregions
  Partitions result;

  IndexPartition inside_part = runtime->create_index_partition(ctx, ckt.all_nodes.get_index_space(), inside_node_map, true/*disjoint*/);
  runtime->attach_name(inside_part, "inside_part");
  result.inside_nodes = runtime->get_logical_partition_by_tree(ctx, inside_part, ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id());
  runtime->attach_name(result.inside_nodes, "inside_nodes");

  IndexPartition priv = runtime->create_index_partition(ctx, all_private, private_node_map, true/*disjoint*/);
  runtime->attach_name(priv, "private");
  result.pvt_nodes = runtime->get_logical_partition_by_tree(ctx, priv, ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id());
  runtime->attach_name(result.pvt_nodes, "private_nodes");
  IndexPartition shared = runtime->create_index_partition(ctx, all_shared, shared_node_map, true/*disjoint*/);
  runtime->attach_name(shared, "shared");
  result.shr_nodes = runtime->get_logical_partition_by_tree(ctx, shared, ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id());
  runtime->attach_name(result.shr_nodes, "shared_nodes");
  IndexPartition ghost = runtime->create_index_partition(ctx, all_shared, ghost_node_map, false/*disjoint*/);
  runtime->attach_name(ghost, "ghost");
  result.ghost_nodes = runtime->get_logical_partition_by_tree(ctx, ghost, ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id());
  runtime->attach_name(result.ghost_nodes, "ghost_nodes");

  IndexPartition pvt_wires = runtime->create_index_partition(ctx, ckt.all_wires.get_index_space(), wire_owner_map, false/*disjoint*/);
  runtime->attach_name(pvt_wires, "private");
  result.pvt_wires = runtime->get_logical_partition_by_tree(ctx, pvt_wires, ckt.all_wires.get_field_space(), ckt.all_wires.get_tree_id()); 
  runtime->attach_name(result.pvt_wires, "private_wires");

  IndexPartition locs = runtime->create_index_partition(ctx, ckt.node_locator.get_index_space(), locator_node_map, true/*disjoint*/);
  runtime->attach_name(locs, "locs");
  result.node_locations = runtime->get_logical_partition_by_tree(ctx, locs, ckt.node_locator.get_field_space(), ckt.node_locator.get_tree_id());
  runtime->attach_name(result.node_locations, "node_locations");

  char buf[100];
  // Build the pieces
  for (int n = 0; n < num_pieces; n++)
  {
    pieces[n].pvt_nodes = runtime->get_logical_subregion_by_color(ctx, result.pvt_nodes, n);
    sprintf(buf, "private_nodes_of_piece_%d", n);
    runtime->attach_name(pieces[n].pvt_nodes, buf);
    pieces[n].shr_nodes = runtime->get_logical_subregion_by_color(ctx, result.shr_nodes, n);
    sprintf(buf, "shared_nodes_of_piece_%d", n);
    runtime->attach_name(pieces[n].shr_nodes, buf);
    pieces[n].ghost_nodes = runtime->get_logical_subregion_by_color(ctx, result.ghost_nodes, n);
    sprintf(buf, "ghost_nodes_of_piece_%d", n);
    runtime->attach_name(pieces[n].ghost_nodes, buf);
    pieces[n].pvt_wires = runtime->get_logical_subregion_by_color(ctx, result.pvt_wires, n);
    sprintf(buf, "private_wires_of_piece_%d", n);
    runtime->attach_name(pieces[n].pvt_wires, buf);

    pieces[n].num_wires = wire_owner_map[n].points.size();
    pieces[n].first_wire = first_wires[n];
    pieces[n].num_nodes = (int)pvt_ptrs[n].size();//piece_node_ptrs[n].size();
    pieces[n].first_node = first_nodes[n];
    pieces[n].node_ptrs = pvt_ptrs[n];

    pieces[n].dt = DELTAT;
    pieces[n].piece_num = n;
  }

  delete [] first_wires;
  delete [] first_nodes;

  printf("Finished initializing simulation...");

  partitions = result;
}

#endif /*sparsematrix_hpp*/
