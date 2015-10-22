#ifndef ell_sparsematrix_hpp
#define ell_sparsematrix_hpp

#include <iostream>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"

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
  /***************newly added*********************/

};

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
#endif /*sparsematrix_hpp*/
