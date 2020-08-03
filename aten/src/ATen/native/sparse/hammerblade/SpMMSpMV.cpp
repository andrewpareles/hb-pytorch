#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

using namespace at::sparse;

IntTensor _to_csr_int( const IntTensor& rowIndices, int64_t dim, int64_t nnz) {

  TORCH_CHECK(rowIndices.is_hammerblade(), "row Indices should be on hammerblade");

  IntTensor csr = at::zeros({dim + 1}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});

  uint32_t dim_uint = (uint32_t)(dim);
  uint32_t nnz_uint = (uint32_t)(nnz);
  hb_offload_kernel(csr, rowIndices, dim_uint, nnz_uint, "tensorlib_coo_to_csr");

/*
  int32_t* indices = rowIndices.data_ptr<int32_t>();

  if (nnz > 0) {
    auto csr_accessor = csr.accessor<int32_t, 1>();
    //Convert the sparse matrix to CSR format
    at::parallel_for(0, nnz, 10000, [&](int64_t start, int64_t end) {
      int64_t h, hp0, hp1;
      for (auto i = start; i < end; i++) {
        hp0 = indices[i];
        hp1 = (i+1 == nnz) ?  dim : indices[i+1];
        if (hp0 != hp1) for (h = hp0; h < hp1; h++) {
          csr_accessor[h+1] = i+1;
        }
      }
    });
  }
*/
  return csr;
}

Tensor _sparse_mm_hb(const SparseTensor& sparse, const Tensor& dense) {

  TORCH_CHECK(sparse.is_hammerblade(), "SpMM: expected 'mat1' to be a HammerBlade tensor");
  TORCH_CHECK(dense.is_hammerblade(), "SpMM: expected 'mat2' to be a HammerBlade tensor");

  if ( (sparse.scalar_type() != ScalarType::Float)
    || (dense.scalar_type() != ScalarType::Float) ) {
    AT_ERROR("HammerBlade SpMM is implemented for Float only"); 
  }
  
  using scalar_t = float;
 
  TORCH_CHECK(sparse.sparse_dim() == 2, "We do not support hybrid sparse tensor for HammerBlade sparse mm !");
  TORCH_CHECK(sparse.dim() == 2 && sparse.dim() == 2, "2D matrix expected, got ", sparse.dim(), " and ", dense.dim(), " tensors");
  TORCH_CHECK(sparse.size(1) == dense.size(0), "Argument #2: Expected dim 0 size ", sparse.size(1), ", got ", dense.size(0));

  int64_t nnz = sparse._nnz();
  int64_t dim = sparse.size(0);

  IntTensor indices = sparse._indices();
  TORCH_CHECK(indices.dtype() == at::kInt, "Indices on HammerBlade should be int32");
  IntTensor colIndices = indices.select(0, 1);
  TORCH_CHECK(colIndices.is_hammerblade(), "colIndices show be HammerBlade Tensor");
  IntTensor rowIndices = indices.select(0, 0);
  IntTensor csr_hb = _to_csr_int(rowIndices, dim, nnz);
//  IntTensor csr_hb = at::empty({csr.size(0)}, {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kInt)});
//  csr_hb.copy_(csr);
  
  Tensor values = sparse._values();

  Tensor result = at::zeros({sparse.size(0), dense.size(1)}, {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kFloat)});

  hb_offload_kernel(result, csr_hb, colIndices, values, dense, "tensorlib_sparse_dense_mm");
  return result;
}

Tensor mv_hb_sparse(const SparseTensor& sparse, const Tensor& dense) {
  TORCH_CHECK(sparse.is_hammerblade(), "SpMV: expected 'mat1' to be a HammerBlade tensor");
  TORCH_CHECK(dense.is_hammerblade(), "SpMV: expected 'mat2' to be a HammerBlade tensor");

  if ( (sparse.scalar_type() != ScalarType::Float)
    || (dense.scalar_type() != ScalarType::Float) ) {
    AT_ERROR("HammerBlade SpMV is implemented for Float only");
  }

  using scalar_t = float;

  TORCH_CHECK(sparse.sparse_dim() == 2, "We do not support hybrid sparse tensor for HammerBlade SpMV !");
  TORCH_CHECK(sparse.dim() == 2 && sparse.dim() == 2, "2D matrix expected, got ", sparse.dim(), " and ", dense.dim(), " tensors");
  TORCH_CHECK(dense.dim() == 1, "Argument #2: Expected vector, got dim", dense.dim()); 
  TORCH_CHECK(sparse.size(1) == dense.size(0), "Argument #2: Expected dim 0 size ", sparse.size(1), ", got ", dense.size(0));
  
  int64_t nnz = sparse._nnz();
  int64_t dim = sparse.size(0);

  IntTensor indices = sparse._indices();
  TORCH_CHECK(indices.dtype() == at::kInt, "Indices on HammerBlade should be int32");
  IntTensor colIndices = indices.select(0, 1);
  TORCH_CHECK(colIndices.is_hammerblade(), "colIndices show be HammerBlade Tensor");
  IntTensor rowIndices = indices.select(0, 0);
  IntTensor csr_hb = _to_csr_int(rowIndices, dim, nnz);

  Tensor values = sparse._values();

  Tensor result = at::zeros({dim}, {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kFloat)});

  hb_offload_kernel(result, csr_hb, colIndices, values, dense, "tensorlib_spmv");
  return result;
}   
}}
