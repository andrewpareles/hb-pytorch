#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/SparseTensorUtils.h>
#include <c10/util/Optional.h>

namespace at {
namespace native {

template <typename scalar_t>
void inline sddmm_kernel(
    const Tensor& a_sparse,
    const Tensor& b_dense,
    const Tensor& c_dense,
    Tensor& out_tensor) {

    int* a_idx = a_sparse._indices().data_ptr<int32_t>();
    for (int i = 0; i < a_sparse.numel(); i++){
      printf("a_idx(%i)=%i", i, a_idx[i]);
    }
    scalar_t* b = b_dense.data_ptr<scalar_t>();
    scalar_t* c = c_dense.data_ptr<scalar_t>();
    scalar_t* out = out_tensor.data_ptr<scalar_t>();


    int dot_len = b_dense.size(1);
    for (int k = 0; k < a_sparse._nnz(); k++) {
      int ai = a_idx[k];//0
      int aj = a_idx[k];  //1

      float dot_total = 0;
      for (int i = 0; i < dot_len; i++)
        dot_total += b[ai*i] * c[i*aj];
        
      out[ai*aj] = dot_total;
    }
}

Tensor sddmm_cpu(
    const Tensor& a_sparse,
    const Tensor& b_dense,
    const Tensor& c_dense) {
  Tensor out_dense = at::zeros(a_sparse.sizes(), {at::requires_grad().device(at::kCPU).dtype(at::kFloat)});

  AT_DISPATCH_ALL_TYPES(a_sparse.scalar_type(), "sddmm_cpu", [&] {
    sddmm_kernel<scalar_t>(a_sparse, b_dense, c_dense, out_dense);
  });

  return out_dense;
}

} // namespace native
} // namespace at