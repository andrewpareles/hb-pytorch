//====================================================================
// Vector - vector add kernel
// 06/02/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_vvadd(
          hb_tensor_t* result_p,
          hb_tensor_t* self_p,
          hb_tensor_t* other_p) {

    // Tutorial TODO:
    // Convert all low level pointers to Tensor objects
    HBTensor<float> result(result_p);
    HBTensor<float> self(self_p);
    HBTensor<float> other(other_p);

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // Use a single tile only
    if (__bsg_id == 0) {
      // Tutorial TODO:
      // add elements from self and other together -- put the result in result
      for (size_t i = 0; i < self.numel() - 1; i=i+2) {
        result(i) = self(i) + other(i);
        result(i+1) = self(i+1) + other(i+1);
      }
    }

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    // Sync
    g_barrier.sync();
    return 0;
  }

  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib_vvadd, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
