#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Resize.h>
#include <ATen/hammerblade/HammerBladeContext.h>

namespace at {
namespace native {

Tensor llcopy_to_hb(const Tensor& self) {

  // get low level storage size
  size_t itemsize = self.storage().itemsize();
  int64_t numel = self.storage().numel();
  size_t storage_size = numel * itemsize;

  // alloc on HB
  c10::Allocator* allocator = at::hammerblade::getHammerBladeDeviceAllocator();

  // device tensor reconstruction
  auto storage_offset = self.storage_offset();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      self.dtype(),
      numel,
      allocator->allocate(storage_size),
      allocator,
      /*resizeable=*/true);
  auto tensor = detail::make_tensor<TensorImpl>(std::move(storage_impl), at::TensorTypeId::HammerBladeTensorId);
  setStrided(tensor, self.sizes(), self.strides(), storage_offset);

  // memcpy
  void* ptr = (void*)self.storage().data();
  void* hb_ptr = (void*)tensor.storage().data();
  c10::hammerblade::memcpy_host_to_device(hb_ptr, ptr, storage_size);

  return tensor;

}

}} // namespace at::native
