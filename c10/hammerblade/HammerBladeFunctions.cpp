#include "HammerBladeFunctions.h"
#include <c10/probe/HBProfiler.h>

#include <mutex>
#include <string>

#include <time.h>
#include <iostream>

namespace c10 {
namespace hammerblade {

namespace {
static std::once_flag init_flag;
static void initHammerBladeDevice() {
  C10_HB_CHECK(hb_mc_device_init_custom_dimensions(&_hb_device, "HB_PYTORCH_PORT", 0, _hb_tg_dim));
  C10_HB_CHECK(hb_mc_device_program_init(&_hb_device, _bin_path, "default_allocator", 0));
  return;
}
} // namespace unnamed

DeviceIndex device_count() noexcept {
  // Assuming that we always have 1 HammerBlade manycore device
  int count = 1;
  // Lazy inialization
  std::call_once(init_flag, initHammerBladeDevice);
  return static_cast<DeviceIndex>(count);
}

DeviceIndex current_device() {
  int cur_device = -1;
  return static_cast<DeviceIndex>(cur_device);
}

void set_device(DeviceIndex device) {
  return; // no-op
}

 /* ------------------------------------------------------------------------------------
 * Interface to bsg_manycore runtime
 * -------------------------------------------------------------------------------------*/

eva_t device_malloc(size_t nbytes) {
  eva_t data_p;
  C10_HB_CHECK(hb_mc_device_malloc(&_hb_device, (uint32_t) nbytes, &data_p));
  return data_p;
}


void device_free(eva_t data_p) {
  C10_HB_CHECK(hb_mc_device_free(&_hb_device, data_p));
}


void* memcpy_host_to_device(void *dst, const void *src, uint32_t nbytes) {
  hb_mc_dma_htod_t job = {.d_addr=(eva_t)((intptr_t)dst), .h_addr=src, .size=nbytes};
  //C10_HB_CHECK(hb_mc_device_memcpy(&_hb_device, dst, src, nbytes, HB_MC_MEMCPY_TO_DEVICE));
  C10_HB_CHECK(hb_mc_device_dma_to_device(&_hb_device, &job, 1));
  return dst;
}


void* memcpy_device_to_host(void *dst, const void *src, uint32_t nbytes) {
  hb_mc_dma_dtoh_t job = {.d_addr=(eva_t)((intptr_t)src), .h_addr=dst, .size=nbytes};
  //C10_HB_CHECK(hb_mc_device_memcpy(&_hb_device, dst, src, nbytes, HB_MC_MEMCPY_TO_HOST));
  C10_HB_CHECK(hb_mc_device_dma_to_host(&_hb_device, &job, 1));
  return dst;
}


void offload_kernel(const char* kernel, std::vector<eva_t> args) {
  std::string kernel_str = "offload_kernel_";
  kernel_str += kernel;
  c10::probe::LogATenKernelWithName(kernel_str);

  eva_t* cuda_argv = (eva_t*) malloc(args.size() * sizeof(eva_t));
  if(!cuda_argv) {
    AT_ERROR("Falied to allocate cuda_argv!");
  }

  for(int i=0; i<args.size(); ++i) {
    cuda_argv[i] = args[i];
  }

  C10_HB_CHECK(hb_mc_kernel_enqueue(&_hb_device, _hb_grid_dim, _hb_tg_dim, kernel,
                                    args.size(), cuda_argv));

  C10_HB_CHECK(hb_mc_device_tile_groups_execute(&_hb_device));

  free(cuda_argv);
}


}} // namespace c10::hammerblade
