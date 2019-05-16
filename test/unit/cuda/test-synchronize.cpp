//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

#include "RAJA_gtest.hpp"

CUDA_TEST(SynchronizeTest, CUDA)
{

  double* managed_data;
  cudaMallocManaged(&managed_data, sizeof(double) * 50);

  RAJA::forall<RAJA::cuda_exec_async<256>>( RAJA::RangeSegment(0, 50),
    [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    managed_data[i] = 1.0 * i;
  });
  RAJA::synchronize<RAJA::cuda_synchronize>();

  RAJA::forall<RAJA::loop_exec>( RAJA::RangeSegment(0, 50),
    [=](RAJA::Index_type i) {
    EXPECT_EQ(managed_data[i], 1.0 * i);
  });

  cudaFree(managed_data);
}
