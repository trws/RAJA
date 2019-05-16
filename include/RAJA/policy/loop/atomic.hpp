/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining loop atomic operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_loop_atomic_HPP
#define RAJA_policy_loop_atomic_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/policy/sequential/atomic.hpp"

namespace RAJA
{
namespace atomic
{

using loop_atomic = seq_atomic;

}  // namespace atomic

}  // namespace RAJA

#endif  // guard
