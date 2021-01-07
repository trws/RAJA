
/* #define RAJA_ENABLE_TARGET_OPENMP 1 */
#include "include/RAJA/RAJA.hpp"
#include "include/RAJA/util/Timer.hpp"
#include <numeric>
#include <iostream>


// template<typename Op, typename T>
// struct Reducer {
//   using op = Op;
//   Reducer(T &target_in) : target(target_in), val(op::identity()) {}
//   T& target;
//   T val;
// };
//
// template<template<typename, typename, typename> class Op, typename T>
// auto Reduce(T &target) {
//   return Reducer<Op<T,T,T>, T>(target);
// }
//
// template<typename ...Params, typename T, camp::idx_t...Idx>
// void apply_combiner(T &l, T const &r, camp::idx_seq<Idx...>) {
// camp::sink((camp::get<Idx>(l) = typename Params::op{}(camp::get<Idx>(l), camp::get<Idx>(r)))...);
// }
//
// template <typename B, typename ...Params>
// void forall_param(int N,
//     B &&body, Params ...params) {
//   auto identity = camp::make_tuple(params.val...);
//
// #pragma omp declare reduction(combine: \
//     decltype(identity): \
//     apply_combiner<Params...>(omp_out, omp_in, camp::make_idx_seq_t<sizeof...(Params)>{})) \
//   initializer(omp_priv = camp::make_tuple(Params::op::identity()...))
//
// #<{(| #pragma omp target teams distribute parallel for simd reduction(combine: identity) |)}>#
// #pragma omp target teams distribute parallel for schedule(static, 1) reduction(combine: identity)
//     for (int i=0; i<N; ++i) {
//       camp::invoke(identity, [=](auto &...args) {
//             body(i, camp::forward<decltype(args)>(args)...);
//           });
//     }
//     camp::tie(params.target...) = identity;
// }

// template<typename ...Ts>
// void fmt(Ts...args) {
//   std::cout <<
// }

int main(int argc, char *argv[])
{
  constexpr int N = 50000000;
  double r = 0;
  double m = 5000;
  double ma = 0;
  double r_host = 0;
  double *a = new double[N]();
  double *b = new double[N]();
  std::iota(a, a+N, 0);
  std::iota(b, b+N, 0);

  RAJA::Timer t;
  t.start();
  /* forall_param(N, [=](int i, double &r, double &m, double &ma) { */
  /*     r += a[i] * b[i]; */
  /*     m = a[i] < m ? a[i] : m; */
  /*     ma = a[i] > m ? a[i] : m; */
  /*     }, */
  /*     Reduce<RAJA::operators::plus>(r), */
  /*     Reduce<RAJA::operators::minimum>(m), */
  /*     Reduce<RAJA::operators::maximum>(ma) */
  /*     ); */
  t.stop();

  RAJA::forall<RAJA::cuda_exec<128>>(RAJA::RangeSegment(0,N), [=] __device__ (int i) {
      });
  RAJA::ReduceSum<RAJA::cuda_reduce, double> rr(0);
  RAJA::ReduceMin<RAJA::cuda_reduce, double> rm(5000);
  RAJA::ReduceMax<RAJA::cuda_reduce, double> rma(0);

  RAJA::Timer rt;
  rt.start();
  RAJA::forall<RAJA::cuda_exec<128>>(RAJA::RangeSegment(0,N), [=] __device__ (int i) {
      rr += a[i] * b[i];
      rm.min(a[i]);
      rma.max(a[i]);
      });
  rt.stop();

  for(int i=0; i<N; ++i) {
    r_host += a[i] * b[i];
  }

  std::cout << r << " " << r_host << " " << t.elapsed() << std::endl;
  std::cout << m << " " << ma << std::endl;

  std::cout << rr.get() << " " << rt.elapsed() << std::endl;
  std::cout << rm.get() << " " << rma.get() << std::endl;

  std::cout << t.elapsed() << " " << rt.elapsed() << std::endl;



  return 0;
}
