#ifndef RAJA_LEGACY_COMPATIBILITY_HXX
#define RAJA_LEGACY_COMPATIBILITY_HXX

#include "RAJA/config.hpp"
#include "RAJA/internal/index_sequence.hpp"
#include "RAJA/internal/tuple.hpp"
#include "RAJA/util/defines.hpp"

#if (!defined(__INTEL_COMPILER)) && (!defined(RAJA_COMPILER_MSVC))
static_assert(__cplusplus >= 201103L,
              "C++ standards below 2011 are not "
              "supported" RAJA_STRINGIFY_HELPER(__cplusplus));
#endif

#include <cstdint>
#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>

namespace VarOps
{

// TODO: clean up usages so these can be removed
using RAJA::util::index_sequence;
using RAJA::util::make_index_sequence;
using RAJA::util::tuple;
using RAJA::util::tuple_cat_pair;
using RAJA::util::get;
using RAJA::util::tuple_size;
using RAJA::util::make_tuple;

// Basics, using c++14 semantics in a c++11 compatible way, credit to libc++

// Forward
template <class T>
struct remove_reference {
  typedef T type;
};
template <class T>
struct remove_reference<T&> {
  typedef T type;
};
template <class T>
struct remove_reference<T&&> {
  typedef T type;
};
template <class T>
RAJA_HOST_DEVICE RAJA_INLINE constexpr T&& forward(
    typename remove_reference<T>::type& t) noexcept
{
  return static_cast<T&&>(t);
}
template <class T>
RAJA_HOST_DEVICE RAJA_INLINE constexpr T&& forward(
    typename remove_reference<T>::type&& t) noexcept
{
  return static_cast<T&&>(t);
}

// FoldL
template <typename Op, typename... Rest>
struct foldl_impl;

template <typename Op, typename Arg1>
struct foldl_impl<Op, Arg1> {
  using Ret = Arg1;
};

template <typename Op, typename Arg1, typename Arg2>
struct foldl_impl<Op, Arg1, Arg2> {
  using Ret = typename std::result_of<Op(Arg1, Arg2)>::type;
};

template <typename Op,
          typename Arg1,
          typename Arg2,
          typename Arg3,
          typename... Rest>
struct foldl_impl<Op, Arg1, Arg2, Arg3, Rest...> {
  using Ret =
      typename foldl_impl<Op,
                          typename std::result_of<Op(
                              typename std::result_of<Op(Arg1, Arg2)>::type,
                              Arg3)>::type,
                          Rest...>::Ret;
};

template <typename Op, typename Arg1>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto foldl(
    Op&& RAJA_UNUSED_ARG(operation),
    Arg1&& arg) -> typename foldl_impl<Op, Arg1>::Ret
{
  return forward<Arg1&&>(arg);
}

template <typename Op, typename Arg1, typename Arg2>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto foldl(Op&& operation,
                                                  Arg1&& arg1,
                                                  Arg2&& arg2) ->
    typename foldl_impl<Op, Arg1, Arg2>::Ret
{
  return forward<Op&&>(operation)(forward<Arg1&&>(arg1), forward<Arg2&&>(arg2));
}

template <typename Op,
          typename Arg1,
          typename Arg2,
          typename Arg3,
          typename... Rest>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto foldl(Op&& operation,
                                                  Arg1&& arg1,
                                                  Arg2&& arg2,
                                                  Arg3&& arg3,
                                                  Rest&&... rest) ->
    typename foldl_impl<Op, Arg1, Arg2, Arg3, Rest...>::Ret
{
  return foldl(forward<Op&&>(operation),
               forward<Op&&>(
                   operation)(forward<Op&&>(operation)(forward<Arg1&&>(arg1),
                                                       forward<Arg2&&>(arg2)),
                              forward<Arg3&&>(arg3)),
               forward<Rest&&>(rest)...);
}

struct adder {
  template <typename Result>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr Result operator()(
      const Result& l,
      const Result& r) const
  {
    return l + r;
  }
};

// Convenience folds
template <typename Result, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr Result sum(Args... args)
{
  return foldl(adder(), args...);
}

template <template <class...> class Seq, class First, class... Ints>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto rotate_left_one(
    const Seq<First, Ints...>) -> Seq<Ints..., First>
{
  return Seq<Ints..., First>{};
}
// Invoke
RAJA_SUPPRESS_HD_WARN 
template <typename Fn, size_t... Sequence, typename TupleLike, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto
invoke_with_order(Fn&& f,
                  TupleLike&& t,
                  index_sequence<Sequence...>,
                  Args... args) -> decltype(f(get<Sequence>(t)..., args...))
{
  return f(get<Sequence>(t)..., args...);
}

RAJA_SUPPRESS_HD_WARN 
template <typename Fn, typename TupleLike, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto invoke(
    Fn&& f,
    TupleLike&& t,
    Args... args)
    -> decltype(invoke_with_order(
        f,
        t,
        make_index_sequence<tuple_size<typename std::remove_cv<
            typename std::remove_reference<TupleLike>::type>::type>::value>{},
        args...))
{
  return invoke_with_order(
      f,
      t,
      make_index_sequence<tuple_size<typename std::remove_cv<
          typename std::remove_reference<TupleLike>::type>::type>::value>{},
      args...);
}

// Ignore helper
template <typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE void ignore_args(Args...)
{
}

// Assign

template <size_t... To, size_t... From, typename ToT, typename FromT>
RAJA_HOST_DEVICE RAJA_INLINE void assign(ToT&& dst,
                                         FromT src,
                                         index_sequence<To...>,
                                         index_sequence<From...>)
{
  ignore_args((dst[To] = src[From])...);
}

template <size_t... To, typename ToT, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE void assign_args(ToT&& dst,
                                              index_sequence<To...>,
                                              Args... args)
{
  ignore_args((dst[To] = args)...);
}

// Get nth element of parameter pack
template <size_t index, size_t first, size_t... rest>
struct get_at {
  static constexpr size_t value = get_at<index - 1, rest...>::value;
};

template <size_t first, size_t... rest>
struct get_at<0, first, rest...> {
  static constexpr size_t value = first;
};

// Get nth element of parameter pack
template <size_t index, typename first, typename... rest>
struct get_type_at {
  using type = typename get_type_at<index - 1, rest...>::type;
};

template <typename first, typename... rest>
struct get_type_at<0, first, rest...> {
  using type = first;
};

// Get offset of element of parameter pack
template <size_t diff, size_t off, size_t match, size_t... rest>
struct get_offset_impl {
  static constexpr size_t value =
      get_offset_impl<match - get_at<off + 1, rest...>::value,
                      off + 1,
                      match,
                      rest...>::value;
};

template <size_t off, size_t match, size_t... rest>
struct get_offset_impl<0, off, match, rest...> {
  static constexpr size_t value = off;
};

template <size_t match, size_t first, size_t... rest>
struct get_offset
    : public get_offset_impl<match - first, 0, match, first, rest...> {
};

// Get nth element of argument list
// TODO: add specializations to make this compile faster and with less
// recursion
template <size_t index>
struct get_arg_at {
  template <typename First, typename... Rest>
  RAJA_HOST_DEVICE RAJA_INLINE static constexpr auto value(
      First&& RAJA_UNUSED_ARG(first),
      Rest&&... rest)
      -> decltype(VarOps::forward<
                  typename VarOps::get_type_at<index - 1, Rest...>::type>(
          get_arg_at<index - 1>::value(VarOps::forward<Rest>(rest)...)))
  {
    static_assert(index < sizeof...(Rest) + 1, "index is past the end");
    return VarOps::forward<
        typename VarOps::get_type_at<index - 1, Rest...>::type>(
        get_arg_at<index - 1>::value(VarOps::forward<Rest>(rest)...));
  }
};

template <>
struct get_arg_at<0> {
  template <typename First, typename... Rest>
  RAJA_HOST_DEVICE RAJA_INLINE static constexpr auto value(
      First&& first,
      Rest&&... RAJA_UNUSED_ARG(rest))
      -> decltype(VarOps::forward<First>(first))
  {
    return VarOps::forward<First>(first);
  }
};
}

#endif
