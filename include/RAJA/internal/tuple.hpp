#ifndef RAJA_internal_tuple_HPP__
#define RAJA_internal_tuple_HPP__

/*!
 * \file
 *
 * \brief   Exceptionally basic tuple for host-device support
 */

#include <RAJA/internal/index_sequence.hpp>
#include <RAJA/util/defines.hpp>
#include <iostream>

namespace RAJA
{
namespace util
{

template <typename... Rest>
struct tuple;
template <size_t i, typename T>
struct tuple_element;

template <typename... Args>
RAJA_HOST_DEVICE constexpr auto make_tuple(Args... args) noexcept
    -> tuple<Args...>;

namespace internal
{
template <typename... Ts>
void ignore_args(Ts... args)
{
}
// Get nth element of parameter pack
template <size_t index, typename first, typename... rest>
struct get_type_at {
  static_assert(sizeof...(rest) + 1 > index, "index out of range");
  using type = typename get_type_at<index - 1, rest...>::type;
};

template <typename first, typename... rest>
struct get_type_at<0, first, rest...> {
  using type = first;
};

template <size_t index, typename... rest>
using get_type_at_t = typename get_type_at<index, rest...>::type;

template <size_t index, typename Type>
struct tuple_storage {
  RAJA_HOST_DEVICE constexpr tuple_storage(Type val) : val{val} {}

  RAJA_HOST_DEVICE
  constexpr const Type& get_inner() const noexcept { return val; }

  RAJA_CXX14_CONSTEXPR
  RAJA_HOST_DEVICE
  Type& get_inner() noexcept { return val; }

public:
  Type val;
};

template <typename... Types>
struct tuple_helper;
template <typename... Types, size_t... Indices>
struct tuple_helper<RAJA::util::index_sequence<Indices...>, Types...>
    : public internal::tuple_storage<Indices, Types>... {

  using Self = tuple_helper<RAJA::util::index_sequence<Indices...>, Types...>;
  RAJA_HOST_DEVICE constexpr tuple_helper(Types... args)
      : internal::tuple_storage<Indices, Types>(std::forward<Types>(args))...
  {
  }

  template <typename... RTypes>
  RAJA_HOST_DEVICE RAJA_CXX14_CONSTEXPR Self& operator=(
      const tuple_helper<RAJA::util::index_sequence<Indices...>, RTypes...>&
          rhs)
  {
    ignore_args((this->tuple_storage<Indices, Types>::get_inner() =
                     rhs.tuple_storage<Indices, RTypes>::get_inner())...);
    return *this;
  }
};
}

template <typename... Elements>
struct tuple
    : public internal::
          tuple_helper<RAJA::util::make_index_sequence<sizeof...(Elements)>,
                       Elements...> {
  using Self = tuple<Elements...>;
  using Base = internal::
      tuple_helper<RAJA::util::make_index_sequence<sizeof...(Elements)>,
                   Elements...>;

  // Constructors
  tuple() = default;
  tuple(tuple const&) = default;
  tuple(tuple&&) = default;
  tuple& operator=(tuple const& rhs) = default;
  tuple& operator=(tuple&& rhs) = default;

  template<typename... OtherTypes>
  RAJA_HOST_DEVICE constexpr explicit tuple(OtherTypes&&... rest)
      : Base{std::forward<OtherTypes>(rest)...}
  {
  }

  template <typename... RTypes>
  RAJA_HOST_DEVICE RAJA_CXX14_CONSTEXPR Self& operator=(
      const tuple<RTypes...>& rhs)
  {
    Base::operator=(rhs);
    return *this;
  }

  template <size_t index>
  RAJA_HOST_DEVICE auto get() noexcept
      -> internal::get_type_at_t<index, Elements..., void>
  {
    static_assert(sizeof...(Elements) > index, "index out of range");
    using ret_type = internal::get_type_at_t<index, Elements...>;
    using storage = internal::tuple_storage<index, ret_type>;
    return this->storage::get_inner();
  }
  template <size_t index>
  RAJA_HOST_DEVICE auto get() const noexcept
      -> const internal::get_type_at_t<index, Elements..., void>
  {
    static_assert(sizeof...(Elements) > index, "index out of range");
    using ret_type = internal::get_type_at_t<index, Elements...>;
    using storage = internal::tuple_storage<index, ret_type>;
    return this->storage::get_inner();
  }
};

template <size_t i, typename First, typename... Rest>
struct tuple_element<i, tuple<First, Rest...>>
    : tuple_element<i - 1, tuple<Rest...>> {
};

template <typename First, typename... Rest>
struct tuple_element<0, tuple<First, Rest...>> {
  using type = First;
};

template <int index, typename... Args>
RAJA_HOST_DEVICE constexpr auto get(const tuple<Args...>& t) noexcept
    -> internal::get_type_at_t<index, Args...>
{
  static_assert(sizeof...(Args) > index, "index out of range");
  using ret_type = internal::get_type_at_t<index, Args...>;
  using storage = internal::tuple_storage<index, ret_type>;
  return t.storage::get_inner();
}

template <typename Tuple>
struct tuple_size;

template <typename... Args>
struct tuple_size<tuple<Args...>> {
  static constexpr size_t value = sizeof...(Args);
};

template <typename... Args>
RAJA_HOST_DEVICE constexpr auto make_tuple(Args... args) noexcept
    -> tuple<Args...>
{
  return tuple<Args...>{std::forward<Args>(args)...};
}

template <class... Types>
RAJA_HOST_DEVICE constexpr tuple<Types&...> tie(Types&... args) noexcept
{
  return tuple<Types&...>{args...};
}

template <typename... Lelem, typename... Relem, size_t... Lidx, size_t... Ridx>
RAJA_HOST_DEVICE constexpr auto tuple_cat_pair(
    tuple<Lelem...>&& l,
    RAJA::util::index_sequence<Lidx...>,
    tuple<Relem...>&& r,
    RAJA::util::index_sequence<Ridx...>) noexcept -> tuple<Lelem..., Relem...>
{
  return make_tuple(get<Lidx>(l)..., get<Ridx>(r)...);
}
}
}

namespace internal
{
template <class Tuple, size_t... Idxs>
void print_tuple(std::ostream& os,
                 Tuple const& t,
                 RAJA::util::index_sequence<Idxs...>)
{
  RAJA::util::internal::ignore_args(
      (void*)&(os << (Idxs == 0 ? "" : ", ") << RAJA::util::get<Idxs>(t))...);
}
}

template <class... Args>
auto operator<<(std::ostream& os, RAJA::util::tuple<Args...> const& t)
    -> std::ostream&
{
  os << "(";
  internal::print_tuple(os,
                        t,
                        RAJA::util::make_index_sequence<sizeof...(Args)>{});
  return os << ")";
}


#endif /* RAJA_internal_tuple_HPP__ */
