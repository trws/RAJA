#ifndef RAJA_internal_ForallNPolicy_HXX_
#define RAJA_internal_ForallNPolicy_HXX_

#include <RAJA/internal/LegacyCompatibility.hpp>
namespace RAJA
{

/******************************************************************
 *  ForallN generic policies
 ******************************************************************/

template <typename P, typename I>
struct ForallN_PolicyPair : public I {
  typedef P POLICY;
  typedef I ISET;

  RAJA_INLINE
  explicit
  RAJA_HOST_DEVICE
  constexpr ForallN_PolicyPair(ISET const &i) : ISET(i) {}
};

template <typename... PLIST>
struct ExecList {
  constexpr const static size_t num_loops = sizeof...(PLIST);
  typedef std::tuple<PLIST...> tuple;
};

// Execute (Termination default)
struct ForallN_Execute_Tag {
};

struct Execute {
  typedef ForallN_Execute_Tag PolicyTag;
};

template <typename EXEC, typename NEXT = Execute>
struct NestedPolicy {
  typedef NEXT NextPolicy;
  typedef EXEC ExecPolicies;
};


template <typename... POLICY_REST>
struct ForallN_Executor {
};

namespace detail {
/*!
 * \brief Functor that binds the first argument of a callable.
 *
 * This version has host-only constructor and host-device operator.
 */
template <typename BODY, typename... Indices>
struct ForallN_Bind_HostDevice {
  BODY const body;
  VarOps::tuple<Indices...> const is;

  RAJA_INLINE
  constexpr ForallN_Bind_HostDevice(BODY const &b, Indices... is)
      : body(b), is(VarOps::make_tuple(is...))
  {
  }

  template <typename... ARGS>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(ARGS... args) const
  {
      VarOps::invoke(body, is, args...);
  }
};

template <typename BODY, typename... Indices>
constexpr auto bindhd(BODY b, Indices ...is)
    -> ForallN_Bind_HostDevice<BODY, Indices...> {
    return ForallN_Bind_HostDevice<BODY, Indices...>{b, is...};
}


/*!
 * \brief Functor that binds the first argument of a callable.
 *
 * This version has host-only constructor and host-only operator.
 */
template <typename BODY, typename... Indices>
struct ForallN_Bind_Host {
  BODY const body;
  VarOps::tuple<Indices...> const is;

  RAJA_INLINE
  constexpr ForallN_Bind_Host(BODY const &b, Indices... is)
      : body(b), is(VarOps::make_tuple(is...))
  {
  }

  template <typename... ARGS>
  RAJA_INLINE void operator()(ARGS... args) const
  {
      VarOps::invoke(body, is, args...);
  }
};

template <typename BODY, typename... Indices>
constexpr auto bindh(BODY b, Indices ...is) -> ForallN_Bind_Host<BODY, Indices...> {
    return ForallN_Bind_Host<BODY, Indices...>{b, is...};
}

}

template <typename NextExec, typename BODY_in>
struct ForallN_PeelOuter {
  NextExec const next_exec;
  using BODY = typename std::remove_reference<BODY_in>::type;
  BODY const body;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr ForallN_PeelOuter(NextExec const &ne, BODY const &b)
      : next_exec(ne), body(b)
  {
  }

  // RAJA_SUPPRESS_HD_WARN
  template<typename ...Indices>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(Indices... is) const
  {
    using detail::bindhd;
    next_exec(bindhd(body, is...));
  }
};

}  // end of RAJA namespace

#endif
