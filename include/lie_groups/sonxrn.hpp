#pragma once

#include "lie_groups/lie_groups.h"
#include "lie_groups/so2.hpp"
#include "lie_groups/so3.hpp"

namespace lie_groups 
{

template <typename P, typename SOn>
class SOnxRn;

namespace internal
{

template <typename P, typename SOn>
struct traits<SOnxRn<P, SOn>>
{
  using SOnType = SOn;
  using TransType = Eigen::Matrix<P, SOn::ACT, 1>;
};

template <typename P, typename SOn>
struct traits<Map<SOnxRn<P, SOn>>>
{
  using SOnType = Map<SOn>;
  using TransType = Eigen::Map<Eigen::Matrix<P, SOn::ACT, 1>>;
};

template <typename P, typename SOn>
struct traits<Map<const SOnxRn<P, SOn>>>
{
  using SOnType = Map<const SOn>;
  using TransType = Eigen::Map<const Eigen::Matrix<P, SOn::ACT, 1>>;
};

}

// this class won't need to be used the way other Lie groups
// are, so don't inherit from LieGroup. Only some functions
// will be provided for convenience
// SOn should always be of Nonmap type
template <typename Derived, typename P, typename SOn>
class SOnxRnBase
{
public:
 // degrees of freedom
  static const int DoF = SOn::DoF + SOn::ACT;
  // amount of memory parameters
  static const int MEM = SOn::MEM + SOn::ACT;
  // size of space it acts on
  static const int ACT = SOn::ACT;

private:
  static const int N = SOn::ACT;

  using SOnType = typename internal::traits<Derived>::SOnType;
  using TransType = typename internal::traits<Derived>::TransType;

  using VectorDp = Eigen::Matrix<P, DoF, 1>;
  using VectorNp = Eigen::Matrix<P, N, 1>;

public:
  using TangentT = VectorDp;
  using JacT = Eigen::Matrix<P, DoF, DoF>;
  using NonmapT = SOnxRn<P, SOn>;

  SOnType const& rotation() const
  {
    return son();
  }

  TransType const& translation() const
  {
    return trans();
  }

  SOnxRn<P, SOn> inverse()
  {
    return SOnxRn<P, SOn>(son().inverse(), -trans());
  }

  static SOnxRn<P, SOn> Exp(VectorDp tau)
  {
    return SOnxRn<P, SOn>(SOn::Exp(tau.template tail<DoF-N>()), tau.template head<N>());
  }

  VectorDp Log()
  {
    return (VectorDp() << trans(), son().Log()).finished();
  }

  template <typename OtherDerived>
  SOnxRn<P, SOn> operator*(const SOnxRnBase<OtherDerived, P, SOn> &T) const
  {
    SOn R = son() * T.son();
    VectorNp t = trans() + T.trans();
    return SOnxRn<P, SOn>(R, t);
  }

  VectorNp operator*(const VectorNp &v)
  {
    return son_nonconst()*v + trans();
  }

  VectorNp operator*(const VectorNp &v) const
  {
    return son()*v + trans();
  }

  SOnxRn<P, SOn> operator+(const VectorDp &tau) const
  {
    return *this * Exp(tau);
  }

  static SOnxRn<P, SOn> random()
  {
    VectorNp rand = VectorNp::Random();
    rand *= 10;

    return SOnxRn<P, SOn>(SOn::random(), rand); 
  }

private:
  SOnType const& son() const
  {
    return static_cast<Derived const*>(this)->son();
  }

  SOnType& son_nonconst()
  {
    return static_cast<Derived*>(this)->son_nonconst();
  }

  TransType const& trans() const
  {
    return static_cast<Derived const*>(this)->trans();
  }

  // this is redundant, but we dont know which type Derived is
  friend class SOnxRnBase<SOnxRn<P, SOn>, P, SOn>;
  friend class SOnxRnBase<Map<SOnxRn<P, SOn>>, P, SOn>;
  friend class SOnxRnBase<Map<const SOnxRn<P, SOn>>, P, SOn>;

};


template <typename P, typename SOn>
class SOnxRn : public SOnxRnBase<SOnxRn<P, SOn>, P, SOn>
{
private:
  static const int N = SOn::ACT;
  using VectorNp = Eigen::Matrix<P, N, 1>;

public:
  using SOnT = SOn;
  using RnT = VectorNp;

  SOnxRn() :
    t_{VectorNp::Zero()}
  {}

  SOnxRn(SOn R, VectorNp t) :
    R_{R}, t_{t}
  {}

private:
  friend class SOnxRnBase<SOnxRn<P, SOn>, P, SOn>;
  friend class SOnxRnBase<Map<SOnxRn<P, SOn>>, P, SOn>;

  SOn const& son() const
  {
    return R_;
  }

  SOn& son_nonconst()
  {
    return R_;
  }

  VectorNp const& trans() const
  {
    return t_;
  }
  
  SOn R_;
  VectorNp t_;

};


template <typename P, typename SOn>
class Map<SOnxRn<P, SOn>> : public SOnxRnBase<Map<SOnxRn<P, SOn>>, P, SOn>
{
private:
  static const int N = SOn::ACT;
  using VectorNp = Eigen::Matrix<P, N, 1>;

public:
  using SOnT = Map<SOn>;
  using RnT = Eigen::Map<VectorNp>;

  Map(P *data) :
    t_{data}, R_{data == nullptr ? nullptr : data + N}
  {}

  Map<SOnxRn<P, SOn>>& operator=(const SOnxRn<P, SOn> &other)
  {
    R_ = other.rotation();
    t_ = other.translation();
    
    return *this;
  }

  P* data()
  {
    return t_.data();
  }

private:
  friend class SOnxRnBase<Map<SOnxRn<P, SOn>>, P, SOn>;
  friend class SOnxRnBase<SOnxRn<P, SOn>, P, SOn>;
  friend class SOnxRn<P, SOn>;

  Map<SOn> const& son() const
  {
    return R_;
  }

  Map<SOn>& son_nonconst()
  {
    return R_;
  }

  Eigen::Map<VectorNp> const& trans() const
  {
    return t_;
  }

  Map<SOn> R_;
  Eigen::Map<VectorNp> t_;

};


template <typename P, typename SOn>
class Map<const SOnxRn<P, SOn>> : public SOnxRnBase<Map<const SOnxRn<P, SOn>>, P, SOn>
{
private:
  static const int N = SOn::ACT;
  using VectorNp = Eigen::Matrix<P, N, 1>;

public:
  using SOnT = Map<const SOn>;
  using RnT = Eigen::Map<const VectorNp>;

  Map(const P *data) :
    t_{data}, R_{data + N}
  {}

  const P* data() const
  {
    return t_.data();
  }

private:
  friend class SOnxRnBase<Map<const SOnxRn<P, SOn>>, P, SOn>;

  Map<const SOn> const& son() const
  {
    return R_;
  }

  Eigen::Map<const VectorNp> const& trans() const
  {
    return t_;
  }

  const Map<const SOn> R_;
  const Eigen::Map<const VectorNp> t_;
};

typedef SOnxRn<double, SO2<double>> SO2xR2d;
typedef SOnxRn<double, SO3<double>> SO3xR3d;

} //namespace lie_groups