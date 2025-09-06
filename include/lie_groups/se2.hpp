#pragma once

#include "lie_groups/lie_groups.h"
#include "lie_groups/so2.hpp"
#include <cmath>
// #include <pybind11/pybind11.h>
// #include <pybind11/eigen.h>
// #include <pybind11/operators.h>

namespace lie_groups 
{

template <typename P>
class SE2;

namespace internal
{

template <typename P>
struct traits<SE2<P>>
{
  using SO2Type = SO2<P>;
  using TransType = Eigen::Matrix<P, 2, 1>;
};

template <typename P>
struct traits<Map<SE2<P>>>
{
  using SO2Type = Map<SO2<P>>;
  using TransType = Eigen::Map<Eigen::Matrix<P, 2, 1>>;
};

template <typename P>
struct traits<Map<const SE2<P>>>
{
  using SO2Type = Map<const SO2<P>>;
  using TransType = Eigen::Map<const Eigen::Matrix<P, 2, 1>>;
};

} //namespace internal

template <typename Derived, typename P>
class SE2Base : public LieGroup<SE2<P>, SE2Base<Derived, P>, P, 3, 3>
{
private:
  using Matrix1p = Eigen::Matrix<P, 1, 1>;
  using Matrix2p = Eigen::Matrix<P, 2, 2>;
  using Vector2p = Eigen::Matrix<P, 2, 1>;
  using Matrix3p = Eigen::Matrix<P, 3, 3>;
  using Vector3p = Eigen::Matrix<P, 3, 1>;

  using SO2Type = typename internal::traits<Derived>::SO2Type;
  using TransType = typename internal::traits<Derived>::TransType;

public:
  using TangentT = Vector3p;
  using JacT = Matrix3p;
  using NonmapT = SE2<P>;

  // degrees of freedom
  static constexpr const int DoF = 3;
  // amount of memory parameters
  static constexpr const int MEM = 3;
  // size of space it acts on
  static constexpr const int ACT = 2;

  virtual Matrix3p matrix() override
  {
    Matrix3p T = Matrix3p::Identity();
    T.template topLeftCorner<2,2>() = so2_nonconst().matrix();
    T.template topRightCorner<2,1>() = trans();
    return T;
  }

  virtual Matrix3p matrix() const override
  {
    Matrix3p T = Matrix3p::Identity();
    T.template topLeftCorner<2,2>() = so2().matrix(); // this calls matrix() const
    T.template topRightCorner<2,1>() = trans();
    return T;
  }

  SO2Type const& rotation() const
  {
    return so2();
  }

  TransType const& translation() const
  {
    return trans();
  }

  virtual SE2<P> inverse() const override
  {
    return SE2<P>(so2().inverse(), -(so2().inverse()*trans()));
  }

  static SE2<P> Exp(Vector3p tau)
  {
    P theta = tau(2);
    return SE2<P>(SO2<P>::Exp((Matrix1p() << theta).finished()), V(theta)*tau.template head<2>());
  }

  virtual Vector3p Log() const override
  {
    P theta = so2().Log()(0);
    return (Vector3p() << V_inv(theta) * trans(), theta).finished();
  }

  virtual Matrix3p Ad() override
  {
    Matrix3p ret = Matrix3p::Identity();
    ret.template topLeftCorner<2,2>() = so2_nonconst().matrix();
    ret.template topRightCorner<2,1>() = -SO2<P>::hat(Matrix1p::Identity())*trans();
    return ret;
  }

  virtual Matrix3p Ad() const
  {
    Matrix3p ret = Matrix3p::Identity();
    ret.template topLeftCorner<2,2>() = so2().matrix();
    ret.template topRightCorner<2,1>() = -SO2<P>::hat(Matrix1p::Identity())*trans();
    return ret;
  }

  static Matrix3p hat(Vector3p tau)
  {
    Matrix3p ret = Matrix3p::Zero();
    ret.template topLeftCorner<2,2>() = SO2<P>::hat((Matrix1p() << tau(2)).finished());
    ret.template topRightCorner<2,1>() = tau.template head<2>();
    return ret;
  }

  static Vector3p vee(Matrix3p tau_hat)
  {
    P theta = SO2<P>::vee(tau_hat.template topLeftCorner<2,2>())(0);
    return (Vector3p() << tau_hat.template topRightCorner<2,1>(), theta).finished();
  }

  static Matrix3p ad(Vector3p tau)
  {
    // I'm pretty sure this is right
    return hat((Vector3p() << -SO2<P>::hat(Matrix1p::Identity())*tau.template head<2>(), tau(2)).finished());
  }

  static Matrix3p Jr(Vector3p tau)
  {
    return Jl(-tau);
  }

  static Matrix3p Jl(Vector3p tau)
  {
    Vector2p rho = tau.template head<2>();
    P theta = tau(2);
    wrap(theta);

    Matrix2p V_th = V(theta);
    Vector2p top_right;
    if(std::abs(theta) < SMALL_ANGLE)
    {
      top_right = -0.5*SO2<P>::hat(Matrix1p::Identity())*rho;
    }
    else
    {
      top_right = (-1.0/theta * (V_th - Matrix2p::Identity()))*rho;
    }
    Matrix3p ret = Matrix3p::Identity();
    ret.template topLeftCorner<2,2>() = V_th;
    ret.template topRightCorner<2,1>() = top_right;
    return ret;
  }

  static Matrix3p Jr_inv(Vector3p tau)
  {
    return Jl_inv(-tau);
  }

  static Matrix3p Jl_inv(Vector3p tau)
  {
    Vector2p rho = tau.template head<2>();
    P theta = tau(2);
    wrap(theta);
    
    Matrix2p V_th_inv = V_inv(theta);
    Vector2p top_right;
    if(std::abs(theta) < SMALL_ANGLE)
    {
      top_right = 0.5*SO2<P>::hat(Matrix1p::Identity())*rho;
    }
    else
    {
      top_right = (-1/theta * (V_th_inv - Matrix2p::Identity()))*rho;
    }
    Matrix3p ret = Matrix3p::Identity();
    ret.template topLeftCorner<2,2>() = V_th_inv;
    ret.template topRightCorner<2,1>() = top_right;
    return ret;
  }

  template <typename OtherDerived>
  SE2<P> operator*(const SE2Base<OtherDerived, P> &T) const
  {
    SO2<P> R = so2() * T.so2();
    Vector2p t = trans() + so2() * T.trans();
    return SE2<P>(R, t);
  }

  Vector2p operator*(const Vector2p &v)
  {
    return so2_nonconst()*v + trans();
  }

  Vector2p operator*(const Vector2p &v) const
  {
    return so2()*v + trans();
  }


  SE2<P> operator+(const Vector3p &tau) const
  {
    return *this * Exp(tau);
  }

  static SE2<P> random()
  {
    Vector3p rand = Vector3p::Random();
    rand(0) *= 10;
    rand(1) *= 10;
    rand(2) *= M_PI;

    return SE2<P>(rand);
  }

private:
  static Matrix2p V(P theta)
  {
    wrap(theta);

    if(std::abs(theta) < SMALL_ANGLE)
    {
      return Matrix2p::Identity();
    }
    return (Matrix2p() << std::sin(theta)/theta, -(1-std::cos(theta))/theta,
                                        (1-std::cos(theta))/theta, std::sin(theta)/theta).finished();
  }

  static Matrix2p V_inv(P theta)
  {
    wrap(theta);

    if(std::abs(theta) < SMALL_ANGLE)
    {
      return Matrix2p::Identity();
    }
    return 0.5*(Matrix2p() << theta*std::sin(theta)/(1-std::cos(theta)), theta,
                                            -theta, theta*std::sin(theta)/(1-std::cos(theta))).finished();
  }

  static void wrap(P &theta)
  {
    while(theta > M_PI)
      theta -= 2.0*M_PI;
    while(theta < -M_PI)
      theta += 2.0*M_PI;
  }

  SO2Type const& so2() const
  {
    return static_cast<Derived const*>(this)->so2();
  }

  SO2Type& so2_nonconst()
  {
    return static_cast<Derived*>(this)->so2_nonconst();
  }

  TransType const& trans() const
  {
    return static_cast<Derived const*>(this)->trans();
  }

  // this is redundant, but we dont know which type Derived is
  friend class SE2Base<SE2<P>, P>;
  friend class SE2Base<Map<SE2<P>>, P>;
  friend class SE2Base<Map<const SE2<P>>, P>;

};


template <typename P>
class SE2 : public SE2Base<SE2<P>, P>
{
private:
  using Matrix1p = Eigen::Matrix<P, 1, 1>;
  using Matrix2p = Eigen::Matrix<P, 2, 2>;
  using Vector2p = Eigen::Matrix<P, 2, 1>;
  using Matrix3p = Eigen::Matrix<P, 3, 3>;
  using Vector3p = Eigen::Matrix<P, 3, 1>;

public:
  using SOnT = SO2<P>;
  using RnT = Vector2p;

  SE2() :
    t_{Vector2p::Zero()}
  {}

  SE2(Matrix2p R, Vector2p t) :
    R_{SO2<P>(R)}, t_{t}
  {}

  SE2(SO2<P> R, Vector2p t) :
    R_{R}, t_{t}
  {}

  SE2(P theta, P x, P y) :
    R_{SO2<P>::Exp((Matrix1p() << theta).finished())}, t_{(Vector2p() << x, y).finished()}
  {}

  SE2(Vector3p tau)
  {
    *this = this->Exp(tau);
  }

  SE2(Matrix3p T) :
    t_{T.template topRightCorner<2,1>()}
  {
    Matrix2p R = T.template topLeftCorner<2,2>();
    R_ = SO2<P>(R);
  }

  SE2(const Map<SE2<P>> &T) :
    R_{T.so2()}, t_{T.trans()}
  {}

private:
  friend class SE2Base<SE2<P>, P>;
  friend class SE2Base<Map<SE2<P>>, P>;

  SO2<P> const& so2() const
  {
    return R_;
  }

  SO2<P>& so2_nonconst()
  {
    return R_;
  }

  Vector2p const& trans() const
  {
    return t_;
  }

  SO2<P> R_;
  Vector2p t_;
};


template <typename P>
class Map<SE2<P>> : public SE2Base<Map<SE2<P>>, P>
{
private:
  using Vector2p = Eigen::Matrix<P, 2, 1>;

public:
  using SOnT = Map<SO2<P>>;
  using RnT = Eigen::Map<Vector2p>;
  
  Map(P *data) :
    t_{data}, R_{data == nullptr ? nullptr : data + 2}
  {}

  Map<SE2<P>>& operator=(const SE2<P> &other)
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
  friend class SE2Base<Map<SE2<P>>, P>;
  friend class SE2Base<SE2<P>, P>;
  friend class SE2<P>;

  Map<SO2<P>> const& so2() const
  {
    return R_;
  }

  Map<SO2<P>>& so2_nonconst()
  {
    return R_;
  }

  Eigen::Map<Vector2p> const& trans() const
  {
    return t_;
  }

  Map<SO2<P>> R_;
  Eigen::Map<Vector2p> t_;
};


template <typename P>
class Map<const SE2<P>> : public SE2Base<Map<const SE2<P>>, P>
{
private:
  using Vector2p = Eigen::Matrix<P, 2, 1>;

public:
  using SOnT = Map<const SO2<P>>;
  using RnT = Eigen::Map<const Vector2p>;
  
  Map(const P *data) :
    t_{data}, R_{data + 2}
  {}

  const P* data() const
  {
    return t_.data();
  }

private:
  friend class SE2Base<Map<const SE2<P>>, P>;

  Map<const SO2<P>> const& so2() const
  {
    return R_;
  }

  Eigen::Map<const Vector2p> const& trans() const
  {
    return t_;
  }

  const Map<const SO2<P>> R_;
  const Eigen::Map<const Vector2p> t_;
};

typedef SE2<double> SE2d;
typedef SE2<float> SE2f;

} //namespace lie_groups