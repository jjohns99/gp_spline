#pragma once

#include <cmath>
#include "lie_groups/lie_groups.h"
#include "lie_groups/so3.hpp"
// #include <pybind11/pybind11.h>
// #include <pybind11/eigen.h>
// #include <pybind11/operators.h>

namespace lie_groups
{

template <typename P>
class SE3;

namespace internal
{

template <typename P>
struct traits<SE3<P>>
{
  using SO3Type = SO3<P>;
  using TransType = Eigen::Matrix<P, 3, 1>;
};

template <typename P>
struct traits<Map<SE3<P>>>
{
  using SO3Type = Map<SO3<P>>;
  using TransType = Eigen::Map<Eigen::Matrix<P, 3, 1>>;
};

template <typename P>
struct traits<Map<const SE3<P>>>
{
  using SO3Type = Map<const SO3<P>>;
  using TransType = Eigen::Map<const Eigen::Matrix<P, 3, 1>>;
};

} //namespace internal


template <typename Derived, typename P>
class SE3Base : public LieGroup<SE3<P>, SE3Base<Derived, P>, P, 4, 6>
{
private:
  using Matrix3p = Eigen::Matrix<P, 3, 3>;
  using Vector3p = Eigen::Matrix<P, 3, 1>;
  using Matrix4p = Eigen::Matrix<P, 4, 4>;
  using Matrix6p = Eigen::Matrix<P, 6, 6>;
  using Vector6p = Eigen::Matrix<P, 6, 1>;

  using SO3Type = typename internal::traits<Derived>::SO3Type;
  using TransType = typename internal::traits<Derived>::TransType;

public:
  using TangentT = Vector6p;
  using JacT = Matrix6p;
  using NonmapT = SE3<P>;

  // degrees of freedom
  static constexpr const int DoF = 6;
  // amount of memory parameters
  static constexpr const int MEM = 7;
  // size of space it acts on
  static constexpr const int ACT = 3;

  virtual Matrix4p matrix() override
  {
    Matrix4p T = Matrix4p::Identity();
    T.template topLeftCorner<3,3>() = so3_nonconst().matrix();
    T.template topRightCorner<3,1>() = trans();
    return T;
  }

  virtual Matrix4p matrix() const override
  {
    Matrix4p T = Matrix4p::Identity();
    T.template topLeftCorner<3,3>() = so3().matrix(); // this calls matrix() const
    T.template topRightCorner<3,1>() = trans();
    return T;
  }

  SO3Type const& rotation() const
  {
    return so3();
  }

  TransType const& translation() const
  {
    return trans();
  }

  virtual SE3<P> inverse() const override
  {
    return SE3<P>(so3().inverse(), -(so3().inverse()*trans()));
  }

  static SE3<P> Exp(Vector6p tau)
  {
    Vector3p rho = tau.template head<3>();
    Vector3p theta = tau.template tail<3>();

    return SE3<P>(SO3<P>::Exp(theta), SO3<P>::Jl(theta)*rho);
  }

  virtual Vector6p Log() const override
  {
    Vector3p theta = so3().Log();
    return (Vector6p() << SO3<P>::Jl_inv(theta) * trans(), theta).finished();
  }

  virtual Matrix6p Ad() override
  {
    Matrix3p R = so3_nonconst().matrix();
    Matrix6p ret = Matrix6p::Zero();
    ret.template topLeftCorner<3,3>() = R;
    ret.template topRightCorner<3,3>() = SO3<P>::hat(trans()) * R;
    ret.template bottomRightCorner<3,3>() = R;

    return ret;
  }

  virtual Matrix6p Ad() const
  {
    Matrix3p R = so3().matrix();
    Matrix6p ret = Matrix6p::Zero();
    ret.template topLeftCorner<3,3>() = R;
    ret.template topRightCorner<3,3>() = SO3<P>::hat(trans()) * R;
    ret.template bottomRightCorner<3,3>() = R;

    return ret;
  }

  static Matrix4p hat(Vector6p tau)
  {
    Vector3p rho = tau.template head<3>();
    Vector3p theta = tau.template tail<3>();

    Matrix4p ret = Matrix4p::Zero();
    ret.template topRightCorner<3,1>() = rho;
    ret.template topLeftCorner<3,3>() = SO3<P>::hat(theta);
    return ret;
  }

  static Vector6p vee(Matrix4p tau_hat)
  {
    return (Matrix6p() << tau_hat.template topRightCorner<3,1>(), SO3<P>::vee(tau_hat.template topLeftCorner<3,3>())).finished();
  }

  static Matrix6p ad(Vector6p tau)
  {
    Vector3p rho = tau.template head<3>();
    Vector3p theta = tau.template tail<3>();

    Matrix6p ret = Matrix6p::Zero();
    ret.template topRightCorner<3,3>() = SO3<P>::hat(rho);
    ret.template topLeftCorner<3,3>() = SO3<P>::hat(theta);
    ret.template bottomRightCorner<3,3>() = SO3<P>::hat(theta);
    return ret;
  }

  static Matrix6p Jr(Vector6p tau)
  {
    return Jl(-tau);
  }

  static Matrix6p Jl(Vector6p tau)
  {
    Vector3p rho = tau.template head<3>();
    Vector3p theta = tau.template tail<3>();

    Matrix3p Jl_theta = SO3<P>::Jl(theta);

    Matrix6p ret = Matrix6p::Zero();
    ret.template topLeftCorner<3,3>() = Jl_theta;
    ret.template topRightCorner<3,3>() = Q(rho, theta);
    ret.template bottomRightCorner<3,3>() = Jl_theta;

    return ret;
  }

  static Matrix6p Jr_inv(Vector6p tau)
  {
    return Jl_inv(-tau);
  }

  static Matrix6p Jl_inv(Vector6p tau)
  {
    Vector3p rho = tau.template head<3>();
    Vector3p theta = tau.template tail<3>();

    Matrix3p Jl_inv_theta = SO3<P>::Jl_inv(theta);

    Matrix6p ret = Matrix6p::Zero();
    ret.template topLeftCorner<3,3>() = Jl_inv_theta;
    ret.template topRightCorner<3,3>() = -Jl_inv_theta * Q(rho, theta) * Jl_inv_theta;
    ret.template bottomRightCorner<3,3>() = Jl_inv_theta;

    return ret;
  }

  template <typename OtherDerived>
  SE3<P> operator*(const SE3Base<OtherDerived, P> &T) const
  {
    SO3<P> R = so3() * T.so3();
    Vector3p t = trans() + so3() * T.trans();
    return SE3<P>(R, t);
  }

  Vector3p operator*(const Vector3p &v)
  {
    return so3_nonconst()*v + trans();
  }

  Vector3p operator*(const Vector3p &v) const
  {
    return so3()*v + trans();
  }

  SE3<P> operator+(const Vector6p &tau) const
  {
    return *this * Exp(tau);
  }

  static SE3<P> random()
  {
    Vector6p rand = Vector6p::Random();
    rand.template head<3>() *= 10;
    rand.template tail<3>() /= rand.template tail<3>().norm();
    rand.template tail<3>() *= Eigen::Matrix<P, 1, 1>::Random() * M_PI;

    return SE3<P>(rand); 
  }

private:
  static Matrix3p Q(Vector3p rho, Vector3p theta)
  {
    Matrix3p rho_x = SO3<P>::hat(rho);
    Matrix3p theta_x = SO3<P>::hat(theta);

    P th = theta.norm();
    P c1, c2, c3;
    if(th < SMALL_ANGLE)
    {
      c1 = -1.0/6.0;
      c2 = 1.0/24.0;
      c3 = 0.5*(c2 - 3.0/120.0);
    }
    else
    {
      c1 = (th - std::sin(th))/(std::pow(th,3));
      c2 = (1.0 - 0.5*th*th - std::cos(th))/std::pow(th, 4);
      c3 = 0.5*(c2 - 3*(th - std::sin(th) - std::pow(th,3)/6)/std::pow(th,5));
    }
    return 0.5*rho_x + c1*(theta_x*rho_x + rho_x*theta_x + theta_x*rho_x*theta_x) - c2*(theta_x*theta_x*rho_x + rho_x*theta_x*theta_x - 3*theta_x*rho_x*theta_x) 
                        - c3*(theta_x*rho_x*theta_x*theta_x + theta_x*theta_x*rho_x*theta_x);
  }

  SO3Type const& so3() const
  {
    return static_cast<Derived const*>(this)->so3();
  }

  SO3Type& so3_nonconst()
  {
    return static_cast<Derived*>(this)->so3_nonconst();
  }

  TransType const& trans() const
  {
    return static_cast<Derived const*>(this)->trans();
  }

  // this is redundant, but we dont know which type Derived is
  friend class SE3Base<SE3<P>, P>;
  friend class SE3Base<Map<SE3<P>>, P>;
  friend class SE3Base<Map<const SE3<P>>, P>;

};


template <typename P>
class SE3 : public SE3Base<SE3<P>, P>
{
private:
  using Matrix3p = Eigen::Matrix<P, 3, 3>;
  using Vector3p = Eigen::Matrix<P, 3, 1>;
  using Matrix4p = Eigen::Matrix<P, 4, 4>;
  using Vector6p = Eigen::Matrix<P, 6, 1>;

public:
  using SOnT = SO3<P>;
  using RnT = Vector3p;
  
  SE3() : R_{SO3<P>()}, t_{Vector3p::Zero()}
  {}

  SE3(Matrix3p R, Vector3p t) : R_{R}, t_{t}
  {}

  SE3(SO3<P> R, Vector3p t) : R_{R}, t_{t}
  {}

  SE3(Eigen::Quaternion<P> q, Vector3p t) : R_{q}, t_{t}
  {}

  SE3(P roll, P pitch, P yaw, P x, P y, P z) :
    R_{roll, pitch, yaw}, t_{x, y, z}
  {}

  SE3(Vector6p tau)
  {
    *this = this->Exp(tau);
  }

  SE3(Matrix4p T) : R_{T.template block<3,3>(0,0).eval()}, t_{T.template block<3,1>(0,3)}
  {}

  SE3(const Map<SE3<P>> &T) :
    R_{T.so3()}, t_{T.trans()}
  {}

private:
  friend class SE3Base<SE3<P>, P>;
  friend class SE3Base<Map<SE3<P>>, P>;
  
  SO3<P> const& so3() const
  {
    return R_;
  }

  SO3<P>& so3_nonconst()
  {
    return R_;
  }

  Vector3p const& trans() const
  {
    return t_;
  }

  SO3<P> R_;
  Vector3p t_;
};


template <typename P>
class Map<SE3<P>> : public SE3Base<Map<SE3<P>>, P>
{
private:
  using Vector3p = Eigen::Matrix<P, 3, 1>;

public:
  using SOnT = Map<SO3<P>>;
  using RnT = Eigen::Map<Vector3p>;
  
  Map(P* data) : 
    t_{data}, R_{data == nullptr ? nullptr : data + 3}
  {}

  Map<SE3<P>>& operator=(const SE3<P> &other)
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
  friend class SE3Base<Map<SE3<P>>, P>;
  friend class SE3<P>;

  Map<SO3<P>> const& so3() const
  {
    return R_;
  }

  Map<SO3<P>>& so3_nonconst()
  {
    return R_;
  }

  Eigen::Map<Vector3p> const& trans() const
  {
    return t_;
  }

  Map<SO3<P>> R_;
  Eigen::Map<Vector3p> t_;

};


template <typename P>
class Map<const SE3<P>> : public SE3Base<Map<const SE3<P>>, P>
{
private:
  using Vector3p = Eigen::Matrix<P, 3, 1>;

public:
  using SOnT = Map<const SO3<P>>;
  using RnT = Eigen::Map<const Vector3p>;
  
  Map(const P *data) :
    t_{data}, R_{data + 3}
  {}

  const P* data() const
  {
    return t_.data();
  }

private:
  friend class SE3Base<Map<const SE3<P>>, P>;

  Map<const SO3<P>> const& so3() const
  {
    return R_;
  }

  Eigen::Map<const Vector3p> const& trans() const
  {
    return t_;
  }

  const Map<const SO3<P>> R_;
  const Eigen::Map<const Vector3p> t_;
};

typedef SE3<double> SE3d;
typedef SE3<float> SE3f;

} //namespace lie_groups