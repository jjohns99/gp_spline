#pragma once

#include <cmath>
#include "lie_groups/lie_groups.h"
// #include <pybind11/pybind11.h>
// #include <pybind11/eigen.h>
// #include <pybind11/operators.h>

namespace lie_groups
{

template <typename P>
class SO3;

namespace internal
{

template <typename P>
struct traits<SO3<P>>
{
  using QuatType = Eigen::Quaternion<P>;
};

template <typename P>
struct traits<Map<SO3<P>>>
{
  using QuatType = Eigen::Map<Eigen::Quaternion<P>>;
};

template <typename P>
struct traits<Map<const SO3<P>>>
{
  using QuatType = Eigen::Map<const Eigen::Quaternion<P>>;
};

} //namespace internal

template <typename Derived, typename P>
class SO3Base : public LieGroup<SO3<P>, SO3Base<Derived, P>, P, 3, 3>
{
private:
  using Matrix3p = Eigen::Matrix<P, 3, 3>;
  using Vector3p = Eigen::Matrix<P, 3, 1>;

  using QuatType = typename internal::traits<Derived>::QuatType;

public:
  using TangentT = Vector3p;
  using JacT = Matrix3p;
  using NonmapT = SO3<P>;

  // degrees of freedom
  static constexpr const int DoF = 3;
  // amount of memory parameters
  static constexpr const int MEM = 4;
  // size of space it acts on
  static constexpr const int ACT = 3;

  virtual Matrix3p matrix() override = 0;
  virtual Matrix3p matrix() const override = 0;

  virtual SO3<P> inverse() const override
  {
    return SO3<P>(quat().inverse());
  }

  static SO3<P> Exp(Vector3p phi)
  {
    Eigen::AngleAxis<P> phi_aa(phi.norm(), phi.normalized());
    return SO3<P>(Eigen::Quaternion<P>(phi_aa));
  }

  virtual Vector3p Log() const override
  {
    Eigen::AngleAxis<P> phi_aa(quat());
    return phi_aa.angle() * phi_aa.axis();
  }

  virtual Matrix3p Ad() override
  {
    return matrix();
  }

  static Matrix3p hat(Vector3p phi)
  {
    return (Matrix3p() << 0.0, -phi(2), phi(1),
                        phi(2), 0.0, -phi(0),
                        -phi(1), phi(0), 0.0).finished();
  }

  static Vector3p vee(Matrix3p phi_hat)
  {
    return (Vector3p() << phi_hat(2,1), phi_hat(0,2), phi_hat(1,0)).finished();
  }

  static Matrix3p ad(Vector3p phi)
  {
    return hat(phi);
  }

  static Matrix3p Jr(Vector3p phi)
  {
    return Jl(phi).transpose();
  }

  static Matrix3p Jl(Vector3p phi)
  {
    P ph = phi.norm();
    Vector3p u = phi/ph;
    // while(ph > 2.0*M_PI)
    //   ph -= 2.0*M_PI;
    // if(ph > M_PI)
    // {
    //   ph = -(ph - 2.0*M_PI);
    //   u *= -1.0;
    // }
    Matrix3p u_hat = hat(u);
    if(ph < SMALL_ANGLE)
      return Matrix3p::Identity();
    return Matrix3p::Identity() + (1.0-std::cos(ph))/ph*u_hat + (ph-std::sin(ph))/ph*u_hat*u_hat;
  }

  static Matrix3p Jr_inv(Vector3p phi)
  {
    return Jl_inv(phi).transpose();
  }

  static Matrix3p Jl_inv(Vector3p phi)
  {
    P ph = phi.norm();
    Vector3p u = phi/ph;
    // while(ph > 2.0*M_PI)
    //   ph -= 2.0*M_PI;
    // if(ph > M_PI)
    // {
    //   ph = -(ph - 2.0*M_PI);
    //   u *= -1.0;
    // }
    Matrix3p u_hat = hat(u);
    if(ph < SMALL_ANGLE)
      return Matrix3p::Identity();
    if(std::abs(ph) > M_PI - SMALL_ANGLE)
      return Matrix3p::Identity() - 0.5*M_PI*u_hat + u_hat*u_hat;
    return Matrix3p::Identity() - 0.5*ph*u_hat + (1.0 - ph*(1.0+std::cos(ph))/(2.0*std::sin(ph)))*u_hat*u_hat;
  }

  template <typename OtherDerived>
  SO3<P> operator*(const SO3Base<OtherDerived, P> &R) const
  {
    return SO3<P>((quat() * R.quat()).normalized());
  }

  Vector3p operator*(const Vector3p &v) const
  {
    Eigen::Quaternion<P> p;
    p.w() = 0;
    p.vec() = v;
    Eigen::Quaternion<P> p_rot = quat() * p * quat().inverse();
    return p_rot.vec();
  }

  SO3<P> operator+(const Vector3p &phi) const
  {
    return *this * Exp(phi);
  }

  static SO3<P> random()
  {
    Eigen::Matrix<P, 3, 1> rand = Eigen::Matrix<P, 3, 1>::Random();
    rand /= rand.norm();
    rand *= Eigen::Matrix<P, 1, 1>::Random() * M_PI;

    return SO3<P>(rand);
  }

  // access minimal representation without reference
  Eigen::Quaternion<P> mem() const
  {
    return quat();
  }

private:
  
  QuatType const& quat() const
  {
    return static_cast<Derived const*>(this)->quat();
  }

  // this is redundant, but we dont know which type Derived is
  friend class SO3Base<SO3<P>, P>;
  friend class SO3Base<Map<SO3<P>>, P>;
  friend class SO3Base<Map<const SO3<P>>, P>;

};


template <typename P>
class SO3 : public SO3Base<SO3<P>, P>
{
private:
  using Matrix3p = Eigen::Matrix<P, 3, 3>;
  using Vector3p = Eigen::Matrix<P, 3, 1>;

public:
  SO3() : quat_{1, 0, 0, 0}, quat_last_eval_{quat_}, mat_{Matrix3p::Identity()}
  {}

  SO3(Matrix3p R) : quat_{R}, quat_last_eval_{quat_}, mat_{R}
  {}

  SO3(Eigen::Quaternion<P> q) : quat_{q.normalized()}, quat_last_eval_{quat_}, mat_{quat_.toRotationMatrix()}
  {}

  SO3(P roll, P pitch, P yaw)
  {
    Eigen::AngleAxis<P> rollAngle(roll, Eigen::Matrix<P, 3, 1>::UnitX());
    Eigen::AngleAxis<P> pitchAngle(pitch, Eigen::Matrix<P, 3, 1>::UnitY());
    Eigen::AngleAxis<P> yawAngle(yaw, Eigen::Matrix<P, 3, 1>::UnitZ());

    quat_ = yawAngle * pitchAngle * rollAngle;

    quat_last_eval_ = quat_;
    mat_ = quat_.toRotationMatrix();
  }

  SO3(Vector3p phi)
  {
    *this = this->Exp(phi);
  }

  SO3(const Map<SO3<P>> &R) :
    quat_{R.quat_}, quat_last_eval_{quat_}, mat_{quat_.toRotationMatrix()}
  {}

  Matrix3p matrix() override
  {
    if(quat_.w() != quat_last_eval_.w() || quat_.vec() != quat_last_eval_.vec())
    {
      mat_ = quat_.toRotationMatrix();
      quat_last_eval_ = quat_;
    }

    return mat_;
  }

  Matrix3p matrix() const override
  {
    assert(quat_.w() == quat_last_eval_.w() && quat_.vec() == quat_last_eval_.vec());
    return mat_;
  }

private:
  friend class SO3Base<SO3<P>, P>;
  friend class SO3Base<Map<SO3<P>>, P>;

  Eigen::Quaternion<P> const& quat() const
  {
    return quat_;
  }

  Eigen::Quaternion<P> quat_;

  Matrix3p mat_;
  Eigen::Quaternion<P> quat_last_eval_;

};


template <typename P>
class Map<SO3<P>> : public SO3Base<Map<SO3<P>>, P>
{
private:
  using Matrix3p = Eigen::Matrix<P, 3, 3>;
  using Vector3p = Eigen::Matrix<P, 3, 1>;

public:
  Map(P *data) : quat_{data} // data needs to be in form (x, y, z, w)
  {
    if(data != nullptr)
    {
      quat_.normalize();
      quat_last_eval_ = quat_;
      mat_ = quat_.toRotationMatrix();
    }
  }

  Matrix3p matrix() override
  {
    if(quat_.w() != quat_last_eval_.w() || quat_.vec() != quat_last_eval_.vec())
    {
      mat_ = quat_.toRotationMatrix();
      quat_last_eval_ = quat_;
    }

    return mat_;
  }

  Matrix3p matrix() const override
  {
    assert(quat_.w() == quat_last_eval_.w() && quat_.vec() == quat_last_eval_.vec());
    return mat_;
  }

  Map<SO3<P>>& operator=(const SO3<P> &other)
  {
    quat_ = other.mem();
    mat_ = quat_.toRotationMatrix();
    quat_last_eval_ = quat_;

    return *this;
  }

  P* data()
  {
    return quat_.vec().data();
  }

private:
  friend class SO3Base<Map<SO3<P>>, P>;
  friend class SO3<P>;

  Eigen::Map<Eigen::Quaternion<P>> const& quat() const
  {
    return quat_;
  }

  Eigen::Map<Eigen::Quaternion<P>> quat_;

  Matrix3p mat_;
  Eigen::Quaternion<P> quat_last_eval_;

};


template <typename P>
class Map<const SO3<P>> : public SO3Base<Map<const SO3<P>>, P>
{
protected:
  using Matrix3p = Eigen::Matrix<P, 3, 3>;
  using Vector3p = Eigen::Matrix<P, 3, 1>;

public:
  Map(const P *data) : quat_{data}, quat_last_eval_{quat_}, mat_{quat_.toRotationMatrix()} // data needs to be in form (x, y, z, w)
  {}

  Matrix3p matrix() override
  {
    assert(quat_.w() == quat_last_eval_.w() && quat_.vec() == quat_last_eval_.vec());
    return mat_;
  }

  Matrix3p matrix() const override
  {
    assert(quat_.w() == quat_last_eval_.w() && quat_.vec() == quat_last_eval_.vec());
    return mat_;
  }

  const P* data()
  {
    return quat_.vec().data();
  }


private:
  friend class SO3Base<Map<const SO3<P>>, P>;

  Eigen::Map<const Eigen::Quaternion<P>> const& quat() const
  {
    return quat_;
  }
  
  const Eigen::Map<const Eigen::Quaternion<P>> quat_;

  const Eigen::Quaternion<P> quat_last_eval_;
  const Matrix3p mat_;
};


typedef SO3<double> SO3d;
typedef SO3<float> SO3f;

} //namespace lie_groups