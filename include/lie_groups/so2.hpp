#pragma once

#include "lie_groups/lie_groups.h"
#include <cmath>
// #include <pybind11/pybind11.h>
// #include <pybind11/eigen.h>
// #include <pybind11/operators.h>

namespace lie_groups 
{

template <typename P>
class SO2;

namespace internal
{

template <typename P>
struct traits<SO2<P>>
{
  using ThetaType = Eigen::Matrix<P, 1, 1>;
};

template <typename P>
struct traits<Map<SO2<P>>>
{
  using ThetaType = Eigen::Map<Eigen::Matrix<P, 1, 1>>;
};

template <typename P>
struct traits<Map<const SO2<P>>>
{
  using ThetaType = Eigen::Map<const Eigen::Matrix<P, 1, 1>>;
};

} //namespace internal

// need a base class that is agnostic to the data type in order to
// implement Eigen::Map-like class for standard data class

template <typename Derived, typename P>
class SO2Base : public LieGroup<SO2<P>, SO2Base<Derived, P>, P, 2, 1>
{
private:
  using Matrix1p = Eigen::Matrix<P, 1, 1>;
  using Matrix2p = Eigen::Matrix<P, 2, 2>;
  using Vector2p = Eigen::Matrix<P, 2, 1>;

  using ThetaType = typename internal::traits<Derived>::ThetaType;

public:
  using TangentT = Matrix1p;
  using JacT = Matrix1p;
  using NonmapT = SO2<P>;

  // degrees of freedom
  static constexpr const int DoF = 1;
  // amount of memory parameters
  static constexpr const int MEM = 1;
  // size of space it acts on
  static constexpr const int ACT = 2;

  virtual Matrix2p matrix() override = 0;
  virtual Matrix2p matrix() const override = 0;

  virtual SO2<P> inverse() const override
  {
    Matrix1p inv_th = -theta();
    return SO2<P>(inv_th);
  }

  static SO2<P> Exp(Matrix1p theta)
  {
    return SO2<P>(theta);
  }

  virtual Matrix1p Log() const override
  {
    return (Matrix1p() << std::atan2(std::sin(theta()(0)), std::cos(theta()(0)))).finished();
  }

  virtual Matrix1p Ad() override
  {
    return (Matrix1p() << 1.0).finished();
  }

  static Matrix2p hat(Matrix1p theta)
  {
    return (Matrix2p() << 0, -theta(0), 
                          theta(0), 0).finished();
  }
  
  static Matrix1p vee(Matrix2p theta_hat)
  {
    return (Matrix1p() << theta_hat(1,0)).finished();
  }

  static Matrix1p ad(Matrix1p theta)
  {
    return Matrix1p::Zero();
  }

  static Matrix1p Jr(Matrix1p theta)
  {
    return Matrix1p::Identity();
  }

  static Matrix1p Jl(Matrix1p theta)
  {
    return Matrix1p::Identity();
  }

  static Matrix1p Jr_inv(Matrix1p theta)
  {
    return Matrix1p::Identity();
  }

  static Matrix1p Jl_inv(Matrix1p theta)
  {
    return Matrix1p::Identity();
  }

  template <typename OtherDerived>
  SO2<P> operator*(const SO2Base<OtherDerived, P> &R) const
  {
    Matrix1p sum_th = theta() + R.theta();
    return SO2<P>(sum_th);
  }

  Vector2p operator*(const Vector2p &v)
  {
    return matrix()*v;
  }

  // for const SO2Type, call matrix() const
  Vector2p operator*(const Vector2p &v) const
  {
    return matrix()*v;
  }

  SO2<P> operator+(const Matrix1p &theta) const
  {
    return *this * Exp(theta);
  }

  static SO2<P> random()
  {
    Eigen::Matrix<P, 1, 1> rand = Matrix1p::Random();
    rand *= M_PI;

    return SO2<P>(rand);
  }

  // access minimal representation without reference
  Matrix1p mem() const
  {
    return theta();
  }

private:

  ThetaType const& theta() const
  {
    return static_cast<Derived const*>(this)->theta();
  }

  // this is redundant, but we dont know which type Derived is
  friend class SO2Base<SO2<P>, P>;
  friend class SO2Base<Map<SO2<P>>, P>;
  friend class SO2Base<Map<const SO2<P>>, P>;
  
};


template <typename P>
class SO2 : public SO2Base<SO2<P>, P>
{
private:
  using Matrix1p = Eigen::Matrix<P, 1, 1>;
  using Matrix2p = Eigen::Matrix<P, 2, 2>;

public:
  SO2() : th_{Matrix1p::Zero()}, th_last_eval_{th_}, mat_{Matrix2p::Identity()}
  {}

  SO2(Matrix2p R) : th_{(Matrix1p() << std::atan2(R(1,0), R(0,0))).finished()},
                    th_last_eval_{th_}, mat_{R}
  {}

  SO2(Matrix1p theta) : th_{theta}, th_last_eval_{th_},
                        mat_{(Matrix2p() << std::cos(th_(0)), -std::sin(th_(0)),
                            std::sin(th_(0)), std::cos(th_(0))).finished()}
  {}

  SO2(const Map<SO2<P>> &R) :
    th_{R.theta()}, th_last_eval_{th_},
    mat_{(Matrix2p() << std::cos(th_(0)), -std::sin(th_(0)),
      std::sin(th_(0)), std::cos(th_(0))).finished()}
  {}

  Matrix2p matrix() override
  {
    if(th_(0) != th_last_eval_(0))
    {
      mat_ = (Matrix2p() << std::cos(th_(0)), -std::sin(th_(0)),
                            std::sin(th_(0)), std::cos(th_(0))).finished();
      th_last_eval_ = th_;
    }
    return mat_;
  }

  Matrix2p matrix() const override
  {
    assert(th_(0) == th_last_eval_(0));
    return mat_;
  }

private:
  friend class SO2Base<SO2<P>, P>;
  friend class SO2Base<Map<SO2<P>>, P>;
  friend class SO2Base<Map<const SO2<P>>, P>;

  Matrix1p const& theta() const
  {
    return th_;
  }

  Matrix1p th_;

  Matrix2p mat_;
  // check this parameter when calling matrix() to see if mat_ needs to be recomputed
  Matrix1p th_last_eval_;
};


template <typename P>
class Map<SO2<P>> : public SO2Base<Map<SO2<P>>, P>
{
protected:
  using Matrix1p = Eigen::Matrix<P, 1, 1>;
  using Matrix2p = Eigen::Matrix<P, 2, 2>;

public:
  Map(P *data) : th_{data}
  {
    if(data != nullptr)
    {
      th_last_eval_ = th_;
      mat_ = (Matrix2p() << std::cos(th_(0)), -std::sin(th_(0)),
                            std::sin(th_(0)), std::cos(th_(0))).finished();
    }
  }

  Matrix2p matrix() override
  {
    if(th_(0) != th_last_eval_(0))
    {
      mat_ = (Matrix2p() << std::cos(th_(0)), -std::sin(th_(0)),
                            std::sin(th_(0)), std::cos(th_(0))).finished();
      th_last_eval_ = th_;
    }
    return mat_;
  }

  Matrix2p matrix() const override
  {
    assert(th_(0) == th_last_eval_(0));
    return mat_;
  }

  Map<SO2<P>>& operator=(const SO2<P> &other)
  {
    th_ = other.mem();
    mat_ = (Matrix2p() << std::cos(th_(0)), -std::sin(th_(0)),
                          std::sin(th_(0)), std::cos(th_(0))).finished();
    th_last_eval_ = th_;

    return *this;
  }

  P* data()
  {
    return th_.data();
  }

private:
  friend class SO2Base<Map<SO2<P>>, P>;
  friend class SO2<P>;

  Eigen::Map<Matrix1p> const& theta() const
  {
    return th_;
  }
  
  Eigen::Map<Matrix1p> th_;
  Matrix1p th_last_eval_;
  Matrix2p mat_;
};


template <typename P>
class Map<const SO2<P>> : public SO2Base<Map<const SO2<P>>, P>
{
protected:
  using Matrix1p = Eigen::Matrix<P, 1, 1>;
  using Matrix2p = Eigen::Matrix<P, 2, 2>;

public:
  Map(const P *data) : th_{data}, th_last_eval_{th_},
                 mat_{(Matrix2p() << std::cos(th_(0)), -std::sin(th_(0)),
                            std::sin(th_(0)), std::cos(th_(0))).finished()}
  {}

  Matrix2p matrix() override
  {
    assert(th_(0) == th_last_eval_(0));
    return mat_;
  }

  Matrix2p matrix() const override
  {
    assert(th_(0) == th_last_eval_(0));
    return mat_;
  }

  const P* data()
  {
    return th_.data();
  }

private:
  friend class SO2Base<Map<const SO2<P>>, P>;

  Eigen::Map<const Matrix1p> const& theta() const
  {
    return th_;
  }
  
  const Eigen::Map<const Matrix1p> th_;
  const Matrix1p th_last_eval_;
  const Matrix2p mat_;
};


typedef SO2<double> SO2d;
typedef SO2<float> SO2f;

} //namespace lie_groups