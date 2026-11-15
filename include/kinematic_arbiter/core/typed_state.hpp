#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Eigen/Dense>

#include "fcarouge/sparse_linear_algebra.hpp"
#include "fcarouge/typed_linear_algebra.hpp"
#include "mp-units/framework/quantity.h"
#include "mp-units/systems/si.h"

namespace fcarouge {
/**
 * @brief The adapter of the typed linear algebra to mp-units elements: a quantity
 * element is stored as its numerical value in its own unit, and a stored value is
 * read back as a quantity of that unit.
 */
template <typename To, mp_units::Quantity From>
struct element_caster<To, From> {
  [[nodiscard]] static constexpr auto operator()(From value) -> To {
    static_assert(std::same_as<typename From::rep, std::remove_cvref_t<To>>);
    return value.numerical_value_in(From::unit);
  }
};

template <mp_units::Quantity To, typename From>
struct element_caster<To, From> {
  [[nodiscard]] static constexpr auto operator()(From value) -> To {
    static_assert(std::same_as<typename To::rep, std::remove_cvref_t<From>>);
    return value * To::reference;
  }
};
}  // namespace fcarouge

/**
 * @brief A typed matrix of quantities is never a quantity's representation:
 * without this, mp-units' `quantity * value` tests the matrix as a
 * representation, `matrix * mag<1>`, which fails to compile inside the typed
 * linear algebra instead of declining. Every typed matrix of this package holds
 * quantities.
 */
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
inline constexpr bool mp_units::disable_representation<
    fcarouge::typed_matrix<Matrix, RowIndexes, ColumnIndexes>>{true};

namespace kinematic_arbiter {
namespace typed {

template <auto Reference>
using Quantity = mp_units::quantity<Reference, double>;

inline constexpr auto metre = mp_units::si::metre;
inline constexpr auto second = mp_units::si::second;

using Length = Quantity<metre>;
/// A quaternion component, a gyroscope's sensitivity: a ratio.
using Ratio = Quantity<mp_units::one>;
using Speed = Quantity<metre / second>;
using Acceleration = Quantity<metre / (second * second)>;
/// The angle is a ratio: mp-units keeps the radian's kind apart, and m rad/s does
/// not add to m/s.
using AngularSpeed = Quantity<mp_units::one / second>;
using AngularAcceleration = Quantity<mp_units::one / (second * second)>;

template <typename Type>
using Inverse = decltype(1. / Type{});

template <std::size_t, typename Type>
using Always = Type;

/// A tuple of `Size` elements of `Type`.
template <typename Type, std::size_t Size>
using Repeat = decltype([]<std::size_t... Index>(std::index_sequence<Index...>) {
  return std::tuple<Always<Index, Type>...>{};
}(std::make_index_sequence<Size>{}));

/// A 3-vector of one quantity type.
template <typename Type>
using Vector = fcarouge::typed_column_vector<
    fcarouge::sparse_matrix<double, fcarouge::sparse::dense<3, 1>>, Type, Type, Type>;

/// The state's elements, in the order of core::StateIndex: the position, the
/// orientation quaternion, the linear and angular velocities and accelerations.
using State = decltype(std::tuple_cat(
    Repeat<Length, 3>{}, Repeat<Ratio, 4>{}, Repeat<Speed, 3>{},
    Repeat<AngularSpeed, 3>{}, Repeat<Acceleration, 3>{},
    Repeat<AngularAcceleration, 3>{}));

template <typename Tuple>
struct inverse_elements;

template <typename... Type>
struct inverse_elements<std::tuple<Type...>> {
  using type = std::tuple<Inverse<Type>...>;
};

/// A matrix of the given row and column element types.
template <typename Rows, typename Columns>
using Matrix = fcarouge::typed_matrix<
    fcarouge::sparse_matrix<double, fcarouge::sparse::dense<std::tuple_size_v<Rows>,
                                                            std::tuple_size_v<Columns>>>,
    Rows, Columns>;

/// The covariance of the state: each element the product of its row's and its
/// column's state units.
using StateCovariance = Matrix<State, State>;

template <typename Tuple>
struct per_second_elements;

template <typename... Type>
struct per_second_elements<std::tuple<Type...>> {
  using type = std::tuple<decltype(Type{} / Quantity<second>{})...>;
};

/// The rate of change of each state element.
using StateRate = typename per_second_elements<State>::type;

template <typename Tuple>
struct column_vector_of;

template <typename... Type>
struct column_vector_of<std::tuple<Type...>> {
  using type = fcarouge::typed_column_vector<
      fcarouge::sparse_matrix<double, fcarouge::sparse::dense<sizeof...(Type), 1>>, Type...>;
};

/// A rate of change of the state.
using StateRateVector = typename column_vector_of<StateRate>::type;

/// The process noise as a rate: each element the product of its row's and its
/// column's state units per second.
using ProcessNoiseRate = Matrix<StateRate, State>;

/// The transition matrix of the state: each element in its row's state unit per
/// its column's, stored at the positions of `Pattern`, structurally zero
/// elsewhere.
template <auto Pattern>
using Transition = fcarouge::typed_matrix<fcarouge::sparse_matrix<double, Pattern>, State,
                                          typename inverse_elements<State>::type>;

/// The Jacobian of a measurement of the given elements with respect to the state:
/// each element in the unit of its measurement element per state element, stored
/// at the positions of `Pattern`, structurally zero elsewhere.
template <typename Measurement,
          auto Pattern = fcarouge::sparse::dense<std::tuple_size_v<Measurement>,
                                                 std::tuple_size_v<State>>>
using Jacobian = fcarouge::typed_matrix<fcarouge::sparse_matrix<double, Pattern>, Measurement,
                                        typename inverse_elements<State>::type>;

/// The storage pattern of a typed matrix.
template <typename Typed>
struct pattern_of;

template <typename Type, auto Pattern, typename Rows, typename Columns>
struct pattern_of<fcarouge::typed_matrix<fcarouge::sparse_matrix<Type, Pattern>, Rows, Columns>> {
  static constexpr auto value{Pattern};
};

/// Whether the typed matrix `Typed` stores its element at `Row`, `Column`.
template <typename Typed, std::size_t Row, std::size_t Column>
inline constexpr bool stores{pattern_of<std::remove_cvref_t<Typed>>::value.stores(Row, Column)};

/// The partial derivatives of `Rows` measurement elements of `Row` with respect
/// to `Columns` state elements of `Column`.
template <typename Row, typename Column, std::size_t Columns = 3, std::size_t Rows = 3>
using Partial = Matrix<Repeat<Row, Rows>, Repeat<Inverse<Column>, Columns>>;

/// The rotation of 3-vectors of `Type`: dimensionless elements.
template <typename Type>
using Rotation = Partial<Type, Type>;

/// The typed matrix of the untyped `value`, each element read in `Reference`,
/// converted to the element's own unit: an element of another dimension fails to
/// compile.
template <typename Typed, typename Untyped>
Typed typed(const Eigen::MatrixBase<Untyped>& value, auto reference) {
  Typed result;
  [&]<std::size_t... Index>(std::index_sequence<Index...>) {
    (result.template at<Index / Typed::columns, Index % Typed::columns>(
         fcarouge::tla::element_at<Typed, Index / Typed::columns,
                                   Index % Typed::columns>(
             value(Index / Typed::columns, Index % Typed::columns) * reference)),
     ...);
  }(std::make_index_sequence<Typed::rows * Typed::columns>{});
  return result;
}

/// The typed matrix of the untyped `value`, each element read in its own unit.
/// Its structural zeros are not read.
template <typename Typed, typename Untyped>
Typed read(const Eigen::MatrixBase<Untyped>& value) {
  Typed result;
  [&]<std::size_t... Index>(std::index_sequence<Index...>) {
    (
        [&] {
          constexpr std::size_t row{Index / Typed::columns};
          constexpr std::size_t column{Index % Typed::columns};
          if constexpr (stores<Typed, row, column> && Typed::columns == 1) {
            result.template at<row>(value(row, column) *
                                    fcarouge::tla::element_at<Typed, row, column>::reference);
          } else if constexpr (stores<Typed, row, column>) {
            result.template at<row, column>(
                value(row, column) *
                fcarouge::tla::element_at<Typed, row, column>::reference);
          }
        }(),
        ...);
  }(std::make_index_sequence<Typed::rows * Typed::columns>{});
  return result;
}

/// The typed vector of the untyped `value`, read in the unit of `Type`.
template <typename Type>
Vector<Type> typed(const Eigen::Vector3d& value) {
  return Vector<Type>{value.x() * Type::reference, value.y() * Type::reference,
                      value.z() * Type::reference};
}

/// The untyped matrix of the typed `value`, each element in its own unit, by
/// default of the typed matrix's fixed size.
template <typename Typed,
          typename Untyped = Eigen::Matrix<double, Typed::rows, Typed::columns>>
Untyped untyped(const Typed& value) {
  Untyped result(Typed::rows, Typed::columns);
  [&]<std::size_t... Index>(std::index_sequence<Index...>) {
    (
        [&] {
          constexpr std::size_t row{Index / Typed::columns};
          constexpr std::size_t column{Index % Typed::columns};
          if constexpr (stores<Typed, row, column> && Typed::columns == 1) {
            result(row, column) = value.template at<row>().numerical_value_in(
                fcarouge::tla::element_at<Typed, row, column>::unit);
          } else if constexpr (stores<Typed, row, column>) {
            result(row, column) = value.template at<row, column>().numerical_value_in(
                fcarouge::tla::element_at<Typed, row, column>::unit);
          } else {
            result(row, column) = 0.0;
          }
        }(),
        ...);
  }(std::make_index_sequence<Typed::rows * Typed::columns>{});
  return result;
}

/// Writes the vector `value` in the untyped `result` from row `Row`, in the unit
/// of `Type`.
template <std::size_t Row, typename Type, typename Untyped>
void write(Eigen::MatrixBase<Untyped>& result, const Vector<Type>& value) {
  result(Row) = value.template at<0>().numerical_value_in(Type::unit);
  result(Row + 1) = value.template at<1>().numerical_value_in(Type::unit);
  result(Row + 2) = value.template at<2>().numerical_value_in(Type::unit);
}

/// Places the block `block` in `matrix` at row `Row` and column `Column`: each
/// element of the block must be of the matrix element it lands on, and stored
/// by the matrix's pattern.
template <std::size_t Row, std::size_t Column, typename Typed, typename Block>
void place(Typed& matrix, const Block& block) {
  [&]<std::size_t... Index>(std::index_sequence<Index...>) {
    (
        [&] {
          constexpr std::size_t row{Index / Block::columns};
          constexpr std::size_t column{Index % Block::columns};
          static_assert(stores<Typed, Row + row, Column + column>,
                        "The block lands on a structural zero of the matrix.");
          matrix.template at<Row + row, Column + column>(block.template at<row, column>());
        }(),
        ...);
  }(std::make_index_sequence<Block::rows * Block::columns>{});
}

/// Places `value` on `Count` diagonal elements of `matrix` from row `Row` and
/// column `Column`: each must be stored by the matrix's pattern.
template <std::size_t Row, std::size_t Column, std::size_t Count, typename Typed, typename Value>
void place_diagonal(Typed& matrix, const Value& value) {
  [&]<std::size_t... Index>(std::index_sequence<Index...>) {
    (
        [&] {
          static_assert(stores<Typed, Row + Index, Column + Index>,
                        "The diagonal lands on a structural zero of the matrix.");
          matrix.template at<Row + Index, Column + Index>(value);
        }(),
        ...);
  }(std::make_index_sequence<Count>{});
}

/// Places the vector `column` in `matrix` as the column `Column` of its first
/// three rows.
template <std::size_t Column, typename Typed, typename Type>
void place_column(Typed& matrix, const Vector<Type>& column) {
  matrix.template at<0, Column>(column.template at<0>());
  matrix.template at<1, Column>(column.template at<1>());
  matrix.template at<2, Column>(column.template at<2>());
}

/// The identity of `Typed`: ones on the diagonal.
template <typename Typed>
Typed identity() {
  Typed result;
  [&]<std::size_t... Index>(std::index_sequence<Index...>) {
    (result.template at<Index, Index>(1. * mp_units::one), ...);
  }(std::make_index_sequence<Typed::rows>{});
  return result;
}

/// The cross product, component by component as Eigen's.
template <typename Lhs, typename Rhs>
auto cross(const Vector<Lhs>& lhs, const Vector<Rhs>& rhs) {
  using Product = decltype(Lhs{} * Rhs{});
  return Vector<Product>{
      lhs.template at<1>() * rhs.template at<2>() - lhs.template at<2>() * rhs.template at<1>(),
      lhs.template at<2>() * rhs.template at<0>() - lhs.template at<0>() * rhs.template at<2>(),
      lhs.template at<0>() * rhs.template at<1>() - lhs.template at<1>() * rhs.template at<0>()};
}

/// The skew-symmetric matrix [v]× of `value`, partial derivatives of `Row` per
/// `Column`: v × w = [v]× w.
template <typename Row, typename Column, typename Type>
Partial<Row, Column> skew(const Vector<Type>& value) {
  using Element = decltype(Row{} / Column{});
  const Element x{value.template at<0>()};
  const Element y{value.template at<1>()};
  const Element z{value.template at<2>()};
  Partial<Row, Column> result;
  result.template at<0, 1>(-z);
  result.template at<0, 2>(y);
  result.template at<1, 0>(z);
  result.template at<1, 2>(-x);
  result.template at<2, 0>(-y);
  result.template at<2, 1>(x);
  return result;
}

/// The state covariance `state_covariance` projected into the measurement space
/// by the measurement's Jacobian `jacobian`, H P Hᵀ: each element the product of
/// two measurement units.
/// The Jacobian's structural zeros, `Pattern`, are not read.
template <typename Measurement, auto Pattern, typename Result, typename UntypedJacobian,
          typename UntypedCovariance>
Result project(const UntypedJacobian& jacobian, const UntypedCovariance& state_covariance) {
  using Covariance = Matrix<Measurement, Measurement>;
  const auto h = read<Jacobian<Measurement, Pattern>>(jacobian);
  return untyped<Covariance, Result>(
      Covariance{h * read<StateCovariance>(state_covariance) * fcarouge::transposed(h)});
}

/// The state covariance `covariance` propagated by the transition matrix
/// `transition`, of storage pattern `Pattern`, with the process noise
/// `process_noise`: A P Aᵀ + Q. The transition's structural zeros are not read.
template <auto Pattern, typename Result, typename UntypedTransition, typename UntypedCovariance>
Result propagate(const UntypedTransition& transition, const UntypedCovariance& covariance,
                 const UntypedCovariance& process_noise) {
  const auto a = read<Transition<Pattern>>(transition);
  return untyped<StateCovariance, Result>(StateCovariance{
      a * read<StateCovariance>(covariance) * fcarouge::transposed(a) +
      read<StateCovariance>(process_noise)});
}

/// The process noise rate `process_noise` moved by a step of `1 / window` towards
/// the outer product of the correction rate `state_diff` over `dt`.
template <typename Result, typename UntypedVector, typename UntypedMatrix>
Result adapt_process_noise(const UntypedVector& state_diff, const UntypedMatrix& process_noise,
                           double dt, double window) {
  const auto d = read<StateRateVector>(state_diff);
  const auto q = read<ProcessNoiseRate>(process_noise);
  return untyped<ProcessNoiseRate, Result>(ProcessNoiseRate{
      q + (d * fcarouge::transposed(d) * (std::abs(dt) * second) - q) / window});
}

}  // namespace typed
}  // namespace kinematic_arbiter
