
/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_SIMULATORSTATE_H
#define TUDAT_SIMULATORSTATE_H

#include <Eigen/Dense>

namespace tudat {

namespace numerical_simulation {

using namespace propagators;
using namespace simulation_setup;

///
/// @tparam F the state floating-point type (e.g. float, double).
/// @tparam T the domain type being integrated over (e.g. time).
/// @tparam H the domain step-size type.
///
template <typename F = double, typename T = double, typename H = double>
struct SimulatorState {
  using S = Eigen::Matrix<F, Eigen::Dynamic, 1>;

  bool terminal;  // has the simulation reached a terminal state.

  T tInitial;  // current domain value of the simulation (e.g. time).
  T tCurrent;  // current domain value of the simulation (e.g. time).
  S stateNew;  // current domain value of the simulation (e.g. time).
  S statePrevious;  // current domain value of the simulation (e.g. time).
  H stepSize;  // current (next) step size.
  S s;         // state of propagated bodies.
};

}  // namespace numerical_simulation
}  // namespace tudat

#endif  // TUDAT_SIMULATORSTATE_H
