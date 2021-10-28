/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */
#ifndef TUDAT_NUMERICALSOLUTION_H
#define TUDAT_NUMERICALSOLUTION_H

#include <tudat/simulation/DataFrame.h>
#include <tudat/simulation/observer/BaseObserver.h>
#include <tudat/simulation/observer/SimulatorState.h>

#include <map>

namespace tudat {

namespace numerical_simulation {
///
/// @tparam F the state floating-point type (e.g. float, double).
/// @tparam T the domain type being integrated over (e.g. time).
/// @tparam H the domain step-size type.
///
template <typename F, typename T>

class SolutionObserver : public BaseObserver<SimulatorState<F, T>> {
 public:
  using S = Eigen::Matrix<F, Eigen::Dynamic, 1>;
  using DataFrameType = std::shared_ptr<DataFrame<T, S>>;

  SolutionObserver()
      : solutionHistory_(std::make_shared<DataFrame<T, S>>({}, "solution")){};

  //! @get_docstring(SolutionObserver.getNumericalSolution)
  std::shared_ptr<Series<T, double>> getNumericalSolution() {
    return solutionHistory_;
  };

  //! @get_docstring(SolutionObserver.update)
  void update(SimulatorState<F, T> state) override {
    if ((updateCount_ % saveFrequency_) == 0) {
      solutionHistory_->append({state.t, state.s});
    }
    updateCount_++;
  }

  //! @get_docstring(SolutionObserver.reset)
  void reset() override {
    solutionHistory_->clear();
    updateCount_ = 0;
  }

 private:
  //! @get_docstring(SolutionObserver.solutionHistory_)
  DataFrameType solutionHistory_;

  //! @get_docstring(SolutionObserver.saveFrequency_)
  int saveFrequency_;

  //! @get_docstring(SolutionObserver.updateCount_)
  int updateCount_;
};

}  // namespace numerical_simulation

}  // namespace tudat

#endif  // TUDAT_NUMERICALSOLUTION_H