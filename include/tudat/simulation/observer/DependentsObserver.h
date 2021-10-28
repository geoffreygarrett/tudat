/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */
#ifndef TUDAT_DEPENDENTSOBSERVER_H
#define TUDAT_DEPENDENTSOBSERVER_H

#include <tudat/simulation/DataFrame.h>
#include <tudat/simulation/observer/BaseObserver.h>
#include <tudat/simulation/observer/SimulatorState.h>

#include <map>
#include <memory>

namespace tudat {

namespace numerical_simulation {
///
/// @tparam F the state floating-point type (e.g. float, double).
/// @tparam T the domain type being integrated over (e.g. time).
/// @tparam H the domain step-size type.
///
template <typename F, typename T>

class DependentsObserver : public BaseObserver<SimulatorState<F, T>> {
 public:
  using DataFrame = std::shared_ptr<DataFrame<T, double>>;
  using DependentsData =
      std::pair<std::function<Eigen::VectorXd()>, std::map<int, std::string>>;

  DependentsObserver(propagatorSettings, bodies, dynamicsStateDerivative,
                     int saveFrequency = 1)
      : cumulativeHistory_(std::make_shared<SeriesType>({}, "cpu_time")),
        dependentsData_(createDependentVariableListFunction<T, F>(
            propagatorSettings_->getDependentVariablesToSave(), bodies_,
            dynamicsStateDerivative_->getStateDerivativeModels())),
        saveFrequency_(saveFrequency),
        updateCount_(0){};

  //! @get_docstring(ComputationExpense.getCumulativeHistory)
  std::shared_ptr<Series<T, double>> getCumulativeHistory() {
    return dependentsHistory_;
  };

  //! @get_docstring(ComputationExpense.update)
  void update(SimulatorState<F, T> state) override {
    if ((updateCount_ % saveFrequency_) == 0) {
      cumulativeHistory_->append({state.t, state.cumulativeTime});
    }
    updateCount_++;
  }

  //! @get_docstring(ComputationExpense.reset)
  void reset() override {
    cumulativeHistory_->clear();
    updateCount_ = 0;
  }

 private:
  DependentsData dependentsData_;

  //! @get_docstring(ComputationExpense.cumulativeHistory_)
  SeriesType cumulativeHistory_;

  //! @get_docstring(ComputationExpense.saveFrequency_)
  int saveFrequency_;

  //! @get_docstring(ComputationExpense.updateCount_)
  int updateCount_;
};

}  // namespace numerical_simulation

}  // namespace tudat

#endif  // TUDAT_DEPENDENTSOBSERVER_H
