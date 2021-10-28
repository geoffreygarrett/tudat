/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_BASESIMULATOR_H
#define TUDAT_BASESIMULATOR_H

#include <tudat/simulation/helpers.h>
#include <tudat/simulation/observer/BaseSubject.h>
#include <tudat/simulation/observer/ComputationExpense.h>
#include <tudat/simulation/observer/NumericalSolution.h>
#include <tudat/simulation/observer/SimulatorState.h>

#include <boost/make_shared.hpp>
#include <chrono>
#include <string>
#include <vector>

#include "tudat/astro/ephemerides/frameManager.h"
#include "tudat/astro/propagators/dynamicsStateDerivativeModel.h"
#include "tudat/astro/propagators/integrateEquations.h"
#include "tudat/astro/propagators/nBodyStateDerivative.h"
#include "tudat/basics/tudatTypeTraits.h"
#include "tudat/basics/utilities.h"
#include "tudat/math/interpolators/lagrangeInterpolator.h"
#include "tudat/simulation/propagation_setup/createEnvironmentUpdater.h"
#include "tudat/simulation/propagation_setup/createStateDerivativeModel.h"
#include "tudat/simulation/propagation_setup/propagationSettings.h"
#include "tudat/simulation/propagation_setup/propagationTermination.h"
#include "tudat/simulation/propagation_setup/setNumericallyIntegratedStates.h"

namespace tudat {

namespace numerical_simulation {

///
/// @tparam F the state floating-point type (e.g. float, double).
/// @tparam T the domain type being integrated over (e.g. time).
///
template <typename F = double, typename T = double,
          typename std::enable_if<is_state_scalar_and_time_type<F, T>::value,
                                  int>::type = 0>
class BaseSimulator : BaseSubject<SimulatorState<F, T>> {
 public:
  using BaseSubject<SimulatorState<F, T>>::attach;
  using BaseSubject<SimulatorState<F, T>>::detach;
  using BaseSubject<SimulatorState<F, T>>::notify;

  using S = Eigen::Matrix<F, Eigen::Dynamic, 1>;

  using SOL = std::vector<std::map<T, S>>;
  using DEP = std::vector<std::map<T, S>>;

  //! @get_docstring(BaseSimulator.ctor)
  explicit BaseSimulator(const SystemOfBodies &bodies,
                         const bool clearNumericalSolutions = true,
                         const bool setIntegratedResult = true,
                         const int cpuTimeSaveFrequency = 1,
                         const int solutionSaveFrequency = 1,
                         const int funcEvalSaveFrequency = 1)
      : bodies_(bodies),
        clearNumericalSolutions_(clearNumericalSolutions),
        setIntegratedResult_(setIntegratedResult),
  {
    if (cpuTimeSaveFrequency) {
      attach(std::make_shared<CPUTimeObserver<F, T>>(cpuTimeSaveFrequency));
    }
    if (solutionSaveFrequency) {
      attach(std::make_shared<SolutionObserver<F, T>>(solutionSaveFrequency));
    }
    if (funcEvalSaveFrequency) {
      attach(std::make_shared<FunctionEvalObserver<F, T>>(funEvalSaveFreq));
    }
  };

  //! default destructor
  virtual ~BaseSimulator() = default;

  ///
  /// @return the cpu time observer
  ///    This is a class that inherits from the BaseObserver class, and
  ///    is updated when the simulation state changes.
  ///
  std::shared_ptr<CPUTimeObserver<F, T>> getCPUTimeObserver() {
    return CPUTimeObserver_;
  };

  ///
  /// @return the numerical solution observer
  ///    This is a class that inherits from the BaseObserver class, and
  ///    is updated when the simulation state changes.
  ///
  std::shared_ptr<SolutionObserver<F, T>> getSolutionObserver() {
    return solutionObserver_;
  };

  ///
  /// @return current state of the simulation
  ///    This is an abstract inherited member function from the ISubject
  ///    class, which facilitates the observer pattern for the current
  ///    state of the simulation.
  ///
  /// @notes
  ///     This is an alias for the inherited member function `getState`
  ///     from BaseSubject. The technique used here is variadic template
  ///     with perfect forwarding. The `getState` member function is
  ///     renamed to avoid confusion with the use of the state in
  ///     astrodynamics.
  ///
  template <typename... Ts>
  auto getSimulatorState(Ts &&...ts) const
      -> decltype(getState(std::forward<Ts>(ts)...)) {
    return getState(std::forward<Ts>(ts)...);
  }

  ///
  /// @return the dependent variables observer
  ///    This is a class that inherits from the BaseObserver class, and
  ///    is updated when the simulation state changes.
  ///
  //  std::shared_ptr<SolutionObserver<F, T>> getDependentsObserver() {
  //    return dependentsObserver_;
  //  };
  //
  //  ///
  //  /// @return the numerical solution observer
  //  ///    This is a class that inherits from the BaseObserver class, and
  //  ///    is updated when the simulation state changes.
  //  ///
  //  std::shared_ptr<FunctionEvalObserver<F, T>> getFunctionEvalObserver() {
  //    return functionEvalObserver_;
  //  };

  //  virtual std::shared_ptr<IntegrationInterface> getIntegrationInterface();

  //  virtual std::shared_ptr<SolutionObserver> getSolutionObserver();

  //  virtual std::shared_ptr<TerminationObserver> getTerminationObserver();

  /// LEGACY MEMBERS.

  //! @get_docstring(BaseSimulator.integrateEquationsOfMotion)
  [[deprecated]] virtual void integrateEquationsOfMotion() = 0;

  //! @get_docstring(BaseSimulator.integrationCompletedSuccessfully)
  virtual bool integrationCompletedSuccessfully() const = 0;

  //! @get_docstring(BaseSimulator.getEquationsOfMotionNumericalSolutionBase)
  virtual SOL getEquationsOfMotionNumericalSolutionBase() = 0;

  //! @get_docstring(BaseSimulator.getDependentVariableNumericalSolutionBase)
  [[deprecated]] virtual SOL getDependentVariableNumericalSolutionBase() = 0;

  //! @get_docstring(BaseSimulator.getCumulativeComputationTimeHistoryBase)
  [[deprecated]] virtual std::vector<std::map<T, double>>
  getCumulativeComputationTimeHistoryBase() = 0;

  //! @get_docstring(BaseSimulator.getSystemOfBodies)
  SystemOfBodies getSystemOfBodies() { return bodies_; }

  //! @get_docstring(BaseSimulator.resetSystemOfBodies)
  void resetSystemOfBodies(const SystemOfBodies &bodies) { bodies_ = bodies; }

  //! @get_docstring(BaseSimulator.getSetIntegratedResult)
  bool getSetIntegratedResult() { return setIntegratedResult_; }

  //! @get_docstring(BaseSimulator.resetSetIntegratedResult)
  void resetSetIntegratedResult(const bool setIntegratedResult) {
    setIntegratedResult_ = setIntegratedResult;
  }

  //! @get_docstring(BaseSimulator.processNumericalEquationsOfMotionSolution)
  virtual void processNumericalEquationsOfMotionSolution() = 0;

 protected:
  //! @get_docstring(BaseSimulator.bodies_)
  SystemOfBodies bodies_;

  //! @get_docstring(BaseSimulator.clearNumericalSolutions_)
  bool clearNumericalSolutions_;

  //! @get_docstring(BaseSimulator.setIntegratedResult_)
  bool setIntegratedResult_;

  //! @get_docstring(BaseSimulator.CPUTimeObserver_)
  std::shared_ptr<CPUTimeObserver<F, T>> CPUTimeObserver_;

  //! @get_docstring(BaseSimulator.SolutionObserver_)
  std::shared_ptr<SolutionObserver<F, T>> solutionObserver_;

  ///
  /// @param state current state of the simulation to be set
  /// @notes
  ///    This is an abstract inherited member function from the ISubject
  ///    class, which facilitates the observer pattern for the current
  ///    state of the simulation.
  ///
  void setState(SimulatorState<F, T> &state) override { simState_ = state; };

  ///
  /// @return current state of the simulation
  ///    This is an abstract inherited member function from the ISubject
  ///    class, which facilitates the observer pattern for the current
  ///    state of the simulation.
  ///
  SimulatorState<F, T> getState() override { return simState_; };

 private:
  //! @get_docstring(BaseSimulator.simState_)
  SimulatorState<F, T> simState_;
};

}  // namespace numerical_simulation

}  // namespace tudat

#endif  // TUDAT_BASESIMULATOR_H
