/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */
#ifndef TUDAT_BASESUBJECT_H
#define TUDAT_BASESUBJECT_H

#include <cassert>
#include <memory>
#include <list>

#include "BasePattern.h"

namespace tudat {

namespace numerical_simulation {

template <typename S>
class BaseSubject : public ISubject<S> {
 public:
  using ISubject<S>::getState;
  using ISubject<S>::setState;

  BaseSubject() = default;

  virtual ~BaseSubject() = default;

  void attach(std::shared_ptr<IObserver<S>> observer) override {
    observers_.push_back(observer);
  };

  void detach(std::shared_ptr<IObserver<S>> observer) override {
    observers_.remove(observer);
  }

  void notify() override {
    for (auto& observer : observers_) {
      observer->update(getState());
    }
  }

 private:
  std::list<std::shared_ptr<IObserver<S>>> observers_;
};

}  // namespace numerical_simulation
}  // namespace tudat
#endif  // TUDAT_BASESUBJECT_H
