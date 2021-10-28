/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */
#ifndef TUDAT_BASEOBSERVER_H
#define TUDAT_BASEOBSERVER_H

#include <memory>

#include "BasePattern.h"

namespace tudat {

namespace numerical_simulation {

template <typename S>
class BaseObserver : public IObserver<S> {
 public:
  BaseObserver() = default;

  virtual void reset();
//  virtual void update(const S &state);
};

}  // namespace numerical_simulation
}  // namespace tudat

#endif  // TUDAT_BASEOBSERVER_H
