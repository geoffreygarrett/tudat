//
// Created by ggarr on 25/10/2021.
//

#ifndef TUDAT_BASEPATTERN_H
#define TUDAT_BASEPATTERN_H

namespace tudat {

namespace numerical_simulation {

template <typename S>
class IObserver {
 public:
  virtual ~IObserver() = default;
  virtual void update(const S &state) = 0;
};

template <typename S>
class ISubject {
 public:
  virtual ~ISubject() = default;
  virtual void attach(std::shared_ptr<IObserver<S>> observer) = 0;
  virtual void detach(std::shared_ptr<IObserver<S>> observer) = 0;
  virtual void notify() = 0;
  virtual S getState() = 0;
  virtual void setState(S &state) = 0;
};

}  // namespace numerical_simulation
}  // namespace tudat

#endif  // TUDAT_BASEPATTERN_H
