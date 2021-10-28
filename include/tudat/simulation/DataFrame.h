/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */
#ifndef TUDAT_DATAFRAME_H
#define TUDAT_DATAFRAME_H

#include <map>
#include <string>

namespace tudat {

namespace numerical_simulation {

template <typename T, typename F>
class Series {
 public:
  //! Legacy tudat data type map[float, eig]
  Series(std::map<T, F> data, std::string name = "", bool copy = true)
      : data_(data), name_(name) {}

  // [] operator
  F& operator[](T i) { return data_[i]; }

  void append(std::map<T, F> data) {
    // iterate through data
    for (auto& i : data) {
      data_[i.first] = i.second;
    }
  }

  void clear() { data_.clear(); }

 private:
  std::string name_;

  std::map<T, F> data_;
};

template <typename T, typename F>
class DataFrame {
 public:
  using S = Eigen::Matrix<F, Eigen::Dynamic, 1>;

  //! Legacy tudat data type map[float, eig]
  DataFrame(std::map<T, F> data, std::vector<std::string> columns = nullptr,
            bool copy = true)
      : data_(data), columns_(columns) {}

  // [] operator
  S& operator[](T i) { return data_[i]; }

  void append(std::map<T, S> data) {
    // iterate through data
    for (auto& i : data) {
      data_[i.first] = i.second;
    }
  }
  void clear() { data_.clear(); }

 private:
  std::vector<std::string> columns_;

  std::map<T, S> data_;
};

}  // namespace numerical_simulation

}  // namespace tudat
#endif  // TUDAT_DATAFRAME_H
