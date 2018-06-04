/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_NUMERICALOBSERVATIONPARTIAL_H
#define TUDAT_NUMERICALOBSERVATIONPARTIAL_H

#include <boost/shared_ptr.hpp>
#include <tr1/functional>
#include <boost/bind.hpp>

#include "Tudat/Astrodynamics/OrbitDetermination/ObservationPartials/observationPartial.h"
#include "Tudat/Astrodynamics/ObservationModels/observationModel.h"
#include "Tudat/Astrodynamics/OrbitDetermination/EstimatableParameters/estimatableParameter.h"
#include "Tudat/SimulationSetup/PropagationSetup/propagationSettings.h"
#include "Tudat/Mathematics/NumericalIntegrators/createNumericalIntegrator.h"
#include "Tudat/SimulationSetup/EnvironmentSetup/body.h"

namespace tudat
{

namespace observation_partials
{

void emptyVoidFunction( );

//! Function to compute numerical partial derivative of double observable w.r.t. double parameter.
Eigen::Matrix< double, 1, 1 > calculateNumericalObservationParameterPartial(
        std::shared_ptr< estimatable_parameters::EstimatableParameter< double > > parameter,
        const double parameterPerturbation,
        std::function< double( const double ) > observationFunction,
        const double evaluationTime,
        std::function< void( ) > updateFunction = &emptyVoidFunction );

//! Function to compute numerical partial derivative of vector observable w.r.t. double parameter.
Eigen::Matrix< double, Eigen::Dynamic, 1 > calculateNumericalObservationParameterPartial(
        std::shared_ptr< estimatable_parameters::EstimatableParameter< double > > parameter,
        const double parameterPerturbation,
        std::function< Eigen::VectorXd( const double ) > observationFunction,
        const double evaluationTime,
        std::function< void( ) > updateFunction = &emptyVoidFunction );

//! Function to compute numerical partial derivative of vector observable w.r.t. vector parameter.
Eigen::MatrixXd calculateNumericalObservationParameterPartial(
        std::shared_ptr< estimatable_parameters::EstimatableParameter< Eigen::VectorXd > > parameter,
        const Eigen::VectorXd parameterPerturbation,
        std::function< Eigen::VectorXd( const double ) > observationFunction,
        const double evaluationTime,
        std::function< void( ) > updateFunction = &emptyVoidFunction );

}

}

#endif // TUDAT_NUMERICALOBSERVATIONPARTIAL_H
