/*    Copyright (c) 2010-2017, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include "Tudat/Astrodynamics/OrbitDetermination/EstimatableParameters/estimatableParameter.h"

namespace tudat
{

namespace estimatable_parameters
{

std::string getParameterTypeString( const EstimatebleParametersEnum parameterType )
{
    std::string parameterDescription;
    switch( parameterType )
    {
    case arc_wise_initial_body_state:
        parameterDescription = "arc-wise translational state ";
        break;
    case initial_body_state:
        parameterDescription = "translational state ";
        break;
    case gravitational_parameter:
        parameterDescription = "gravitational parameter ";
        break;
    case constant_drag_coefficient:
        parameterDescription = "constant drag coefficient ";
        break;
    case radiation_pressure_coefficient:
        parameterDescription = "radiation pressure coefficient ";
        break;
    case arc_wise_radiation_pressure_coefficient:
        parameterDescription = "arc-wise radiation pressure coefficient ";
        break;
    case spherical_harmonics_cosine_coefficient_block:
        parameterDescription = "cosine spherical harmonic coefficient block ";
        break;
    case spherical_harmonics_sine_coefficient_block:
        parameterDescription = "sine spherical harmonic coefficient block ";
        break;
    case constant_rotation_rate:
        parameterDescription = "constant rotation rate ";
        break;
    case rotation_pole_position:
        parameterDescription = "pole position ";
        break;
    case constant_additive_observation_bias:
        parameterDescription = "absolute observation bias ";
        break;
    case constant_relative_observation_bias:
        parameterDescription = "relative observation bias ";
        break;
    case ground_station_position:
        parameterDescription = "ground station position ";
        break;
    case equivalence_principle_lpi_violation_parameter:
        parameterDescription = " equivalence principle violation parameter ";
        break;
    case empirical_acceleration_coefficients:
        parameterDescription = " empirical acceleration coefficients ";
        break;
    case arc_wise_empirical_acceleration_coefficients:
        parameterDescription = " arc-wise empirical acceleration coefficients ";
        break;
    case full_degree_tidal_love_number:
        parameterDescription = " tidal Love number at full degree ";
        break;
    case single_degree_variable_tidal_love_number:
        parameterDescription = " tidal Love number at separate orders of single degree ";
        break;
    case ppn_parameter_gamma:
        parameterDescription = "PPN parameter gamma ";
        break;
    case ppn_parameter_beta:
        parameterDescription = "PPN parameter beta ";
        break;
    default:
        std::string errorMessage = "Error when getting parameter string, did not recognize parameter " +
                boost::lexical_cast< std::string >( parameterType );
        throw std::runtime_error( errorMessage );
    }
    return parameterDescription;
}

//! Function to determine whether the given parameter represents an initial dynamical state, or a static parameter.
bool isParameterDynamicalPropertyInitialState( const EstimatebleParametersEnum parameterType )
{
    bool flag;
    switch( parameterType )
    {
    case arc_wise_initial_body_state:
        flag = true;
        break;
    case initial_body_state:
        flag = true;
        break;
    default:
        flag = false;
        break;
    }
    return flag;
}

//! Function to determine whether the given (non-dynamical) parameter is a double or vector parameter.
bool isDoubleParameter( const EstimatebleParametersEnum parameterType )
{
    bool isDoubleParameter;
    switch( parameterType )
    {
    case gravitational_parameter:
        isDoubleParameter = true;
        break;
    case constant_drag_coefficient:
        isDoubleParameter = true;
        break;
    case radiation_pressure_coefficient:
        isDoubleParameter = true;
        break;
    case arc_wise_radiation_pressure_coefficient:
        isDoubleParameter = false;
        break;
    case spherical_harmonics_cosine_coefficient_block:
        isDoubleParameter = false;
        break;
    case spherical_harmonics_sine_coefficient_block:
        isDoubleParameter = false;
        break;
    case constant_rotation_rate:
        isDoubleParameter = true;
        break;
    case rotation_pole_position:
        isDoubleParameter = false;
        break;
    case constant_additive_observation_bias:
        isDoubleParameter = false;
        break;
    case constant_relative_observation_bias:
        isDoubleParameter = false;
        break;
    case ppn_parameter_gamma:
        isDoubleParameter = true;
        break;
    case ppn_parameter_beta:
        isDoubleParameter = true;
        break;
    case ground_station_position:
        isDoubleParameter = false;
        break;
    case equivalence_principle_lpi_violation_parameter:
        isDoubleParameter = true;
        break;
   case full_degree_tidal_love_number:
        isDoubleParameter = false;
        break;
    case single_degree_variable_tidal_love_number:
         isDoubleParameter = false;
         break;
    case empirical_acceleration_coefficients:
         isDoubleParameter = false;
         break;
     case arc_wise_empirical_acceleration_coefficients:
          isDoubleParameter = false;
          break;
    default:
        throw std::runtime_error( "Error, parameter type " + boost::lexical_cast< std::string >( parameterType ) +
                                  " not found when getting parameter type" );
    }
    return isDoubleParameter;
}

//! Function to determine whether the given (non-dynamical) parameter influences a body's orientation.
bool isParameterRotationMatrixProperty( const EstimatebleParametersEnum parameterType )
{
    bool flag;
    switch( parameterType )
    {
    case constant_rotation_rate:
        flag = true;
        break;
    case rotation_pole_position:
        flag = true;
        break;
    default:
        flag = false;
        break;
    }
    return flag;
}

//! Function to determine whether the given parameter influences an observation link directly
bool isParameterObservationLinkProperty( const EstimatebleParametersEnum parameterType )
{
    bool flag;
    switch( parameterType )
    {
    case constant_additive_observation_bias:
        flag = true;
        break;
    case constant_relative_observation_bias:
        flag = true;
        break;
    default:
        flag = false;
        break;
    }
    return flag;
}

//! Function to determine whether the given parameter influences a body's tidal gravity field variations.
bool isParameterTidalProperty( const EstimatebleParametersEnum parameterType )
{
    bool flag;
    switch( parameterType )
    {
    case full_degree_tidal_love_number:
        flag = true;
        break;
    case single_degree_variable_tidal_love_number:
        flag = true;
        break;
    default:
        flag = false;
        break;
    }
    return flag;
}




}

}

