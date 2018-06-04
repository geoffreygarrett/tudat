/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include <boost/make_shared.hpp>
#include <boost/bind.hpp>

#include "Tudat/Astrodynamics/Aerodynamics/exponentialAtmosphere.h"
#include "Tudat/Astrodynamics/Aerodynamics/tabulatedAtmosphere.h"
#if USE_NRLMSISE00
#include "Tudat/Astrodynamics/Aerodynamics/nrlmsise00Atmosphere.h"
#include "Tudat/Astrodynamics/Aerodynamics/nrlmsise00InputFunctions.h"
#endif
#include "Tudat/InputOutput/basicInputOutput.h"
#include "Tudat/InputOutput/solarActivityData.h"
#include "Tudat/SimulationSetup/EnvironmentSetup/createAtmosphereModel.h"

namespace tudat
{

namespace simulation_setup
{

//! Function to create a wind model.
std::shared_ptr< aerodynamics::WindModel > createWindModel(
        const std::shared_ptr< WindModelSettings > windSettings,
        const std::string& body )
{
    std::shared_ptr< aerodynamics::WindModel > windModel;

    // Check wind model type and create requested model
    switch( windSettings->getWindModelType( ) )
    {
    case custom_wind_model:
    {
        // Check input consistency
        std::shared_ptr< CustomWindModelSettings > customWindModelSettings =
                std::dynamic_pointer_cast< CustomWindModelSettings >( windSettings );
        if( customWindModelSettings == nullptr )
        {
            throw std::runtime_error( "Error when making custom wind model for body " + body + ", input is incompatible" );
        }
        windModel = std::make_shared< aerodynamics::CustomWindModel >(
                    customWindModelSettings->getWindFunction( ) );
        break;
    }
    default:
        throw std::runtime_error( "Error when making wind model for body " + body + ", input type not recognized" );
    }

    return windModel;

}

//! Function to create an atmosphere model.
std::shared_ptr< aerodynamics::AtmosphereModel > createAtmosphereModel(
        const std::shared_ptr< AtmosphereSettings > atmosphereSettings,
        const std::string& body )
{
    using namespace tudat::aerodynamics;

    // Declare return object.
    std::shared_ptr< AtmosphereModel > atmosphereModel;

    // Check which type of atmosphere model is to be created.
    switch( atmosphereSettings->getAtmosphereType( ) )
    {
    case exponential_atmosphere:
    {
        // Check whether settings for atmosphere are consistent with its type.
        std::shared_ptr< ExponentialAtmosphereSettings > exponentialAtmosphereSettings =
                std::dynamic_pointer_cast< ExponentialAtmosphereSettings >( atmosphereSettings );
        if( exponentialAtmosphereSettings == nullptr )
        {
            throw std::runtime_error(
                        "Error, expected exponential atmosphere settings for body " + body );
        }
        else
        {
            // Create and initialize exponential atmosphere model.
            std::shared_ptr< ExponentialAtmosphere > exponentialAtmosphereModel =
                    std::make_shared< ExponentialAtmosphere >(
                        exponentialAtmosphereSettings->getDensityScaleHeight( ) ,
                        exponentialAtmosphereSettings->getConstantTemperature( ),
                        exponentialAtmosphereSettings->getDensityAtZeroAltitude( ),
                        exponentialAtmosphereSettings->getSpecificGasConstant( ) );
            atmosphereModel = exponentialAtmosphereModel;
        }
        break;
    }
    case tabulated_atmosphere:
    {
        // Check whether settings for atmosphere are consistent with its type
        std::shared_ptr< TabulatedAtmosphereSettings > tabulatedAtmosphereSettings =
                std::dynamic_pointer_cast< TabulatedAtmosphereSettings >( atmosphereSettings );
        if( tabulatedAtmosphereSettings == nullptr )
        {
            throw std::runtime_error(
                        "Error, expected tabulated atmosphere settings for body " + body );
        }
        else
        {
            // Create and initialize tabulatedl atmosphere model.
            atmosphereModel = std::make_shared< TabulatedAtmosphere >(
                        tabulatedAtmosphereSettings->getAtmosphereFile( ) );
        }
        break;
    }
#if USE_NRLMSISE00
    case nrlmsise00:
    {
        std::string spaceWeatherFilePath;
        std::shared_ptr< NRLMSISE00AtmosphereSettings > nrlmsise00AtmosphereSettings =
                std::dynamic_pointer_cast< NRLMSISE00AtmosphereSettings >( atmosphereSettings );
        if( nrlmsise00AtmosphereSettings == nullptr )
        {
            // Use default space weather file stored in tudatBundle.
            spaceWeatherFilePath = input_output::getSpaceWeatherDataPath( ) + "sw19571001.txt";
        }
        else
        {
            // Use space weather file specified by user.
            spaceWeatherFilePath = nrlmsise00AtmosphereSettings->getSpaceWeatherFile( );
        }

        tudat::input_output::solar_activity::SolarActivityDataMap solarActivityData =
                tudat::input_output::solar_activity::readSolarActivityData( spaceWeatherFilePath ) ;

        // Create atmosphere model using NRLMISE00 input function
        std::function< tudat::aerodynamics::NRLMSISE00Input (double,double,double,double) > inputFunction =
                std::bind(&tudat::aerodynamics::nrlmsiseInputFunction,_1,_2,_3,_4, solarActivityData , false , TUDAT_NAN );
        atmosphereModel = std::make_shared< aerodynamics::NRLMSISE00Atmosphere >( inputFunction );
        break;
    }
#endif
    default:
        throw std::runtime_error(
                    "Error, did not recognize atmosphere model settings type " +
                    std::to_string( atmosphereSettings->getAtmosphereType( ) ) );
    }

    if( atmosphereSettings->getWindSettings( ) != nullptr )
    {
        atmosphereModel->setWindModel( createWindModel( atmosphereSettings->getWindSettings( ), body ) );
    }

    return atmosphereModel;
}


} // namespace simulation_setup

} // namespace tudat
