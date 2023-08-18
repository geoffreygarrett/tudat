/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>

#include "tudat/basics/testMacros.h"

#include "tudat/astro/basic_astro/unitConversions.h"
#include "tudat/astro/ground_stations/pointingAnglesCalculator.h"
#include "tudat/astro/ground_stations/groundStationState.h"
#include "tudat/astro/basic_astro/sphericalBodyShapeModel.h"
#include "tudat/math/basic/coordinateConversions.h"
#include "tudat/simulation/environment_setup/body.h"
#include "tudat/simulation/estimation.h"
#include "tudat/astro/basic_astro/dateTime.h"
#include "tudat/astro/earth_orientation/terrestrialTimeScaleConverter.h"

using namespace tudat::simulation_setup;
using namespace tudat::basic_astrodynamics;
using namespace tudat::ground_stations;
using namespace tudat::observation_models;
using namespace tudat::spice_interface;
using namespace tudat::ephemerides;
using namespace tudat::unit_conversions;
using namespace tudat::basic_astrodynamics;
using namespace tudat;


namespace tudat
{
namespace unit_tests
{

BOOST_AUTO_TEST_SUITE( test_pointing_angles_calculator )

// Testing computation of azimuth and elevation given a topocentric vector
// Values compared against data generated by PyGeodesy (https://github.com/mrJean1/PyGeodesy) with the class ltpTuples.Ned
BOOST_AUTO_TEST_CASE( test_TopocentricVectorToAzEl )
{
    Eigen::Vector3d topocentricVector;
    double expectedAzimuth;
    double expectedElevation;

    double degreesToRadians = unit_conversions::convertDegreesToRadians( 1.0 );

    // Define Earth shape model (sphere)
    std::shared_ptr< SphericalBodyShapeModel > bodyShape = std::make_shared< SphericalBodyShapeModel >( 6.371E6 );
    // Create pointing angles calculator
    std::shared_ptr< GroundStationState > stationState = std::make_shared< GroundStationState >(
                Eigen::Vector3d::Zero(), coordinate_conversions::cartesian_position, bodyShape );
    std::shared_ptr< PointingAnglesCalculator > pointAnglesCalculator = std::make_shared< PointingAnglesCalculator >(
                [ & ]( const double ){ return Eigen::Quaterniond( Eigen::Matrix3d::Identity( ) ); },
                std::bind( &GroundStationState::getRotationFromBodyFixedToTopocentricFrame, stationState, std::placeholders::_1 ) );

    // Case 1
    topocentricVector = ( Eigen::Vector3d( ) << 69282032.302755102515, 0, 39999999.999999992549 ).finished( );
    expectedAzimuth = 90.0 * degreesToRadians;
    expectedElevation = 30.0 * degreesToRadians;

    BOOST_CHECK_CLOSE_FRACTION( expectedAzimuth, pointAnglesCalculator->calculateAzimuthAngle( topocentricVector ), 3.0 * std::numeric_limits< double >::epsilon( ) );
    BOOST_CHECK_CLOSE_FRACTION( expectedElevation, pointAnglesCalculator->calculateElevationAngle( topocentricVector ), 3.0 * std::numeric_limits< double >::epsilon( ) );

    // Case 2
    topocentricVector = ( Eigen::Vector3d( ) << 7806858.1854810388759, 74277294.019097447395, 28669435.963624030352 ).finished( );
    expectedAzimuth = 6.0 * degreesToRadians;
    expectedElevation = 21.0 * degreesToRadians;

    BOOST_CHECK_CLOSE_FRACTION( expectedAzimuth, pointAnglesCalculator->calculateAzimuthAngle( topocentricVector ), 3.0 * std::numeric_limits< double >::epsilon( ) );
    BOOST_CHECK_CLOSE_FRACTION( expectedElevation, pointAnglesCalculator->calculateElevationAngle( topocentricVector ), 3.0 * std::numeric_limits< double >::epsilon( ) );

    // Case 1
    topocentricVector = ( Eigen::Vector3d( ) << -37054487.969432726502, -51001127.313443794847, -49252918.02605265379 ).finished( );
    expectedAzimuth = ( 216.0 - 360.0 ) * degreesToRadians;
    expectedElevation = -38.0 * degreesToRadians;

    BOOST_CHECK_CLOSE_FRACTION( expectedAzimuth, pointAnglesCalculator->calculateAzimuthAngle( topocentricVector ), 3.0 * std::numeric_limits< double >::epsilon( ) );
    BOOST_CHECK_CLOSE_FRACTION( expectedElevation, pointAnglesCalculator->calculateElevationAngle( topocentricVector ), 3.0 * std::numeric_limits< double >::epsilon( ) );
}

BOOST_AUTO_TEST_CASE( test_PointingAnglesCalculator )
{
    // Define Earth shape model (sphere)
    std::shared_ptr< SphericalBodyShapeModel > bodyShape = std::make_shared< SphericalBodyShapeModel >( 6.371E6 );

    // Define test ground station point.
    double stationAltitude = 0.0;
    double stationLatitude = convertDegreesToRadians( 0.0 ); //lat;
    double stationLongitude = convertDegreesToRadians( 0.0 ); //lon;

    double degreesToRadians = unit_conversions::convertDegreesToRadians( 1.0 );

    // Test analytically checked azimuth and elevation
    {
        // Create ground station properties
        std::shared_ptr< GroundStationState > stationState = std::make_shared< GroundStationState >(
                    ( Eigen::Vector3d( ) << stationAltitude, stationLatitude, stationLongitude ).finished( ),
                    coordinate_conversions::geodetic_position, bodyShape );
        std::shared_ptr< PointingAnglesCalculator > pointAnglesCalculator = std::make_shared< PointingAnglesCalculator >(
                    [ & ]( const double ){ return Eigen::Quaterniond( Eigen::Matrix3d::Identity( ) ); },
                    std::bind( &GroundStationState::getRotationFromBodyFixedToTopocentricFrame, stationState, std::placeholders::_1 ) );

        // Define state of viewed point
        double testLatitude = 30.0 * degreesToRadians;
        double testLongitude = 0.0 * degreesToRadians;
        double testRadius = 8.0E7;
        Eigen::Vector3d testSphericalPoint( testRadius, mathematical_constants::PI / 2.0 - testLatitude, testLongitude );
        Eigen::Vector3d testCartesianPoint = coordinate_conversions::convertSphericalToCartesian( testSphericalPoint );

        // Compute azimuth/elevation angles from PointingAnglesCalculator
        double testAzimuth = pointAnglesCalculator->calculateAzimuthAngle( testCartesianPoint, 0.0 );
        double testElevation = pointAnglesCalculator->calculateElevationAngle( testCartesianPoint, 0.0 );

        double expectedAzimuth = 0.0 * degreesToRadians;
        double expectedElevation = 60.0 * degreesToRadians;

        std::cout<<testElevation / degreesToRadians<<" "<<testAzimuth / degreesToRadians<<std::endl;

        BOOST_CHECK_SMALL( std::fabs( expectedAzimuth - testAzimuth ), 1.0E-12 );
        BOOST_CHECK_SMALL( std::fabs( expectedElevation - testElevation ), 1.0E-12 );
    }
}


BOOST_AUTO_TEST_CASE( test_PointingAnglesCalculatorHorizons )
{
    loadStandardSpiceKernels( );

    // Set observation time
    DateTime observationDateTime = DateTime(
        2023, 12, 30, 0, 47, 0.0L );
    std::shared_ptr< earth_orientation::TerrestrialTimeScaleConverter > timeConverter =
        earth_orientation::createDefaultTimeConverter( );
    double observationTime = timeConverter->getCurrentTime( utc_scale, tdb_scale, observationDateTime.epoch< double >( ) );


    std::string globalFrameOrientation = "J2000";
    std::string globalFrameOrigin = "Earth";


    // Create bodies
    std::vector< std::string > bodiesToCreate = { "Earth", "Jupiter" };
    BodyListSettings bodySettings = getDefaultBodySettings( bodiesToCreate, globalFrameOrigin, globalFrameOrientation );
    bodySettings.at( "Earth" )->rotationModelSettings = gcrsToItrsRotationModelSettings( basic_astrodynamics::iau_2006, "J2000" );
    SystemOfBodies bodies = createSystemOfBodies( bodySettings );


    // Create ground station
    double stationAltitude = 6378.0E3;
    double stationLatitude = convertDegreesToRadians( 60.0 ); //lat;
    double stationLongitude = convertDegreesToRadians( 0.0 ); //lon;

    std::pair< std::string, std::string > station = std::pair< std::string, std::string >( "Earth", "Station" );
    createGroundStation( bodies.at( "Earth" ), "Station", ( Eigen::Vector3d( ) << stationAltitude, stationLatitude, stationLongitude ).finished( ),
                         coordinate_conversions::spherical_position );

    // Retrieve pointing angle calculator
    std::shared_ptr< PointingAnglesCalculator > pointingAngleCalculator =
        bodies.at( "Earth" )->getGroundStation( "Station" )->getPointingAnglesCalculator( );


    // Get inertial ground station state function
    std::function< Eigen::Vector6d( const double ) > groundStationStateFunction =
        getLinkEndCompleteEphemerisFunction( std::make_pair< std::string, std::string >( "Earth", "Station" ), bodies );
    std::function< Eigen::Vector6d( const double ) > targetFunction =
        getLinkEndCompleteEphemerisFunction( std::make_pair< std::string, std::string >( "Jupiter", "" ), bodies );

    Eigen::Vector6d jupiterState = targetFunction( observationTime );
    Eigen::Vector6d groundStationState = groundStationStateFunction( observationTime );
    Eigen::Vector6d relativeState = jupiterState - groundStationState;


    double horizonsElevationInDegrees = 17.443175;
    BOOST_CHECK_SMALL( horizonsElevationInDegrees - convertRadiansToDegrees(
        pointingAngleCalculator->calculateElevationAngle( relativeState.segment(0, 3), observationTime ) ), 5.0E-3 );

    double horizonsAzimuthInDegrees = 264.303204;
    BOOST_CHECK_SMALL( horizonsAzimuthInDegrees - ( convertRadiansToDegrees(
        pointingAngleCalculator->calculateAzimuthAngle( relativeState.segment(0, 3), observationTime ) ) +360.0 ), 5.0E-3 );


}

BOOST_AUTO_TEST_SUITE_END( )

}

}
