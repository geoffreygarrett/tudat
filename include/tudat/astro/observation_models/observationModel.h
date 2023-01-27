/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_OBSERVATIONMODEL_H
#define TUDAT_OBSERVATIONMODEL_H

#include <vector>


#include <memory>
#include <functional>

#include <Eigen/Core>

#include "tudat/basics/basicTypedefs.h"
#include "tudat/basics/timeType.h"
#include "tudat/basics/tudatTypeTraits.h"
#include "tudat/basics/utilities.h"

#include "tudat/astro/observation_models/linkTypeDefs.h"
#include "tudat/astro/observation_models/observableTypes.h"
#include "tudat/astro/observation_models/observationBias.h"



namespace tudat
{

namespace observation_models
{


enum ObservationAncilliarySimulationVariable
{
    doppler_integration_time,
    retransmission_delays
};

template< typename TimeType = double >
struct ObservationAncilliarySimulationSettings
{
public:
    ObservationAncilliarySimulationSettings( ){ }

    virtual ~ObservationAncilliarySimulationSettings( ){ }

    void setAncilliaryDoubleData( const ObservationAncilliarySimulationVariable& variableType, const double variable )
    {
        switch( variableType )
        {
        case doppler_integration_time:
            doubleData_[ doppler_integration_time ] = variable;
            break;
        default:
            throw std::runtime_error( "Error when setting double ancilliary observation data; could not set type " +
                                      std::to_string( variableType ) );
        }
    }

    void setAncilliaryDoubleVectorData( const ObservationAncilliarySimulationVariable& variableType,
                                        const std::vector< double >& variable )
    {
        switch( variableType )
        {
        case retransmission_delays:
            doubleVectorData_[ retransmission_delays ] = variable;
            break;
        default:
            throw std::runtime_error( "Error when setting double vector ancilliary observation data; could not set type " +
                                      std::to_string( variableType ) );
        }
    }

    double getAncilliaryDoubleData( const ObservationAncilliarySimulationVariable& variableType,
                                    const bool throwException = true )
    {
        double returnVariable = TUDAT_NAN;
        try
        {
            switch( variableType )
            {
            case doppler_integration_time:
                returnVariable = doubleData_.at( doppler_integration_time );
                break;
            default:
                break;
            }
        }
        catch( ... )
        {
            if( throwException )
            {
                throw std::runtime_error( "Error when getting double ancilliary observation data; could not retrieve type " +
                                          std::to_string( variableType ) );
            }
        }
        return returnVariable;
    }

    std::vector< double > getAncilliaryDoubleVectorData( const ObservationAncilliarySimulationVariable& variableType,
                                                         const bool throwException = true )
    {
        std::vector< double > returnVariable = std::vector< double >( );
        try
        {
            switch( variableType )
            {
            case retransmission_delays:
                returnVariable = doubleVectorData_.at( retransmission_delays );
                break;
            default:
                break;
            }
        }
        catch( ... )
        {
            if( throwException )
            {
                throw std::runtime_error( "Error when getting double vector ancilliary observation data; could not retrieve type " +
                                          std::to_string( variableType ) );
            }
        }
        return returnVariable;
    }

protected:

    std::map< ObservationAncilliarySimulationVariable, double > doubleData_;

    std::map< ObservationAncilliarySimulationVariable, std::vector< double > > doubleVectorData_;

};

template< typename TimeType = double >
std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > getAveragedDopplerAncilliarySettings(
        const TimeType integrationTime = 60.0 )
{
    std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > ancilliarySettings =
            std::make_shared< ObservationAncilliarySimulationSettings< TimeType > >( );
    ancilliarySettings->setAncilliaryDoubleData( doppler_integration_time, integrationTime );
    return ancilliarySettings;
}

template< typename TimeType = double >
std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > getNWayRangeAncilliarySettings(
        const std::vector< double > retransmissionTimes = std::vector< double >( ) )
{
    std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > ancilliarySettings =
            std::make_shared< ObservationAncilliarySimulationSettings< TimeType > >( );
    ancilliarySettings->setAncilliaryDoubleVectorData( retransmission_delays, retransmissionTimes );
    return ancilliarySettings;
}

template< typename TimeType = double >
std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > getNWayAveragedDopplerAncilliarySettings(
        const TimeType integrationTime = 60.0,
        const std::vector< double > retransmissionTimes = std::vector< double >( ) )
{
    std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > ancilliarySettings =
            std::make_shared< ObservationAncilliarySimulationSettings< TimeType > >( );
    ancilliarySettings->setAncilliaryDoubleData( doppler_integration_time, integrationTime );
    ancilliarySettings->setAncilliaryDoubleVectorData( retransmission_delays, retransmissionTimes );
    return ancilliarySettings;
}

template< typename TimeType = double >
std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > getTwoWayRangeAncilliarySettings(
        const double retransmissionTime )
{
    return getNWayRangeAncilliarySettings< TimeType >( std::vector< double >( { retransmissionTime } ) );
}

template< typename TimeType = double >
std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > getTwoWayAveragedDopplerAncilliarySettings(
        const TimeType integrationTime = 60.0,
        const double retransmissionTime = 0.0 )
{
    return getNWayAveragedDopplerAncilliarySettings< TimeType >( integrationTime, std::vector< double >( { retransmissionTime } ) );

}




template< typename TimeType = double >
std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > getDefaultAncilliaryObservationSettings(
        const observation_models::ObservableType observableType )
{
    std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > ancilliarySettings = nullptr;
    switch( observableType )
    {
    case observation_models::one_way_differenced_range:
        ancilliarySettings = getAveragedDopplerAncilliarySettings< TimeType >( 60.0 );
        break;
    case observation_models::n_way_differenced_range:
        ancilliarySettings = getAveragedDopplerAncilliarySettings< TimeType >( 60.0 );
        break;
    default:
        break;
    }
    return ancilliarySettings;
}



//! Base class for models of observables (i.e. range, range-rate, etc.).
/*!
 *  Base class for models of observables to be used in (for instance) orbit determination.
 *  Each type of observables (1-way range, 2-way range, Doppler, VLBI, etc.) has its own
 *  derived class capable of simulating observables of the given type using given link ends.
 *  The functions to be used for computing the observables can be called with/without deviations from ideal observable
 *  (see base class member functions). Corrections are computed from an observationBiasCalculator member object, which is
 *  empty by default. Also, the observable may be a with/without returning (by reference) the times and states
 *  at each of the link ends. Returning these times/states prevents recomputations of these quantities in later calculations.
 */
template< int ObservationSize = Eigen::Dynamic, typename ObservationScalarType = double, typename TimeType = double,
          typename std::enable_if< is_state_scalar_and_time_type< ObservationScalarType, TimeType >::value, int >::type = 0 >
class ObservationModel
{
public:

    //! Constructor
    /*!
     * Base class constructor.
     * \param observableType Type of observable, used for derived class type identification without
     * explicit casts.
     * \param observationBiasCalculator Object for calculating system-dependent errors in the
     * observable, i.e. deviations from the physically ideal observable between reference points (default none).
     */
    ObservationModel(
            const ObservableType observableType,
            const LinkEnds linkEnds,
            const std::shared_ptr< ObservationBias< ObservationSize > > observationBiasCalculator = nullptr ):
        observableType_( observableType ),
        linkEnds_( linkEnds ),
        observationBiasCalculator_( observationBiasCalculator )
    {
        // Check if bias is empty
        if( observationBiasCalculator_ != nullptr )
        {
            isBiasnullptr_ = 0;
            if( observationBiasCalculator_->getObservationSize( ) != ObservationSize )
            {
                throw std::runtime_error( "Error when making observation model, bias size is inconsistent" );
            }
        }
        else
        {
            isBiasnullptr_ = 1;
        }

        // Check if time biases are included and retrieve time biases functions
        retrieveTimeBiasesFunctions( );
    }

    //! Virtual destructor
    virtual ~ObservationModel( ) { }

    //! Function to return the type of observable.
    /*!
     *  Function to return the type of observable.
     *  \return Type of observable.
     */
    ObservableType getObservableType( )
    {
        return observableType_;
    }

    LinkEnds getLinkEnds( )
    {
        return linkEnds_;
    }

    //! Function to compute the observable without any corrections
    /*!
     * Function to compute the observable without any corrections, i.e. the ideal physical observable as computed
     *  from the defined link ends (in the derived class). Note that this observable does include e.g. light-time
     *  corrections, which represent physically true corrections. It does not include e.g. system-dependent measurement
     *  errors, such as biases or clock errors.
     *  The times and states of the link ends are also returned in full precision (determined by class template
     *  arguments). These states and times are returned by reference.
     *  \param time Time at which observable is to be evaluated.
     *  \param linkEndAssociatedWithTime Link end at which given time is valid, i.e. link end for which associated time
     *  is kept constant (to input value)
     *  \param linkEndTimes List of times at each link end during observation (returned by reference).
     *  \param linkEndStates List of states at each link end during observation (returned by reference).
     *  \return Ideal observable.
     */
    virtual Eigen::Matrix< ObservationScalarType, ObservationSize, 1 > computeIdealObservationsWithLinkEndData(
            const TimeType time,
            const LinkEndType linkEndAssociatedWithTime,
            std::vector< double >& linkEndTimes,
            std::vector< Eigen::Matrix< double, 6, 1 > >& linkEndStates,
            const std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > ancilliarySetings = nullptr ) = 0;

    //! Function to compute full observation at given time.
    /*!
     *  Function to compute observation at given time (include any defined non-ideal corrections). The
     *  times and states of the link ends are given in full precision (determined by class template
     *  arguments). These states and times are returned by reference.
     *  \param time Time at which observation is to be simulated
     *  \param linkEndAssociatedWithTime Link end at which current time is measured, i.e. reference
     *  link end for observable.
     *  \param linkEndTimes List of times at each link end during observation (returned by reference).
     *  \param linkEndStates List of states at each link end during observation (returned by reference).
     *  \return Calculated observable value.
     */
    Eigen::Matrix< ObservationScalarType, ObservationSize, 1 > computeObservationsWithLinkEndData(
            const TimeType time,
            const LinkEndType linkEndAssociatedWithTime,
            std::vector< double >& linkEndTimes ,
            std::vector< Eigen::Matrix< double, 6, 1 > >& linkEndStates,
            const std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > ancilliarySetings = nullptr )
    {
        // Add time bias if necessary
        TimeType observationTime = computeBiasedObservationTime( time );

        // Check if any non-ideal models are set.
        if( isBiasnullptr_ )
        {
            return computeIdealObservationsWithLinkEndData(
                        observationTime /*time*/, linkEndAssociatedWithTime, linkEndTimes, linkEndStates, ancilliarySetings );
        }
        else
        {
            // Check that time biases are associated with the time reference time link.
            checkReferenceLinkEndForTimeBiases( linkEndAssociatedWithTime );

            // Compute ideal observable
            Eigen::Matrix< ObservationScalarType, ObservationSize, 1 > currentObservation = computeIdealObservationsWithLinkEndData(
                    observationTime /*time*/, linkEndAssociatedWithTime, linkEndTimes, linkEndStates, ancilliarySetings );

            // Add correction
            return currentObservation + this->observationBiasCalculator_->getObservationBias(
                    linkEndTimes, linkEndStates, currentObservation.template cast< double >( ) ). template cast< ObservationScalarType >( );
        }
    }

    //! Function to compute the observable without any corrections.
    /*!
     * Function to compute the observable without any corrections, i.e. the ideal physical observable as computed
     * from the defined link ends (in the derived class). Note that this observable does include e.g. light-time
     * corrections, which represent physically true corrections. It does not include e.g. system-dependent measurement
     * errors, such as biases or clock errors. This function may be redefined in derived class for improved efficiency.
     * \param time Time at which observable is to be evaluated.
     * \param linkEndAssociatedWithTime Link end at which given time is valid, i.e. link end for which associated time
     * is kept constant (to input value)
     * \return Ideal observable.
     */
    virtual Eigen::Matrix< ObservationScalarType, ObservationSize, 1 > computeIdealObservations(
            const TimeType time,
            const LinkEndType linkEndAssociatedWithTime,
            const std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > ancilliarySetings = nullptr )
    {
        // Compute ideal observable from derived class.
        return this->computeIdealObservationsWithLinkEndData(
                    time, linkEndAssociatedWithTime, this->linkEndTimes_, this->linkEndStates_, ancilliarySetings );
    }

    //! Function to compute full observation at given time.
    /*!
     *  Function to compute observation at given time (include any defined non-ideal corrections).
     * \param time Time at which observable is to be evaluated.
     * \param linkEndAssociatedWithTime Link end at which given time is valid, i.e. link end for which associated time
     * is kept constant (to input value)
     *  \return Calculated (non-ideal) observable value.
     */
    Eigen::Matrix< ObservationScalarType, ObservationSize, 1 > computeObservations(
            const TimeType time,
            const LinkEndType linkEndAssociatedWithTime,
            const std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > ancilliarySetings = nullptr )
    {
        // Check if any non-ideal models are set.
        if( isBiasnullptr_ )
        {
            return computeIdealObservationsWithLinkEndData(
                        time, linkEndAssociatedWithTime, linkEndTimes_, linkEndStates_, ancilliarySetings );
        }
        else
        {
            // Add time bias if necessary
            TimeType observationTime = computeBiasedObservationTime( time );

            // Compute ideal observable
            Eigen::Matrix< ObservationScalarType, ObservationSize, 1 > currentObservation = computeIdealObservationsWithLinkEndData(
                    observationTime /*time*/, linkEndAssociatedWithTime, linkEndTimes_, linkEndStates_, ancilliarySetings );

            // Add correction
            return currentObservation + this->observationBiasCalculator_->getObservationBias(
                    linkEndTimes_, linkEndStates_, currentObservation.template cast< double >( ) ).template cast< ObservationScalarType >( );
        }
    }

    //! Function to retrieve a single entry of the observation value
    /*!
     *  Function to retrieve a single entry of the observation value. Generally, the observable is a vector, this function
     *  allows a single entry to be retrieve
     *  \param time Time at which observable is to be evaluated.
     *  \param linkEndAssociatedWithTime Link end at which given time is valid, i.e. link end for which associated time
     *  is kept constant (to input value)
     *  \param observationEntry entry from observable vector that is to be retrieved.
     *  \return Calculated (non-ideal, i.e with biases) observable value.
     */
    ObservationScalarType computeObservationEntry(
            const TimeType time,
            const LinkEndType linkEndAssociatedWithTime,
            const int observationEntry,
            const std::shared_ptr< ObservationAncilliarySimulationSettings< TimeType > > ancilliarySetings = nullptr )
    {
        if( observationEntry < ObservationSize )
        {
            return computeObservations( time, linkEndAssociatedWithTime, ancilliarySetings )( observationEntry );
        }
        else
        {
            throw std::runtime_error( "Error, requesting out-of-bounds index for observation model" );
        }
    }

    TimeType computeBiasedObservationTime( const double observationTime )
    {
        double updatedObservationTime = observationTime;
        if ( !isTimeBiasNullptr_ )
        {
            for ( unsigned int i = 0 ; i < getTimeBiasFunctions_.size( ) ; i++ )
            {
                updatedObservationTime -= getTimeBiasFunctions_[ i ]( observationTime );
            }
        }

        return updatedObservationTime;
    }

    //! Function to return the size of the observable
    /*!
     *  Function to return the size of the observable
     *  \return Size of the observable
     */
    int getObservationSize( )
    {
        return ObservationSize;
    }

    //! Functiomn to return the object for calculating system-dependent errors in the observable.
    /*!
     * Functiomn to return the object for calculating system-dependent errors in the observable.
     * \return Object for calculating system-dependent errors in the observable.
     */
    std::shared_ptr< ObservationBias< ObservationSize > > getObservationBiasCalculator( )
    {
        return observationBiasCalculator_;
    }


protected:

    void checkReferenceLinkEndForTimeBiases( const LinkEndType linkEndAssociatedWithTime ) const
    {
        std::vector< int > indexRefLinkEndTimeBias;
        std::shared_ptr< observation_models::ObservationBias< ObservationSize > > biasCalculator = this->observationBiasCalculator_;
        if ( std::dynamic_pointer_cast< observation_models::ConstantTimeBias< ObservationSize > >( biasCalculator ) != nullptr )
        {
            indexRefLinkEndTimeBias.push_back( std::dynamic_pointer_cast< observation_models::ConstantTimeBias< ObservationSize > >( biasCalculator )->getLinkEndIndexForTime( ) );
        }
        else if ( std::dynamic_pointer_cast< observation_models::ArcWiseTimeBias< ObservationSize > >( biasCalculator ) != nullptr )
        {
            indexRefLinkEndTimeBias.push_back( std::dynamic_pointer_cast< observation_models::ArcWiseTimeBias< ObservationSize > >( biasCalculator )->getLinkEndIndexForTime( ) );
        }
        else if ( std::dynamic_pointer_cast< observation_models::MultiTypeObservationBias< ObservationSize > >( biasCalculator ) != nullptr )
        {
            std::vector< std::shared_ptr< observation_models::ObservationBias< ObservationSize > > > biasList =
                    std::dynamic_pointer_cast< observation_models::MultiTypeObservationBias< ObservationSize > >( biasCalculator )->getBiasList( );
            for ( unsigned int j = 0 ; j < biasList.size( ) ; j++ )
            {
                if ( std::dynamic_pointer_cast< observation_models::ConstantTimeBias< ObservationSize > >( biasList[ j ] ) != nullptr )
                {
                    indexRefLinkEndTimeBias.push_back( std::dynamic_pointer_cast< observation_models::ConstantTimeBias< ObservationSize > >( biasList[ j ] )->getLinkEndIndexForTime( ) );
                }
                else if ( std::dynamic_pointer_cast< observation_models::ArcWiseTimeBias< ObservationSize > >( biasList[ j ] ) != nullptr )
                {
                    indexRefLinkEndTimeBias.push_back( std::dynamic_pointer_cast< observation_models::ArcWiseTimeBias< ObservationSize > >( biasList[ j ] )->getLinkEndIndexForTime( ) );
                }
            }
        }

        for ( unsigned int j = 0 ; j < indexRefLinkEndTimeBias.size( ) ; j++ )
        {
            if ( indexRefLinkEndTimeBias[ j ] != observation_models::getLinkEndIndicesForLinkEndTypeAtObservable(
                    observableType_, linkEndAssociatedWithTime, linkEnds_.size( ) ).at( 0 ) )
            {
                throw std::runtime_error( "Error when setting time biases, ref link end is different from time ref link in observation simulation settings." );
            }
        }
    }

    void retrieveTimeBiasesFunctions( )
    {
        isTimeBiasNullptr_ = true;

        if ( std::dynamic_pointer_cast< MultiTypeObservationBias< ObservationSize > >( this->observationBiasCalculator_) != nullptr )
        {
            std::vector< std::shared_ptr< ObservationBias< ObservationSize > > > biasList =
                    std::dynamic_pointer_cast< MultiTypeObservationBias< ObservationSize > >( this->observationBiasCalculator_)->getBiasList( );
            for ( unsigned int i = 0 ; i < biasList.size( ) ; i++ )
            {
                if ( std::dynamic_pointer_cast< ConstantTimeBias< ObservationSize > >( biasList[ i ] ) != nullptr )
                {
                    isTimeBiasNullptr_ = false;
                    getTimeBiasFunctions_.push_back(
                            std::bind( &observation_models::ConstantTimeBias< ObservationSize >::getConstantTimeBias,
                                       std::dynamic_pointer_cast< ConstantTimeBias< ObservationSize > >( biasList[ i ] ), std::placeholders::_1 ) );
                }

                else if ( std::dynamic_pointer_cast< ArcWiseTimeBias< ObservationSize > >( biasList[ i ] ) != nullptr )
                {
                    isTimeBiasNullptr_ = false;
                    getTimeBiasFunctions_.push_back(
                            std::bind( &observation_models::ArcWiseTimeBias< ObservationSize >::getArcWiseTimeBias,
                                       std::dynamic_pointer_cast< ArcWiseTimeBias< ObservationSize > >( biasList[ i ] ), std::placeholders::_1 ) );
                }
            }
        }
        else if ( std::dynamic_pointer_cast< ConstantTimeBias< ObservationSize > >( this->observationBiasCalculator_ ) != nullptr )
        {
            isTimeBiasNullptr_ = false;
            getTimeBiasFunctions_.push_back( std::bind( &observation_models::ConstantTimeBias< ObservationSize >::getConstantTimeBias,
                                                        std::dynamic_pointer_cast< ConstantTimeBias< ObservationSize > >( this->observationBiasCalculator_ ), std::placeholders::_1 ) );
        }
        else if ( std::dynamic_pointer_cast< ArcWiseTimeBias< ObservationSize > >( this->observationBiasCalculator_ ) != nullptr )
        {
            isTimeBiasNullptr_ = false;
            getTimeBiasFunctions_.push_back( std::bind( &observation_models::ArcWiseTimeBias< ObservationSize >::getArcWiseTimeBias,
                                                        std::dynamic_pointer_cast< ArcWiseTimeBias< ObservationSize > >( this->observationBiasCalculator_ ), std::placeholders::_1 ) );
        }
    }

    //! Type of observable, used for derived class type identification without explicit casts.
    ObservableType observableType_;

    LinkEnds linkEnds_;

    //! Object for calculating system-dependent errors in the observable.
    /*!
     *  Object for calculating system-dependent errors in the observable, i.e. deviations from the
     *  physically true observable
     */
    std::shared_ptr< ObservationBias< ObservationSize > > observationBiasCalculator_;

    //! Boolean set by constructor to denote whether observationBiasCalculator_ is nullptr.
    bool isBiasnullptr_;


    //! Pre-define list of times used when calling function returning link-end states/times from interface function.
    std::vector< double > linkEndTimes_;

    //! Pre-define list of states used when calling function returning link-end states/times from interface function.
    std::vector< Eigen::Matrix< double, 6, 1 > > linkEndStates_;

    //! Boolean set by constructor denoting whether time biases are included.
    bool isTimeBiasNullptr_;

    //! List of time biases functions, if any.
    std::vector< std::function< double( const double ) > > getTimeBiasFunctions_;

};

extern template class ObservationModel< 1, double, double >;
extern template class ObservationModel< 2, double, double >;
extern template class ObservationModel< 3, double, double >;
extern template class ObservationModel< 6, double, double >;


//! Function to compute an observation of size 1 at double precision, with double precision input
/*!
 *  Function to compute an observation at double precision, with double precision input, from an observation function
 *  templated at state scalar and time type.
 *  \param observationFunction Function that computes the observation as a function of observation time and reference link end
 *  time, templated by the state and time scalar type.
 *  \param currentTime Time at which to evaluate the observation function
 *  \param referenceLinkEnd Reference link end for the observation
 *  \return Observation computed by observationFunction, cast to double precision, with input time at double precision
 */
template< typename ObservationScalarType = double, typename TimeType = double >
double getSizeOneObservationAtDoublePrecision(
        std::function< Eigen::Matrix< ObservationScalarType, 1, 1 >( const TimeType, const observation_models::LinkEndType ) >
        observationFunction, const double currentTime, const LinkEndType referenceLinkEnd )
{
    return static_cast< double >( observationFunction( static_cast< TimeType >( currentTime ), referenceLinkEnd )( 0 ) );
}

//! Function to generate a function that computes a size 1 observation at double precision, from a templated observation function.
/*!
 *  Function to generate a function that computes a size 1 observation at double precision, from a templated observation function.
 *  \param observationFunction Function that computes the observation as a function of observation time and reference link end
 *  time, templated by the state and time scalar type.
 *  \return Function that computes the observation as a function of observation time and reference link end time.
 */
template< typename ObservationScalarType = double, typename TimeType = double >
std::function< double( const double, const observation_models::LinkEndType ) > getSizeOneObservationFunctionAtDoublePrecision(
        std::function< Eigen::Matrix< ObservationScalarType, 1, 1 >(
            const TimeType, const observation_models::LinkEndType ) > observationFunction )
{
    return std::bind( &getSizeOneObservationAtDoublePrecision< ObservationScalarType, TimeType >, observationFunction, std::placeholders::_1, std::placeholders::_2 );
}

//! Function to generate a function that computes an observation  from an ObservationModel
/*!
 *  Function to generate a function that produces an observation, only applicable for observation models
 *  of size one. This function uses std::bind to link the computeObservations function of the observationModel to the output
 *  of this function.
 *  \param observationModel Observation model for which the observation function is to be returned.
 *  \return Function that computes the observation as a function of observation time and reference link end time.
 */
template< typename ObservationScalarType = double, typename TimeType = double >
std::function< Eigen::Matrix< ObservationScalarType, 1, 1 >( const TimeType, const observation_models::LinkEndType ) >
getSizeOneObservationFunctionFromObservationModel(
        const std::shared_ptr< ObservationModel< 1, ObservationScalarType, TimeType > > observationModel )
{
    return std::bind( &ObservationModel< 1, ObservationScalarType, TimeType >::computeObservations, observationModel, std::placeholders::_1, std::placeholders::_2, nullptr );
}

//! Function to generate a function that computes an observation at double precision from an ObservationModel
/*!
 *  Function to generate a function that computes an observation at double precision, only applicable for observation models
 *  of size one. This function uses std::bind to link the computeObservations function of the observationModel to the output
 *  of this function, casting in/and output to double precisiono if needed.
 *  \param observationModel Observation model for which the observation function is to be returned.
 *  \return Function that computes the observation as a function of observation time and reference link end time.
 */
template< typename ObservationScalarType = double, typename TimeType = double >
std::function< double( const double, const observation_models::LinkEndType ) >
getSizeOneObservationFunctionAtDoublePrecisionFromObservationModel(
        const std::shared_ptr< ObservationModel< 1, ObservationScalarType, TimeType > > observationModel )
{
    return getSizeOneObservationFunctionAtDoublePrecision(
                getSizeOneObservationFunctionFromObservationModel( observationModel ) );
}

//! Function to extract a list of observtion bias models from a list of observation models.
/*!
 *  Function to extract a list of observtion bias models from a list of observation models. Function iterates over input
 *  map of observationModels, extracts the bias from it and adds it to the list of bias objects if it is not nullptr.
 *  \param observationModels List of observation models (per LinkEnds) from which the bias objects are to be extracted
 *  \return List of observation bias objects (per LinkEnds), as extracted from observationModels (nullptr bias objects not
 *  added to list).
 */
template< int ObservationSize = Eigen::Dynamic, typename ObservationScalarType = double, typename TimeType = double >
std::map< LinkEnds, std::shared_ptr< ObservationBias< ObservationSize > > > extractObservationBiasList(
        std::map< LinkEnds, std::shared_ptr< ObservationModel< ObservationSize, ObservationScalarType, TimeType > > >
        observationModels )
{
    std::map< LinkEnds, std::shared_ptr< ObservationBias< ObservationSize > > > biasList;
    for( typename std::map< LinkEnds, std::shared_ptr<
         ObservationModel< ObservationSize, ObservationScalarType, TimeType > > >::const_iterator
         observationModelIterator = observationModels.begin( ); observationModelIterator != observationModels.end( );
         observationModelIterator++ )
    {
        if( observationModelIterator->second->getObservationBiasCalculator( ) != nullptr )
        {
            biasList[ observationModelIterator->first ] = observationModelIterator->second->getObservationBiasCalculator( );
        }
    }
    return biasList;
}

} // namespace observation_models

} // namespace tudat
#endif // TUDAT_OBSERVATIONMODEL_H
