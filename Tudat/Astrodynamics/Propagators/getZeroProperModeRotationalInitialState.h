#ifndef GETZEROPROPERMODEROTATIONALINITIALSTATE_H
#define GETZEROPROPERMODEROTATIONALINITIALSTATE_H

#include <Eigen/Core>

#include "Tudat/Astrodynamics/BasicAstrodynamics/dissipativeTorqueModel.h"
#include "Tudat/SimulationSetup/PropagationSetup/dynamicsSimulator.h"
#include "Tudat/InputOutput/basicInputOutput.h"

namespace tudat
{

namespace propagators
{

Eigen::Matrix3d getDissipationMatrix(
        const double dampingTime,
        const Eigen::Matrix3d& inertiaTensor )
{
    return inertiaTensor / dampingTime;
}

//class RotationalDissipationWrapper
//{
//public:
//    RotationalDissipationWrapper( const Eigen::Matrix3d& dissipationMatrix ):
//        dissipationMatrix_( dissipationMatrix ){ }

//    Eigen::Matrix3d getDissipationMatrix( )
//    {
//        return dissipationMatrix_;
//    }

//    void setDissipationMatrix( const Eigen::Matrix3d dissipationMatrix )
//    {
//        dissipationMatrix_ = dissipationMatrix;
//    }

//private:
//    Eigen::Matrix3d dissipationMatrix_;
//};


//template< typename StateScalarType = double, typename TimeType = double >
//std::map< TimeType, Eigen::Matrix< StateScalarType, 3, 3 > > getRotationMatrixHistoryFromRotationalStateHistory(
//        const std::map< TimeType, Eigen::Matrix< StateScalarType, Eigen::Dynamic, 1 > > rotationalStateHistory )
//{
//    std::map< TimeType, Eigen::Matrix< StateScalarType, 3, 3 > > rotationMatrixHistory;
//    for( typename std::map< TimeType, Eigen::Matrix< StateScalarType, Eigen::Dynamic, 1 > >::const_iterator stateIterator =
//         rotationalStateHistory.begin( ); stateIterator != rotationalStateHistory.end( ); stateIterator++ )
//    {
//        Eigen::Quaterniond currentRotation = ( Eigen::Quaterniond( Eigen::Vector4d( stateIterator->second.block( 0, 0, 4, 1 ) ) ) );

//        rotationMatrixHistory[ stateIterator->first ] = ( currentRotation ).toRotationMatrix( );
//    }
//    return rotationMatrixHistory;
//}


template< typename StateScalarType = double, typename TimeType = double >
void integrateForwardWithDissipationAndBackwardsWithout(
        std::shared_ptr< SingleArcDynamicsSimulator< StateScalarType, TimeType > > dynamicsSimulator,
        std::shared_ptr< basic_astrodynamics::DissipativeTorqueModel > dissipativeTorque,
        std::pair< std::map< TimeType, Eigen::Matrix< StateScalarType, Eigen::Dynamic, 1> >,
        std::map< TimeType, Eigen::Matrix< StateScalarType, Eigen::Dynamic, 1 > > >& propagatedStates,
        std::pair< std::map< TimeType, Eigen::Matrix< double, Eigen::Dynamic, 1> >,
        std::map< TimeType, Eigen::Matrix< double, Eigen::Dynamic, 1 > > >& dependentVariables )
{
    using namespace tudat::numerical_integrators;

    std::shared_ptr< SingleArcPropagatorSettings< TimeType > > propagatorSettings = dynamicsSimulator->getPropagatorSettings( );
    Eigen::Matrix< StateScalarType, Eigen::Dynamic, 1> initialState = propagatorSettings->getInitialStates( );
    std::shared_ptr< numerical_integrators::IntegratorSettings< TimeType > > integratorSettings =
            dynamicsSimulator->getIntegratorSettings( );

    dynamicsSimulator->integrateEquationsOfMotion( initialState );

    std::map< TimeType, Eigen::Matrix< StateScalarType, Eigen::Dynamic, 1 > > forwardIntegrated =
            dynamicsSimulator->getEquationsOfMotionNumericalSolution( );
    std::map< TimeType, Eigen::Matrix< double, Eigen::Dynamic, 1 > > forwardIntegratedDependent =
            dynamicsSimulator->getDependentVariableHistory( );


    TimeType originalStartTime = integratorSettings->initialTime_;
    TimeType originalEndTime = std::dynamic_pointer_cast< PropagationTimeTerminationSettings >(
                propagatorSettings->getTerminationSettings( ) )->terminationTime_;
    double originalTimeStep = integratorSettings->initialTimeStep_;

    dissipativeTorque->setDampingMatrixFunction( Eigen::Matrix3d::Zero( ) );

    auto outputMapIterator = forwardIntegrated.rbegin( );

    integratorSettings->initialTime_ = outputMapIterator->first;
    dynamicsSimulator->resetInitialPropagationTime( outputMapIterator->first );
    propagatorSettings->resetTerminationSettings(
                std::make_shared< PropagationTimeTerminationSettings >( originalStartTime ) );
    integratorSettings->initialTimeStep_ = -originalTimeStep;


    dynamicsSimulator->integrateEquationsOfMotion(
                outputMapIterator->second );
    std::map< TimeType, Eigen::Matrix< StateScalarType, Eigen::Dynamic, 1> > backwardIntegrated =
            dynamicsSimulator->getEquationsOfMotionNumericalSolution( );
    std::map< TimeType, Eigen::Matrix< double, Eigen::Dynamic, 1 > > backwardIntegratedDependent =
            dynamicsSimulator->getDependentVariableHistory( );

    integratorSettings->initialTime_ = originalStartTime;
    dynamicsSimulator->resetInitialPropagationTime( originalStartTime );

    propagatorSettings->resetTerminationSettings(
                std::make_shared< PropagationTimeTerminationSettings >( originalEndTime ) );

    integratorSettings->initialTimeStep_ = originalTimeStep;

    propagatedStates = std::make_pair( forwardIntegrated, backwardIntegrated );
    dependentVariables  = std::make_pair( forwardIntegratedDependent, backwardIntegratedDependent );
}

template< typename TimeType, typename StateScalarType >
Eigen::VectorXd getZeroProperModeRotationalState(
        const simulation_setup::NamedBodyMap& bodyMap,
        const std::shared_ptr< numerical_integrators::IntegratorSettings< TimeType > > integratorSettings,
        const std::shared_ptr< SingleArcPropagatorSettings< StateScalarType > > propagatorSettings,
        const double bodyMeanRotationRate,
        const std::vector< double > dissipationTimes,
        std::vector< std::pair< std::map< TimeType, Eigen::Matrix< StateScalarType, Eigen::Dynamic, 1 > >,
        std::map< TimeType, Eigen::Matrix< StateScalarType, Eigen::Dynamic, 1 > > > >& propagatedStates,
        std::vector< std::pair< std::map< TimeType, Eigen::Matrix< double, Eigen::Dynamic, 1 > >,
        std::map< TimeType, Eigen::Matrix< double, Eigen::Dynamic, 1 > > > >& dependentVariables,
        const bool propagateNominal = true )
{
    propagatedStates.resize( dissipationTimes.size( ) + 1 );
    dependentVariables.resize( dissipationTimes.size( ) + 1 );

    basic_astrodynamics::TorqueModelMap torqueModelMap;

    std::shared_ptr< RotationalStatePropagatorSettings< StateScalarType > > rotationPropagationSettings_ =
            std::dynamic_pointer_cast< RotationalStatePropagatorSettings< StateScalarType > >(
                propagatorSettings );
    if( rotationPropagationSettings_ == NULL )
    {
        std::shared_ptr< MultiTypePropagatorSettings< StateScalarType > > multiTypePropagatorSettings =
                std::dynamic_pointer_cast< MultiTypePropagatorSettings< StateScalarType > >( propagatorSettings );
        if( multiTypePropagatorSettings != NULL )
        {
            std::map< IntegratedStateType, std::vector< std::shared_ptr< SingleArcPropagatorSettings< StateScalarType > > > >
                    propagatorSettingsMap = multiTypePropagatorSettings->propagatorSettingsMap_;
            if( propagatorSettingsMap.count( rotational_state ) == 0 )
            {
                throw std::runtime_error( "Error when finding initial rotational state, no rotational dynamics in multi-type list" );
            }
            else if( propagatorSettingsMap.at( rotational_state ).size( ) != 1 )
            {
                throw std::runtime_error( "Error when finding initial rotational state, multiple rotational dynamics in multi-type list" );
            }
            else
            {
                rotationPropagationSettings_ = std::dynamic_pointer_cast< RotationalStatePropagatorSettings< StateScalarType > >(
                            propagatorSettingsMap.at( rotational_state ).at( 0 ) );
            }
        }
    }

    torqueModelMap = rotationPropagationSettings_->getTorqueModelsMap( );


    if( torqueModelMap.size( ) != 1 )
    {
        std::cerr<<"Error when finding initial rotational state, "<<torqueModelMap.size( )<<" bodies are propagated."<<std::endl;
    }

    Eigen::Matrix3d inertiaTensor = bodyMap.at( torqueModelMap.begin( )->first )->getBodyInertiaTensor( );

    std::shared_ptr< basic_astrodynamics::DissipativeTorqueModel > dissipativeTorque =
            std::make_shared< basic_astrodynamics::DissipativeTorqueModel >(
                std::bind( &simulation_setup::Body::getCurrentAngularVelocityVectorInLocalFrame,
                           bodyMap.at( torqueModelMap.begin( )->first ) ),
                boost::lambda::constant( Eigen::Matrix3d::Zero( ) ),
                bodyMeanRotationRate );
    torqueModelMap[ torqueModelMap.begin( )->first ][ torqueModelMap.begin( )->first ].push_back(
                dissipativeTorque );

    rotationPropagationSettings_->resetTorqueModelsMap( torqueModelMap );
    std::shared_ptr< SingleArcDynamicsSimulator< StateScalarType, TimeType > > dynamicsSimulator =
            std::make_shared< SingleArcDynamicsSimulator< StateScalarType, TimeType > >(
                bodyMap, integratorSettings, propagatorSettings, 0, 0, 0 );

    std::string dataFolder = "/home/dominic/Documents/Articles/PhobosCoupledDynamics/Data/Preliminary/";

    if( propagateNominal )
    {
        integrateForwardWithDissipationAndBackwardsWithout< StateScalarType, TimeType >(
                    dynamicsSimulator, dissipativeTorque, propagatedStates.at( 0 ), dependentVariables.at( 0 ) );
        int i = 0;
        input_output::writeDataMapToTextFile(
                    dependentVariables.at( i ).second,
                    "rotStateDependentVariables_stateOnly_damped_" + std::to_string( i ) + ".dat", dataFolder );
        input_output::writeDataMapToTextFile(
                    dependentVariables.at( i ).first,
                    "rotStateDependentVariables_stateOnly_damped_forward_" + std::to_string( i ) + ".dat", dataFolder );
        input_output::writeDataMapToTextFile(
                    propagatedStates.at( i ).second,
                    "rotState_stateOnly_damped_" + std::to_string( i ) + ".dat", dataFolder );
        input_output::writeDataMapToTextFile(
                    propagatedStates.at( i ).first,
                    "rotState_stateOnly_damped_forward_" + std::to_string( i ) + ".dat", dataFolder );

        propagatedStates[ 0 ].first.clear( );
        dependentVariables[ 0 ].first.clear( );

        propagatedStates[ 0 ].second.clear( );
        dependentVariables[ 0 ].second.clear( );
    }
    Eigen::Matrix< StateScalarType, Eigen::Dynamic, 1 > currentInitialState =
            propagatorSettings->getInitialStates( );

    std::cout<<"Initial state: "<< propagatorSettings->getInitialStates( ).transpose( )<<std::endl;

    double newFinalTime;

    for( unsigned int i = 0; i < dissipationTimes.size( ); i++ )
    {
        std::cout<<"Getting zero proper mode, iteration "<<i<<std::endl;

        dissipativeTorque->setDampingMatrixFunction(
                    getDissipationMatrix( dissipationTimes.at( i ), inertiaTensor ) );

        newFinalTime = integratorSettings->initialTime_ + 10.0 * dissipationTimes.at( i );

        propagatorSettings->resetTerminationSettings(
                    std::make_shared< PropagationTimeTerminationSettings >( newFinalTime ) );


        integrateForwardWithDissipationAndBackwardsWithout< StateScalarType, TimeType >(
                    dynamicsSimulator, dissipativeTorque, propagatedStates.at( i + 1 ), dependentVariables.at( i + 1 ) );

        currentInitialState = propagatedStates.at( i + 1 ).second.begin( )->second;
        std::cout<<"New initial state: "<< propagatedStates.at( i + 1 ).first.begin( )->second.transpose( )<<std::endl;

        propagatorSettings->resetInitialStates( currentInitialState );
        std::cout<<"New initial state: "<< propagatedStates.at( i + 1 ).second.begin( )->second.transpose( )<<std::endl;

        input_output::writeDataMapToTextFile(
                    dependentVariables.at( i + 1 ).second,
                    "rotStateDependentVariables_stateOnly_damped_" + std::to_string( i + 1  ) + ".dat", dataFolder );
        input_output::writeDataMapToTextFile(
                    dependentVariables.at( i ).first,
                    "rotStateDependentVariables_stateOnly_damped_forward_" + std::to_string( i + 1  ) + ".dat", dataFolder );
        input_output::writeDataMapToTextFile(
                    propagatedStates.at( i + 1  ).second,
                    "rotState_stateOnly_damped_" + std::to_string( i ) + ".dat", dataFolder );
        input_output::writeDataMapToTextFile(
                    propagatedStates.at( i + 1  ).first,
                    "rotState_stateOnly_damped_forward_" + std::to_string( i + 1  ) + ".dat", dataFolder );

        propagatedStates[ i + 1 ].first.clear( );
        dependentVariables[ i + 1 ].first.clear( );

        propagatedStates[ i + 1 ].second.clear( );
        dependentVariables[ i + 1 ].second.clear( );


    }
    return currentInitialState;
}

template< typename TimeType, typename StateScalarType >
Eigen::VectorXd getZeroProperModeRotationalState(
        const simulation_setup::NamedBodyMap& bodyMap,
        const std::shared_ptr< numerical_integrators::IntegratorSettings< TimeType > > integratorSettings,
        const std::shared_ptr< SingleArcPropagatorSettings< StateScalarType > > propagatorSettings,
        const double bodyMeanRotationRate,
        const std::vector< double > dissipationTimes,
        const bool propagateNominal = true )
{

    std::vector< std::pair< std::map< TimeType, Eigen::Matrix< StateScalarType, Eigen::Dynamic, 1 > >,
            std::map< TimeType, Eigen::Matrix< StateScalarType, Eigen::Dynamic, 1 > > > > propagatedStates;
    std::vector< std::pair< std::map< TimeType, Eigen::Matrix< double, Eigen::Dynamic, 1 > >,
            std::map< TimeType, Eigen::Matrix< double, Eigen::Dynamic, 1 > > > > dependentVariables;
    return getZeroProperModeRotationalState( bodyMap, integratorSettings, propagatorSettings, bodyMeanRotationRate, dissipationTimes,
                                             propagatedStates, dependentVariables, propagateNominal );
}

}

}

#endif // GETZEROPROPERMODEROTATIONALINITIALSTATE_H