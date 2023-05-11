/*    Copyright (c) 2010-2023, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

/*!
 * \file orbitDeterminationManager.h
 * \brief Top-level class for performing orbit determination.
 */

#ifndef TUDAT_ORBITDETERMINATIONMANAGER_H
#define TUDAT_ORBITDETERMINATIONMANAGER_H

#include <algorithm>

#include "tudat/astro/observation_models/observationManager.h"
#include "tudat/astro/orbit_determination/estimatable_parameters/initialTranslationalState.h"
#include "tudat/astro/orbit_determination/podInputOutputTypes.h"
#include "tudat/io/basicInputOutput.h"
#include "tudat/math/basic/leastSquaresEstimation.h"
#include "tudat/simulation/estimation_setup/createNumericalSimulator.h"
#include "tudat/simulation/estimation_setup/createObservationManager.h"
#include "tudat/simulation/estimation_setup/variationalEquationsSolver.h"
#include "tudat/simulation/propagation_setup/dependentVariablesInterface.h"

namespace tudat {

namespace simulation_setup {

//! Top-level class for performing orbit determination.
/*!
 *  This class handles the orbit determination process. It takes
 * propagation/estimation settings as input,
 *  creates required objects for the process, and performs parameter estimation using measurement data and
 *  related metadata (as EstimationInput) provided to the estimateParameters
 * function.
 */
template <typename ObservationScalarType = double,
          typename TimeType = double,
          typename std::enable_if<is_state_scalar_and_time_type<ObservationScalarType, TimeType>::value, int>::type = 0>
class OrbitDeterminationManager {
  public:
    // clang-format off
    // Type aliases for readability
    using ObservationVectorType =Eigen::Matrix<ObservationScalarType, Eigen::Dynamic, 1>;
    using ParameterVectorType =Eigen::Matrix<ObservationScalarType, Eigen::Dynamic, 1>;
    using IntegratorSettingsPointer =std::shared_ptr<numerical_integrators::IntegratorSettings<TimeType> >;
    using EstimatableParameterSetPointer =std::shared_ptr<estimatable_parameters::EstimatableParameterSet<ObservationScalarType> >;
    using PropagatorSettingsPointer = std::shared_ptr<propagators::PropagatorSettings<ObservationScalarType> >;
    using ObservationModelSettingsPointer =std::shared_ptr<observation_models::ObservationModelSettings>;
    using ObservationCollectionType =observation_models::ObservationCollection<ObservationScalarType,TimeType>;
    using ObservationCollectionPointer =std::shared_ptr<ObservationCollectionType>;
    using DataIteratorType = typename ObservationCollectionType::SortedObservationSets::value_type::second_type::value_type;
    using ObservationSimulatorBasePointer = std::shared_ptr<observation_models::ObservationSimulatorBase<ObservationScalarType, TimeType>>;
    using ObservationManagerBaseType = observation_models::ObservationManagerBase<ObservationScalarType, TimeType>;
    // clang-format on

    //! Preprocess deprecated integrator settings
    /*!
     *  Preprocesses deprecated integrator settings and returns a vector of updated integrator settings.
     *  \param parametersToEstimate Pointer to the set of parameters to estimate.
     *  \param integratorSettings Vector of deprecated integrator settings.
     *  \param propagatorSettings Pointer to the propagator settings.
     *  \param integratorIndexOffset Offset for the integrator index (default is 0).
     *  \return Vector of updated integrator settings.
     */
    std::vector<IntegratorSettingsPointer> preprocessDeprecatedIntegratorSettings(
        const EstimatableParameterSetPointer parametersToEstimate,
        const std::vector<IntegratorSettingsPointer> integratorSettings,
        const PropagatorSettingsPointer propagatorSettings,
        const int integratorIndexOffset = 0 ) {
        auto independentIntegratorSettingsList = utilities::cloneDuplicatePointers( integratorSettings );

        auto multiArcPropagatorSettings = std::dynamic_pointer_cast<
            propagators::MultiArcPropagatorSettings<ObservationScalarType, TimeType>>( propagatorSettings );
        if ( multiArcPropagatorSettings != nullptr ) {
            std::vector<double>
                arcStartTimes = estimatable_parameters::getMultiArcStateEstimationArcStartTimes( parametersToEstimate,
                                                                                                 ( integratorIndexOffset == 0 ) );
            if ( multiArcPropagatorSettings->getSingleArcSettings().size() != arcStartTimes.size() ) {
                throw std::runtime_error(
                    "Error when processing deprecated integrator/propagator settings in "
                    "estimation; inconsistent number of arcs" );
            }
            for( unsigned int i = 0; i < arcStartTimes.size( ); i++ )
            {
                multiArcPropagatorSettings->getSingleArcSettings( ).at( i )->resetInitialTime( arcStartTimes.at( i ) );
            }
        }
        else if( std::dynamic_pointer_cast< propagators::HybridArcPropagatorSettings< ObservationScalarType, TimeType > >( propagatorSettings ) != nullptr )
        {
            independentIntegratorSettingsList = preprocessDeprecatedIntegratorSettings(
                        parametersToEstimate, integratorSettings,
                        std::dynamic_pointer_cast< propagators::HybridArcPropagatorSettings< ObservationScalarType, TimeType > >( propagatorSettings )->getMultiArcPropagatorSettings( ),
                        1 );
        }
        return independentIntegratorSettingsList;
    }

    //! Constructor with multiple integrator settings
    /*!
     *  Constructs an OrbitDeterminationManager with multiple integrator
     * settings.
     *  \param bodies System of bodies.
     *  \param parametersToEstimate Pointer to the set of parameters to estimate.
     *  \param observationSettingsList List of observation model settings.
     *  \param integratorSettings Vector of integrator settings.
     *  \param propagatorSettings Pointer to the propagator settings.
     *  \param propagateOnCreation Flag to indicate whether to propagate on creation (default is true).
     */
    OrbitDeterminationManager( const SystemOfBodies &bodies,
                               const EstimatableParameterSetPointer parametersToEstimate,
                               const std::vector<ObservationModelSettingsPointer> &observationSettingsList,
                               const std::vector<IntegratorSettingsPointer> integratorSettings,
                               const PropagatorSettingsPointer propagatorSettings,
                               const bool propagateOnCreation = true )
        : parametersToEstimate_( parametersToEstimate ) {
        auto processedIntegratorSettings = preprocessDeprecatedIntegratorSettings(
            parametersToEstimate,
            integratorSettings,
            propagatorSettings );

        initializeOrbitDeterminationManager(
            bodies,
            observationSettingsList,
            propagators::validateDeprecatePropagatorSettings(
                processedIntegratorSettings, propagatorSettings ),
            propagateOnCreation );
    }

    //! Constructor with single integrator setting
    /*!
     *  Constructs an OrbitDeterminationManager with a single integrator
     * setting.
     *  \param bodies System of bodies.
     *  \param parametersToEstimate Pointer to the set of parameters to estimate.
     *  \param observationSettingsList List of observation model settings.
     *  \param integratorSettings Pointer to the integrator settings.
     *  \param propagatorSettings Pointer to the propagator settings.
     *  \param propagateOnCreation Flag to indicate whether to propagate on creation (default is true).
     */
    OrbitDeterminationManager( const SystemOfBodies &bodies,
                               const EstimatableParameterSetPointer parametersToEstimate,
                               const std::vector<ObservationModelSettingsPointer> &observationSettingsList,
                               const IntegratorSettingsPointer integratorSettings,
                               const PropagatorSettingsPointer propagatorSettings,
                               const bool propagateOnCreation = true )
        : parametersToEstimate_( parametersToEstimate ), bodies_( bodies ) {
        std::vector<IntegratorSettingsPointer> processedIntegratorSettings;
        if ( std::dynamic_pointer_cast<propagators::SingleArcPropagatorSettings<ObservationScalarType, TimeType>>( propagatorSettings ) != nullptr ) {
            processedIntegratorSettings = { integratorSettings };
        } else if ( std::dynamic_pointer_cast<propagators::MultiArcPropagatorSettings<ObservationScalarType, TimeType>>( propagatorSettings ) != nullptr ) {
            int numberOfArcs = estimatable_parameters::getMultiArcStateEstimationArcStartTimes( parametersToEstimate, true ).size();
            auto unprocessedIntegratorSettings = std::vector<IntegratorSettingsPointer>(
                numberOfArcs,
                integratorSettings );
            processedIntegratorSettings = preprocessDeprecatedIntegratorSettings(
                parametersToEstimate,
                unprocessedIntegratorSettings,
                propagatorSettings );
        } else if ( std::dynamic_pointer_cast<propagators::HybridArcPropagatorSettings<ObservationScalarType, TimeType>>( propagatorSettings ) != nullptr ) {
            int numberOfArcs = estimatable_parameters::getMultiArcStateEstimationArcStartTimes( parametersToEstimate, false ).size();
            std::vector<IntegratorSettingsPointer> unprocessedIntegratorSettings = std::vector<IntegratorSettingsPointer>( numberOfArcs + 1, integratorSettings );
            processedIntegratorSettings = preprocessDeprecatedIntegratorSettings(
                parametersToEstimate,
                unprocessedIntegratorSettings,
                propagatorSettings );
        }

        initializeOrbitDeterminationManager(
                    bodies, observationSettingsList, propagators::validateDeprecatePropagatorSettings(
                        processedIntegratorSettings, propagatorSettings ),
                    propagateOnCreation );
    }

    //! Constructor without integrator settings
    /*!
     *  Constructs an OrbitDeterminationManager without integrator settings.
     *  \param bodies System of bodies.
     *  \param parametersToEstimate Pointer to the set of parameters to estimate.
     *  \param observationSettingsList List of observation model settings.
     *  \param propagatorSettings Pointer to the propagator settings.
     *  \param propagateOnCreation Flag to indicate whether to propagate on creation (default is true).
     */
    OrbitDeterminationManager(
            const SystemOfBodies &bodies,
            const EstimatableParameterSetPointer parametersToEstimate,
            const std::vector<ObservationModelSettingsPointer >& observationSettingsList,
            const PropagatorSettingsPointer propagatorSettings,
            const bool propagateOnCreation = true ): 
            parametersToEstimate_( parametersToEstimate ), 
            bodies_( bodies ) 
    {
        initializeOrbitDeterminationManager( bodies,
                                             observationSettingsList,
                                             propagatorSettings,
                                             propagateOnCreation );
    }

    //! Get the set of parameters to estimate
    /*!
     *  \return Pointer to the set of parameters to estimate.
     */
    EstimatableParameterSetPointer getParametersToEstimate( )
    {
        return parametersToEstimate_;
    }

    //! Get the system of bodies
    /*!
     *  \return System of bodies.
     */
    SystemOfBodies getBodies( )
    {
        return bodies_;
    }

    //! Retrieve map of all observation managers
    /*!
     *  \return Map of observation managers for each observable type.
     */
    std::map<observation_models::ObservableType, std::shared_ptr<ObservationManagerBaseType>> getObservationManagers() const {
        return observationManagers_;
    }

    //! Retrieve map of all observation simulators
    /*!
     *  \return Vector of observation simulators for each observable type.
     */
    std::vector<ObservationSimulatorBasePointer> getObservationSimulators() const {
        std::vector<ObservationSimulatorBasePointer> observationSimulators;

        using ObservationManagerBasePointer = std::shared_ptr<observation_models::ObservationManagerBase<ObservationScalarType, TimeType>>;

        for ( const auto &managerItem : observationManagers_ ) {
            observation_models::ObservableType observableType = managerItem.first;
            ObservationManagerBasePointer observationManagerBase = managerItem.second;

            ObservationSimulatorBasePointer observationSimulator = observationManagerBase->getObservationSimulator();
            observationSimulators.push_back( observationSimulator );
        }

        return observationSimulators;
    }

    /*!
     *  Calculates the observation partials matrix and residuals based on the
     * state transition matrix,
     *  sensitivity matrix, and body states resulting from the previous numerical integration iteration.
     *  Partials and observations are calculated by the observationManagers_.
     *  \param observationsCollection Observable values and associated time tags, per observable type and set of link ends.
     *  \param parameterVectorSize Length of the vector of estimated parameters.
     *  \param totalObservationSize Total number of observations in observationsAndTimes map.
     *  \param designMatrix Matrix of observation partials w.r.t. parameter vector (return by reference).
     *  \param residuals Residuals of computed w.r.t. input observable values (return by reference).
     *  \param calculateResiduals Flag to indicate whether to calculate residuals (default is true).
     */
    void calculateDesignMatrixAndResiduals(
        const ObservationCollectionPointer observationsCollection,
        const int parameterVectorSize,
        const int totalObservationSize,
        Eigen::MatrixXd &designMatrix,
        Eigen::VectorXd &residuals,
        const bool calculateResiduals = true ) {
        // Initialize return data.
        initializeReturnData( totalObservationSize,
                              parameterVectorSize,
                              designMatrix,
                              residuals );

        // Iterate over all observable types in observationsAndTimes
        for ( auto observablesIterator : observationsCollection->getObservations() ) {
            observation_models::ObservableType currentObservableType = observablesIterator.first;

            // Iterate over all link ends for current observable type in observationsAndTimes
            for ( auto dataIterator : observablesIterator.second ) {
                // observation_models::LinkEnds currentLinkEnds = dataIterator.first;
                processLinkEnds( dataIterator,
                                 dataIterator.first, // currentLinkEnds
                                 currentObservableType,
                                 observationsCollection,
                                 designMatrix,
                                 residuals,
                                 calculateResiduals,
                                 parameterVectorSize );
            }

            if ( calculateResiduals ) {
                handleResidualDiscontinuities(
                    currentObservableType, observationsCollection, residuals );
            }
        }
    }

    //! Initialize return data
    /*!
     *  Initializes the design matrix and residuals with appropriate sizes.
     *  \param totalObservationSize Total number of observations.
     *  \param parameterVectorSize Length of the vector of estimated parameters.
     *  \param designMatrix Matrix of observation partials w.r.t. parameter vector (return by reference).
     *  \param residuals Residuals of computed w.r.t. input observable values (return by reference).
     */
    void initializeReturnData( const int totalObservationSize,
                               const int parameterVectorSize,
                               Eigen::MatrixXd &designMatrix,
                               Eigen::VectorXd &residuals ) {
        designMatrix = Eigen::MatrixXd::Zero( totalObservationSize, parameterVectorSize );
        residuals = Eigen::VectorXd::Zero( totalObservationSize );
    }

    //! Process link ends
    /*!
     *  Processes link ends for the given observable type and updates the design matrix and residuals.
     *  \param dataIterator Data iterator for the current observable type.
     *  \param currentLinkEnds Current link ends.
     *  \param currentObservableType Current observable type.
     *  \param observationsCollection Pointer to the observation collection.
     *  \param designMatrix Matrix of observation partials w.r.t. parameter vector (return by reference).
     *  \param residuals Residuals of computed w.r.t. input observable values (return by reference).
     *  \param calculateResiduals Flag to indicate whether to calculate residuals.
     *  \param parameterVectorSize Length of the vector of estimated parameters.
     */
    void processLinkEnds(
        const DataIteratorType &dataIterator,
        const observation_models::LinkEnds &currentLinkEnds,
        const observation_models::ObservableType &currentObservableType,
        const ObservationCollectionPointer observationsCollection,
        Eigen::MatrixXd &designMatrix,
        Eigen::VectorXd &residuals,
        const bool calculateResiduals,
        const int parameterVectorSize ) {
        for ( unsigned int i = 0; i < dataIterator.second.size(); i++ ) {
            std::shared_ptr<
                observation_models::SingleObservationSet<ObservationScalarType, TimeType>>
                currentObservations = dataIterator.second.at( i );
            std::pair<int, int> observationIndices =
                observationsCollection->getObservationSetStartAndSize()
                    .at( currentObservableType )
                    .at( currentLinkEnds )
                    .at( i );

            // Compute estimated ranges and range partials from current
            // parameter estimate.
            std::pair<ObservationVectorType, Eigen::MatrixXd>
                observationsWithPartials;
            observationsWithPartials =
                observationManagers_[currentObservableType]
                    ->computeObservationsWithPartials(
                        currentObservations->getObservationTimes(),
                        currentLinkEnds,
                        currentObservations->getReferenceLinkEnd(),
                        currentObservations->getAncilliarySettings() );

            // Compute residuals for current link ends and observabel type.
            if ( calculateResiduals ) {
                residuals.segment( observationIndices.first,
                                   observationIndices.second ) =
                    ( currentObservations->getObservationsVector() -
                      observationsWithPartials.first )
                        .template cast<double>();
            }

            // Set current observation partials in matrix of all partials
            designMatrix.block( observationIndices.first,
                                0,
                                observationIndices.second,
                                parameterVectorSize ) =
                observationsWithPartials.second;
        }
    }

    //! Handle residual discontinuities
    /*!
     *  Handles residual discontinuities for the given observable type.
     *  \param currentObservableType Current observable type.
     *  \param observationsCollection Pointer to the observation collection.
     *  \param residuals Residuals of computed w.r.t. input observable values (return by reference).
     */
    void handleResidualDiscontinuities(
        const observation_models::ObservableType &currentObservableType,
        const ObservationCollectionPointer &observationsCollection,
        Eigen::VectorXd &residuals ) {
        std::pair<int, int> observableStartAndSize =
            observationsCollection->getObservationTypeStartAndSize().at(
                currentObservableType );

        observation_models::checkObservationResidualDiscontinuities(
            residuals.block( observableStartAndSize.first,
                             0,
                             observableStartAndSize.second,
                             1 ),
            currentObservableType );
    }

    //! Calculate design matrix
    /*!
     *  Calculates the design matrix based on the observation collection.
     *  \param observationsCollection Pointer to the observation collection.
     *  \param parameterVectorSize Length of the vector of estimated parameters.
     *  \param totalObservationSize Total number of observations.
     *  \param designMatrix Matrix of observation partials w.r.t. parameter vector (return by reference).
     */
    void calculateDesignMatrix(
            const std::shared_ptr<ObservationCollectionType> observationsCollection,
            const int parameterVectorSize, 
            const int totalObservationSize,
            Eigen::MatrixXd& designMatrix )
    {
        Eigen::VectorXd dummyVector;
        calculateDesignMatrixAndResiduals(
                    observationsCollection, parameterVectorSize, totalObservationSize, designMatrix, dummyVector, false );

    }

    //! Normalize a priori covariance
    /*!
     *  Normalizes the a priori covariance matrix.
     *  \param inverseAPrioriCovariance Inverse of the a priori covariance matrix.
     *  \param normalizationValues Vector of normalization values.
     *  \return Normalized a priori covariance matrix.
     */
    Eigen::MatrixXd normalizeAprioriCovariance(
            const Eigen::MatrixXd& inverseAPrioriCovariance,
            const Eigen::VectorXd& normalizationValues )
    {
        int numberOfEstimatedParameters = inverseAPrioriCovariance.rows( );
        Eigen::MatrixXd normalizedInverseAprioriCovarianceMatrix = Eigen::MatrixXd::Zero(
                    numberOfEstimatedParameters, numberOfEstimatedParameters );

        for( int j = 0; j < numberOfEstimatedParameters; j++ )
        {
            for( int k = 0; k < numberOfEstimatedParameters; k++ )
            {
                normalizedInverseAprioriCovarianceMatrix( j, k ) = inverseAPrioriCovariance( j, k ) /
                        ( normalizationValues( j ) * normalizationValues( k ) );
            }
        }
        return normalizedInverseAprioriCovarianceMatrix;
    }

    //! Normalize design matrix
    /*!
     * Function to normalize the matrix of partial derivatives so that each column is in the range [-1,1]
     * \param observationMatrix Matrix of partial derivatives. Matrix modified by this function, and normalized matrix is
     * returned by reference
     * \return Vector with scaling values used for normalization
     */
    Eigen::VectorXd normalizeDesignMatrix( Eigen::MatrixXd& observationMatrix )
    {
        Eigen::VectorXd normalizationTerms = Eigen::VectorXd( observationMatrix.cols( ) );

        for( int i = 0; i < observationMatrix.cols( ); i++ )
        {
            Eigen::VectorXd currentVector = observationMatrix.block( 0, i, observationMatrix.rows( ), 1 );
            double minimum = currentVector.minCoeff( );
            double maximum = currentVector.maxCoeff( );
            if( std::fabs( minimum ) > maximum )
            {
                normalizationTerms( i ) = minimum;
            }
            else
            {
                normalizationTerms( i ) = maximum;
            }
            if( normalizationTerms( i ) == 0.0 )
            {
                normalizationTerms( i ) = 1.0;
            }
            currentVector = currentVector / normalizationTerms( i );

            observationMatrix.block( 0, i, observationMatrix.rows( ), 1 ) = currentVector;
        }

        return normalizationTerms;
    }

    //! Compute covariance
    /*!
     *  Computes the covariance based on the estimation input.
     *  \param estimationInput Pointer to the covariance analysis input.
     *  \return Pointer to the covariance analysis output.
     */
    std::shared_ptr< CovarianceAnalysisOutput< ObservationScalarType, TimeType > > computeCovariance(
            const std::shared_ptr< CovarianceAnalysisInput< ObservationScalarType, TimeType > > estimationInput )
    {
        // Get size of parameter vector and number of observations (total and per type)
        int numberOfEstimatedParameters = parametersToEstimate_->getParameterSetSize( );
        int totalNumberOfObservations = estimationInput->getObservationCollection( )->getTotalObservableSize( );

        bool exceptionDuringPropagation = false;

        try
        {
            if( estimationInput->getReintegrateEquationsOnFirstIteration( ) )
            {
                resetParameterEstimate(
                            parametersToEstimate_->template getFullParameterValues< ObservationScalarType >( ),
                            estimationInput->getReintegrateVariationalEquations( ) );
            }
        }
        catch( std::runtime_error& error )
        {
            std::cerr<<"Error when resetting parameters during covariance calculation: "<<std::endl<<
                       error.what( )<<std::endl<<"Terminating calculation"<<std::endl;
            exceptionDuringPropagation = true;
        }

        if( estimationInput->getPrintOutput( ) )
        {
            std::cout << "Calculating residuals and partials " << totalNumberOfObservations << std::endl;
        }

        // Calculate residuals and observation matrix for current parameter estimate.
        Eigen::MatrixXd designMatrix;
        calculateDesignMatrix(
                    estimationInput->getObservationCollection( ),
                    numberOfEstimatedParameters, totalNumberOfObservations, designMatrix );

        Eigen::VectorXd normalizationTerms = normalizeDesignMatrix( designMatrix );
        Eigen::MatrixXd normalizedInverseAprioriCovarianceMatrix = normalizeAprioriCovariance(
                estimationInput->getInverseOfAprioriCovariance( numberOfEstimatedParameters ), normalizationTerms );

        Eigen::MatrixXd constraintStateMultiplier;
        Eigen::VectorXd constraintRightHandSide;
        parametersToEstimate_->getConstraints( constraintStateMultiplier, constraintRightHandSide );

        Eigen::MatrixXd inverseNormalizedCovariance = linear_algebra::calculateInverseOfUpdatedCovarianceMatrix(
                               designMatrix.block( 0, 0, designMatrix.rows( ), numberOfEstimatedParameters ),
                               estimationInput->getWeightsMatrixDiagonals( ),
                               normalizedInverseAprioriCovarianceMatrix, constraintStateMultiplier, constraintRightHandSide );

        std::shared_ptr< CovarianceAnalysisOutput< ObservationScalarType, TimeType > > estimationOutput =
                std::make_shared< CovarianceAnalysisOutput< ObservationScalarType, TimeType > >(
                     designMatrix, estimationInput->getWeightsMatrixDiagonals( ), normalizationTerms,
                    inverseNormalizedCovariance, exceptionDuringPropagation );

        return estimationOutput;
    }

    //! Estimate parameters
    /*!
     *  Estimates the parameters based on the estimation input.
     *  \param estimationInput Pointer to the estimation input.
     *  \return Pointer to the estimation output.
     */
    std::shared_ptr< EstimationOutput< ObservationScalarType, TimeType > > estimateParameters(
            const std::shared_ptr< EstimationInput< ObservationScalarType, TimeType > > estimationInput )

    {
        currentParameterEstimate_ = parametersToEstimate_->template getFullParameterValues< ObservationScalarType >( );
//        std::cout << "current parameter estimate: " << currentParameterEstimate_.transpose( ) << "\n\n";

        // Get size of parameter vector and number of observations (total and per type)
        int parameterVectorSize = currentParameterEstimate_.size( );
        int totalNumberOfObservations = estimationInput->getObservationCollection( )->getTotalObservableSize( );

        // Declare variables to be returned (i.e. results from best iteration)
        double bestResidual = TUDAT_NAN;
        ParameterVectorType bestParameterEstimate = ParameterVectorType::Constant( parameterVectorSize, TUDAT_NAN );
        Eigen::VectorXd bestTransformationData = Eigen::VectorXd::Constant( parameterVectorSize, TUDAT_NAN );
        Eigen::VectorXd bestResiduals = Eigen::VectorXd::Constant( totalNumberOfObservations, TUDAT_NAN );
        Eigen::MatrixXd bestDesignMatrix = Eigen::MatrixXd::Constant( totalNumberOfObservations, parameterVectorSize, TUDAT_NAN );
        Eigen::VectorXd bestWeightsMatrixDiagonal = Eigen::VectorXd::Constant( totalNumberOfObservations, TUDAT_NAN );
        Eigen::MatrixXd bestInverseNormalizedCovarianceMatrix = Eigen::MatrixXd::Constant( parameterVectorSize, parameterVectorSize, TUDAT_NAN );

        std::vector< Eigen::VectorXd > residualHistory;
        std::vector< ParameterVectorType > parameterHistory;
        std::vector< std::shared_ptr< propagators::SimulationResults< ObservationScalarType, TimeType > > > simulationResultsPerIteration;

        // Declare residual bookkeeping variables
        std::vector< double > rmsResidualHistory;
        double residualRms;

        // Declare variables to be used in loop.

        // Set current parameter estimate as both previous and current estimate
        ParameterVectorType newParameterEstimate = currentParameterEstimate_;
        ParameterVectorType oldParameterEstimate = currentParameterEstimate_;

        int numberOfEstimatedParameters = parameterVectorSize;

        bool exceptionDuringPropagation = false, exceptionDuringInversion = false;
        // Iterate until convergence (at least once)
        int bestIteration = -1;
        int numberOfIterations = 0;
        do
        {
            // Re-integrate equations of motion and variational equations with new parameter estimate.
            try
            {
                if( ( numberOfIterations > 0 ) || ( estimationInput->getReintegrateEquationsOnFirstIteration( ) ) )
                {
                    resetParameterEstimate( newParameterEstimate, estimationInput->getReintegrateVariationalEquations( ) );
                }

                if( estimationInput->getSaveStateHistoryForEachIteration( ) )
                {
                    simulationResultsPerIteration.push_back( variationalEquationsSolver_->getVariationalPropagationResults( ) );
                }
            }
            catch( std::runtime_error& error )
            {
                std::cerr<<"Error when resetting parameters during parameter estimation: "<<std::endl<<
                           error.what( )<<std::endl<<"Terminating estimation"<<std::endl;
                exceptionDuringPropagation = true;
                break;
            }

            oldParameterEstimate = newParameterEstimate;

            if( estimationInput->getPrintOutput( ) )
            {
                std::cout << "Calculating residuals and partials " << totalNumberOfObservations << std::endl;
            }

            // Calculate residuals and observation matrix for current parameter estimate.
            Eigen::VectorXd residuals;
            Eigen::MatrixXd designMatrix;
            calculateDesignMatrixAndResiduals(
                        estimationInput->getObservationCollection( ),
                        parameterVectorSize,
                        totalNumberOfObservations,
                        designMatrix,
                        residuals,
                        true );

            Eigen::VectorXd normalizationTerms = normalizeDesignMatrix( designMatrix );
            Eigen::MatrixXd normalizedInverseAprioriCovarianceMatrix = normalizeAprioriCovariance(
                    estimationInput->getInverseOfAprioriCovariance( parameterVectorSize ), normalizationTerms );

            // Perform least squares calculation for correction to parameter vector.
            std::pair< Eigen::VectorXd, Eigen::MatrixXd > leastSquaresOutput;
            try
            {
                Eigen::MatrixXd constraintStateMultiplier;
                Eigen::VectorXd constraintRightHandSide;
                parametersToEstimate_->getConstraints( constraintStateMultiplier, constraintRightHandSide );
//                std::cout << "before least-squares adjustment" << "\n\n";
                leastSquaresOutput =
                        std::move( linear_algebra::performLeastSquaresAdjustmentFromDesignMatrix(
                                       designMatrix.block( 0, 0, designMatrix.rows( ), numberOfEstimatedParameters ),
                                       residuals, estimationInput->getWeightsMatrixDiagonals( ),
                                       normalizedInverseAprioriCovarianceMatrix, 1, 1.0E8, constraintStateMultiplier, constraintRightHandSide ) );

                if( constraintStateMultiplier.rows( ) > 0 )
                {
                    leastSquaresOutput.first.conservativeResize( parameterVectorSize );
                }
            }
            catch( std::runtime_error& error )
            {
                std::cerr<<"Error when solving normal equations during parameter estimation: "<<std::endl<<error.what( )<<
                           std::endl<<"Terminating estimation"<<std::endl;
                exceptionDuringInversion = true;
                break;
            }

            ParameterVectorType parameterAddition =
                    ( leastSquaresOutput.first.cwiseQuotient( normalizationTerms.segment( 0, numberOfEstimatedParameters ) ) ).
                    template cast< ObservationScalarType >( );

            // Update value of parameter vector
            newParameterEstimate = oldParameterEstimate + parameterAddition;
            parametersToEstimate_->template resetParameterValues< ObservationScalarType >( newParameterEstimate );
            newParameterEstimate = parametersToEstimate_->template getFullParameterValues< ObservationScalarType >( );

            if( estimationInput->getSaveResidualsAndParametersFromEachIteration( ) )
            {
                residualHistory.push_back( residuals );
                if( numberOfIterations == 0 )
                {
                    parameterHistory.push_back( oldParameterEstimate );
                }
                parameterHistory.push_back( newParameterEstimate );
            }

            oldParameterEstimate = newParameterEstimate;

            if( estimationInput->getPrintOutput( ) )
            {
                std::cout << "Parameter update" << parameterAddition.transpose( ) << std::endl;
            }

            // Calculate mean residual for current iteration.
            residualRms = linear_algebra::getVectorEntryRootMeanSquare( residuals );

            rmsResidualHistory.push_back( residualRms );
            if( estimationInput->getPrintOutput( ) )
            {
                std::cout << "Current residual: " << residualRms << std::endl;
            }

            // If current iteration is better than previous one, update 'best' data.
            if( residualRms < bestResidual || !( bestResidual == bestResidual ) )
            {
                bestResidual = residualRms;
                bestParameterEstimate = std::move( oldParameterEstimate );
                bestResiduals = std::move( residuals );
                if( estimationInput->getSaveDesignMatrix( ) )
                {
                    bestDesignMatrix = std::move( designMatrix );
                }
                bestWeightsMatrixDiagonal = std::move( estimationInput->getWeightsMatrixDiagonals( ) );
                bestTransformationData = std::move( normalizationTerms );
                bestInverseNormalizedCovarianceMatrix = std::move( leastSquaresOutput.second );
                bestIteration = numberOfIterations;
            }

            // Increment number of iterations
            numberOfIterations++;

            // Check for convergence
        } while( estimationInput->getConvergenceChecker( )->isEstimationConverged( numberOfIterations, rmsResidualHistory ) == false );

        if( estimationInput->getPrintOutput( ) )
        {
            std::cout << "Final residual: " << bestResidual << std::endl;
        }

        std::shared_ptr< EstimationOutput< ObservationScalarType, TimeType > > estimationOutput =
                std::make_shared< EstimationOutput< ObservationScalarType, TimeType > >(
                    bestParameterEstimate, bestResiduals, bestDesignMatrix, bestWeightsMatrixDiagonal, bestTransformationData,
                    bestInverseNormalizedCovarianceMatrix, bestResidual, bestIteration,
                    residualHistory, parameterHistory, exceptionDuringInversion,
                    exceptionDuringPropagation );

        if ( estimationInput->getSaveStateHistoryForEachIteration() ) {
            estimationOutput->setSimulationResults( simulationResultsPerIteration );
        }

        return estimationOutput;
    }

    //! Reset the current parameter estimate
    /*!
     *  Resets the current parameter estimate and reintegrates the variational equations and equations of motion with the new estimate.
     *  \param newParameterEstimate New estimate of parameter vector.
     *  \param reintegrateVariationalEquations Boolean denoting whether the variational equations are to be reintegrated.
     */
    void resetParameterEstimate( const ParameterVectorType& newParameterEstimate, const bool reintegrateVariationalEquations = 1 )
    {
        if( integrateAndEstimateOrbit_ )
        {
            variationalEquationsSolver_->resetParameterEstimate( newParameterEstimate, reintegrateVariationalEquations );
        }
        else
        {
            parametersToEstimate_->template resetParameterValues< ObservationScalarType>( newParameterEstimate );
        }
        currentParameterEstimate_ = newParameterEstimate;
    }

    //! Get the variational equations solver
    /*!
     *  Retrieves the object to numerically integrate and update the variational equations and equations of motion.
     *  \return Object to numerically integrate and update the variational equations and equations of motion.
     */
    std::shared_ptr< propagators::VariationalEquationsSolver< ObservationScalarType, TimeType > >
    getVariationalEquationsSolver( ) const
    {
        return variationalEquationsSolver_;
    }

    //! Get the observation manager
    /*!
     *  Retrieves the observation manager for a single observable type. The
     * observation manager can simulate observations and calculate observation
     * partials for all link ends involved in the given observable type.
     *  \param observableType Type of observable for which manager is to be retrieved.
     *  \return Observation manager for given observable type.
     */
    std::shared_ptr< ObservationManagerBaseType > getObservationManager(
            const observation_models::ObservableType observableType ) const
    {
        // Check if manager exists for requested observable type.
        if( observationManagers_.count( observableType ) == 0 )
        {
            throw std::runtime_error(
                        "Error when retrieving observation manager of type " + std::to_string(
                            observableType ) + ", manager not found" );
        }

        return observationManagers_.at( observableType );
    }

    //! Get the current parameter estimate
    /*!
     *  Retrieves the current parameter estimate.
     *  \return Current parameter estimate.
     */
    ParameterVectorType getCurrentParameterEstimate( )
    {
        return currentParameterEstimate_;
    }

    //! Get the state transition and sensitivity matrix interface
    /*!
     *  Retrieves the object used to propagate/process the numerical solution of
     * the variational equations/dynamics.
     *  \return Object used to propagate/process the numerical solution of the variational equations/dynamics.
     */
    std::shared_ptr< propagators::CombinedStateTransitionAndSensitivityMatrixInterface >
    getStateTransitionAndSensitivityMatrixInterface( )
    {
        return stateTransitionAndSensitivityMatrixInterface_;
    }

  protected:
    //! Initialize OrbitDeterminationManager
    /*!
     *  Initializes the OrbitDeterminationManager with the provided settings.
     *  \param bodies Map of body objects with names of bodies, storing all environment models used in simulation.
     *  \param observationSettingsList Sets of observation model settings per link ends (i.e. transmitter, receiver, etc.)
     *  for which measurement data is to be provided in orbit determination
     * process (through estimateParameters function)
     *  \param propagatorSettings Settings for propagator.
     *  \param propagateOnCreation Boolean denoting whether initial propagatoon is to be performed upon object creation (default
     *  true)
     */
    void initializeOrbitDeterminationManager(
            const SystemOfBodies &bodies,
            const std::vector<ObservationModelSettingsPointer >& observationSettingsList,
            const std::shared_ptr< propagators::PropagatorSettings< ObservationScalarType > > propagatorSettings,
            const bool propagateOnCreation = true )
    {
        propagators::toggleIntegratedResultSettings< ObservationScalarType, TimeType >( propagatorSettings );
        using namespace numerical_integrators;
        using namespace orbit_determination;
        using namespace observation_models;

        // Check if any dynamics is to be estimated
        std::map< propagators::IntegratedStateType, std::vector< std::pair< std::string, std::string > > >
                initialDynamicalStates =
                estimatable_parameters::getListOfInitialDynamicalStateParametersEstimate< ObservationScalarType >(
                    parametersToEstimate_ );
        if( initialDynamicalStates.size( ) > 0 )
        {
            integrateAndEstimateOrbit_ = true;
        }
        else
        {
            integrateAndEstimateOrbit_ = false;
        }

        propagatorSettings->getOutputSettingsBase( )->setCreateDependentVariablesInterface( true );
        if( integrateAndEstimateOrbit_ )
        {
            variationalEquationsSolver_ =
                    simulation_setup::createVariationalEquationsSolver< ObservationScalarType, TimeType >(
                        bodies, propagatorSettings, parametersToEstimate_, propagateOnCreation );
        }

        if( integrateAndEstimateOrbit_ )
        {
            stateTransitionAndSensitivityMatrixInterface_ = variationalEquationsSolver_->getStateTransitionMatrixInterface( );
        }
        else if( propagatorSettings == nullptr )
        {
            stateTransitionAndSensitivityMatrixInterface_ = createStateTransitionAndSensitivityMatrixInterface< ObservationScalarType, TimeType >(
                        propagatorSettings, parametersToEstimate_, 0, parametersToEstimate_->getParameterSetSize( ) );
        }
        else
        {
            throw std::runtime_error( "Error, cannot parse propagator settings without estimating dynamics in OrbitDeterminationManager" );
        }

        // TODO correct this when moving dependent variable interface into results object
        if( std::dynamic_pointer_cast< propagators::HybridArcVariationalEquationsSolver< ObservationScalarType, TimeType > >( variationalEquationsSolver_ ) == nullptr )
        {
            dependentVariablesInterface_ = variationalEquationsSolver_->getDynamicsSimulatorBase( )->getDependentVariablesInterface( );
        }

        // Iterate over all observables and create observation managers.
        std::map< ObservableType, std::vector< std::shared_ptr< ObservationModelSettings > > > sortedObservationSettingsList =
                sortObservationModelSettingsByType( observationSettingsList );
        for( auto it : sortedObservationSettingsList )
        {
            // Call createObservationSimulator of required observation size
            ObservableType observableType = it.first;

            // Create observation manager for current observable.
            observationManagers_[ observableType ] =
                    createObservationManagerBase< ObservationScalarType, TimeType >(
                        observableType,
                        it.second,
                        bodies, parametersToEstimate_,
                        stateTransitionAndSensitivityMatrixInterface_, dependentVariablesInterface_ );
        }

        // Set current parameter estimate from body initial states and parameter set.
        currentParameterEstimate_ = parametersToEstimate_->template getFullParameterValues< ObservationScalarType >( );

    }

    //! Boolean to denote whether any dynamical parameters are estimated
    bool integrateAndEstimateOrbit_;

    //! Object used to propagate/process the numerical solution of the variational equations/dynamics
    std::shared_ptr< propagators::VariationalEquationsSolver< ObservationScalarType, TimeType > >
    variationalEquationsSolver_;

    //! List of object that compute the values/partials of the observables
    std::map< observation_models::ObservableType,
    std::shared_ptr< observation_models::ObservationManagerBase< ObservationScalarType, TimeType > > > observationManagers_;

    //! Container object for all parameters that are to be estimated
    std::shared_ptr< estimatable_parameters::EstimatableParameterSet< ObservationScalarType > > parametersToEstimate_;

    SystemOfBodies bodies_;

    //! Current values of the vector of estimated parameters
    ParameterVectorType currentParameterEstimate_;

    //! Object used to interpolate the numerically integrated result of the state transition/sensitivity matrices.
    std::shared_ptr< propagators::CombinedStateTransitionAndSensitivityMatrixInterface >
    stateTransitionAndSensitivityMatrixInterface_;

    //! Object used to interpolate the numerically integrated result of the dependent variables.
    std::shared_ptr< propagators::DependentVariablesInterface< TimeType > > dependentVariablesInterface_;

};
} // namespace simulation_setup
} // namespace tudat
#endif // TUDAT_ORBITDETERMINATIONMANAGER_H
