/*    Copyright (c) 2010-2017, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 *
 */

#include "variable.h"

#include "acceleration.h"
#include "torque.h"
#include "referenceFrames.h"

namespace tudat
{

namespace propagators
{

/// VariableSettings

//! Create a `json` object from a shared pointer to a `SingleDependentVariableSaveSettings` object.
void to_json( json& jsonObject, const boost::shared_ptr< VariableSettings >& variableSettings )
{
    if ( ! variableSettings )
    {
        return;
    }
    using namespace json_interface;
    using K = Keys::Variable;

    const VariableType variableType = variableSettings->variableType_;

    switch ( variableType )
    {
    case epochVariable:
    {
        jsonObject[ K::type ] = variableType;
        return;
    }
    case stateVariable:
    {
        boost::shared_ptr< BodyVariableSettings > bodyVariableSettings =
                boost::dynamic_pointer_cast< BodyVariableSettings >( variableSettings );
        enforceNonNullPointer( bodyVariableSettings );
        jsonObject[ K::type ] = variableType;
        jsonObject[ K::body ] = bodyVariableSettings->associatedBody_;
        return;
    }
    case dependentVariable:
    {
        jsonObject = boost::dynamic_pointer_cast< SingleDependentVariableSaveSettings >( variableSettings );
        return;
    }
    default:
        jsonObject = handleUnimplementedEnumValueToJson( variableType, variableTypes, unsupportedVariableTypes );
    }
}

//! Create a shared pointer to a `VariableSettings` object from a `json` object.
void from_json( const json& jsonObject, boost::shared_ptr< VariableSettings >& variableSettings )
{
    using namespace basic_astrodynamics;
    using namespace reference_frames;
    using namespace json_interface;
    using K = Keys::Variable;

    const VariableType variableType = getValue< VariableType >( jsonObject, K::type, dependentVariable );

    switch ( variableType )
    {
    case epochVariable:
    {
        variableSettings = boost::make_shared< VariableSettings >( variableType );
        return;
    }
    case stateVariable:
    {
        variableSettings = boost::make_shared< BodyVariableSettings >(
                    variableType, getValue< std::string >( jsonObject, K::body ) );
        return;
    }
    case dependentVariable:
    {
        variableSettings = getAs< boost::shared_ptr< SingleDependentVariableSaveSettings > >( jsonObject );
        return;
    }
    default:
        handleUnimplementedEnumValueFromJson( variableType, variableTypes, unsupportedVariableTypes );
    }
}


/// SingleDependentVariableSaveSettings

//! Create a `json` object from a shared pointer to a `SingleDependentVariableSaveSettings` object.
void to_json( json& jsonObject,
              const boost::shared_ptr< SingleDependentVariableSaveSettings >& dependentVariableSettings )
{
    if ( ! dependentVariableSettings )
    {
        return;
    }
    using namespace json_interface;
    using K = Keys::Variable;

    const PropagationDependentVariables dependentVariableType = dependentVariableSettings->dependentVariableType_;
    jsonObject[ K::dependentVariableType ] = dependentVariableType;
    jsonObject[ K::body ] = dependentVariableSettings->associatedBody_;

    switch ( dependentVariableType )
    {
    case single_acceleration_norm_dependent_variable:
    case single_acceleration_dependent_variable:
    {
        boost::shared_ptr< SingleAccelerationDependentVariableSaveSettings > accelerationVariableSettings =
                boost::dynamic_pointer_cast< SingleAccelerationDependentVariableSaveSettings >(
                    dependentVariableSettings );
        enforceNonNullPointer( accelerationVariableSettings );
        jsonObject[ K::accelerationType ] = accelerationVariableSettings->accelerationModeType_;
        jsonObject[ K::bodyExertingAcceleration ] = dependentVariableSettings->secondaryBody_;
        return;
    }
    case single_torque_norm_dependent_variable:
    case single_torque_dependent_variable:
    {
        boost::shared_ptr< SingleTorqueDependentVariableSaveSettings > torqueVariableSettings =
                boost::dynamic_pointer_cast< SingleTorqueDependentVariableSaveSettings >(
                    dependentVariableSettings );
        enforceNonNullPointer( torqueVariableSettings );
        jsonObject[ K::torqueType ] = torqueVariableSettings->torqueModeType_;
        jsonObject[ K::bodyExertingTorque ] = dependentVariableSettings->secondaryBody_;
        return;
    }
    case intermediate_aerodynamic_rotation_matrix_variable:
    {
        boost::shared_ptr< IntermediateAerodynamicRotationVariableSaveSettings > aerodynamicRotationVariableSettings =
                boost::dynamic_pointer_cast< IntermediateAerodynamicRotationVariableSaveSettings >(
                    dependentVariableSettings );
        enforceNonNullPointer( aerodynamicRotationVariableSettings );
        jsonObject[ K::baseFrame ] = aerodynamicRotationVariableSettings->baseFrame_;
        jsonObject[ K::targetFrame ] = aerodynamicRotationVariableSettings->targetFrame_;
        return;
    }
    case relative_body_aerodynamic_orientation_angle_variable:
    {
        boost::shared_ptr< BodyAerodynamicAngleVariableSaveSettings > aerodynamicAngleVariableSettings =
                boost::dynamic_pointer_cast< BodyAerodynamicAngleVariableSaveSettings >(
                    dependentVariableSettings );
        enforceNonNullPointer( aerodynamicAngleVariableSettings );
        jsonObject[ K::angle ] = aerodynamicAngleVariableSettings->angle_;
        return;
    }
    default:
    {
        assignIfNotEmpty( jsonObject, K::relativeToBody, dependentVariableSettings->secondaryBody_ );
        return;
    }
    }
}

//! Create a shared pointer to a `SingleDependentVariableSaveSettings` object from a `json` object.
void from_json( const json& jsonObject,
                boost::shared_ptr< SingleDependentVariableSaveSettings >& dependentVariableSettings )
{
    using namespace basic_astrodynamics;
    using namespace reference_frames;
    using namespace json_interface;
    using K = Keys::Variable;

    const PropagationDependentVariables dependentVariableType =
            getValue< PropagationDependentVariables >( jsonObject, K::dependentVariableType );
    const std::string bodyName = getValue< std::string>( jsonObject, K::body );

    switch ( dependentVariableType )
    {
    case single_acceleration_norm_dependent_variable:
    case single_acceleration_dependent_variable:
    {
        dependentVariableSettings = boost::make_shared< SingleAccelerationDependentVariableSaveSettings >(
                    getValue< AvailableAcceleration >( jsonObject, K::accelerationType ),
                    bodyName, getValue< std::string>( jsonObject, K::bodyExertingAcceleration ),
                    dependentVariableType == single_acceleration_norm_dependent_variable );
        return;
    }
    case single_torque_norm_dependent_variable:
    case single_torque_dependent_variable:
    {
        dependentVariableSettings = boost::make_shared< SingleTorqueDependentVariableSaveSettings >(
                    getValue< AvailableTorque >( jsonObject, K::torqueType ),
                    bodyName, getValue< std::string>( jsonObject, K::bodyExertingTorque ),
                    dependentVariableType == single_torque_norm_dependent_variable );
        return;
    }
    case intermediate_aerodynamic_rotation_matrix_variable:
    {
        dependentVariableSettings = boost::make_shared< IntermediateAerodynamicRotationVariableSaveSettings >(
                    bodyName, getValue< AerodynamicsReferenceFrames >( jsonObject, K::baseFrame ),
                    getValue< AerodynamicsReferenceFrames >( jsonObject, K::targetFrame ) );
        return;
    }
    case relative_body_aerodynamic_orientation_angle_variable:
    {
        dependentVariableSettings = boost::make_shared< BodyAerodynamicAngleVariableSaveSettings >(
                    bodyName, getValue< AerodynamicsReferenceFrameAngles >( jsonObject, K::angle ) );
        return;
    }
    default:
    {
        dependentVariableSettings = boost::make_shared< SingleDependentVariableSaveSettings >(
                    dependentVariableType, bodyName, getValue< std::string>( jsonObject, K::relativeToBody, "" ) );
        return;
    }
    }
}

} // namespace propagators


namespace json_interface
{

//! -DOC
boost::shared_ptr< propagators::VariableSettings > getVariable( const json& jsonObject, const KeyPath& keyPath )
{
    using namespace propagators;
    using K = Keys::Variable;

    json variable = getValue< json >( jsonObject, keyPath );
    if ( variable.is_string( ) )
    {
        const std::string variableName = variable.get< std::string >( );
        if ( variableName == variableTypes.at( epochVariable ) )
        {
            variable = json( );
            variable[ K::type ] = epochVariable;
        }
        else
        {
            const std::vector< std::string > bodyAndVariable = split( variableName, '.' );
            if ( bodyAndVariable.size( ) == 2 && bodyAndVariable.back( ) == variableTypes.at( stateVariable ) )
            {
                variable = json( );
                variable[ K::type ] = stateVariable;
                variable[ K::body ] = bodyAndVariable.front( );
            }
            else
            {
                try
                {
                    variable = getValue< json >( jsonObject, SpecialKeys::root / Keys::variables / variableName );
                }
                catch ( const UndefinedKeyError& error )
                {
                    error.rethrowIfNotTriggeredByMissingValueAt( variableName );
                    std::cerr << "Undefined variable: " << variableName << std::endl;
                    throw IllegalValueError< propagators::VariableSettings >(
                                keyPath.canonical( getKeyPath( jsonObject ) ), variable );
                }
            }
        }
    }
    return getAs< boost::shared_ptr< VariableSettings > >( variable );
}

//! -DOC
std::vector< boost::shared_ptr< propagators::VariableSettings > > getVariables(
        const json& jsonObject, const KeyPath& keyPath )
{
    using namespace propagators;
    const std::vector< json > jsonVector = getValue< std::vector< json > >( jsonObject, keyPath );
    std::vector< boost::shared_ptr< VariableSettings > > variables;
    for ( unsigned int i = 0; i < jsonVector.size( ); ++i )
    {
        variables.push_back( getVariable( jsonObject, keyPath / i ) );
    }
    return variables;
}

//! -DOC
std::vector< boost::shared_ptr< propagators::SingleDependentVariableSaveSettings > > getDependentVariables(
        const json& jsonObject, const KeyPath& keyPath )
{
    using namespace propagators;
    const std::vector< boost::shared_ptr< VariableSettings > > variables = getVariables( jsonObject, keyPath );
    std::vector< boost::shared_ptr< SingleDependentVariableSaveSettings > > dependentVariables;
    for ( const boost::shared_ptr< VariableSettings > variable : variables )
    {
        boost::shared_ptr< SingleDependentVariableSaveSettings > dependentVariable =
                boost::dynamic_pointer_cast< SingleDependentVariableSaveSettings > ( variable );
        if ( dependentVariable )
        {
            dependentVariables.push_back( dependentVariable );
        }
    }
    return dependentVariables;
}

} // json_interface

} // namespace tudat