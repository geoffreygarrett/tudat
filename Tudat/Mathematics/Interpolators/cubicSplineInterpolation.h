/*    Copyright (c) 2010-2012, Delft University of Technology
 *    All rights reserved.
 *
 *    Redistribution and use in source and binary forms, with or without modification, are
 *    permitted provided that the following conditions are met:
 *      - Redistributions of source code must retain the above copyright notice, this list of
 *        conditions and the following disclaimer.
 *      - Redistributions in binary form must reproduce the above copyright notice, this list of
 *        conditions and the following disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *      - Neither the name of the Delft University of Technology nor the names of its contributors
 *        may be used to endorse or promote products derived from this software without specific
 *        prior written permission.
 *
 *    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
 *    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *    COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 *    GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *    AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 *    OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *    Changelog
 *      YYMMDD    Author            Comment
 *      110620    F.M. Engelen      File created.
 *      110707    E.A.G. Heeren     Minor spelling/lay-out corrections.
 *      110714    E.A.G. Heeren     Minor spelling/lay-out corrections.
 *      110905    S. Billemont      Reorganized includes.
 *                                  Moved (con/de)structors and getter/setters to header.
 *
 *    References
 *      Press W.H., et al. Numerical Recipes in C++: The Art of
 *          Scientific Computing. Cambridge University Press, February 2002.
 *
 */

#ifndef TUDAT_CUBIC_SPLINE_INTERPOLATION_H
#define TUDAT_CUBIC_SPLINE_INTERPOLATION_H

#include <cmath>

#include <Eigen/Core>

namespace tudat
{
namespace mathematics
{
namespace interpolators
{

//! The cubic spline interpolation class.
/*!
 * The cubic spline interpolation class.
 */
class CubicSplineInterpolation
{
public:

    //! Default constructor.
    /*!
     * Default constructor.
     */
    CubicSplineInterpolation( ) : lowerEntry_( -0 ), numberOfDataPoints_( -0 ),
        targetIndependentVariableValue_( -0.0 ), coefficientb2_( -0.0 ), coefficientA_( -0.0 ),
        coefficientB_( -0.0 ), coefficientC_( -0.0 ), coefficientD_( -0.0 ) { }

    //! Initialize cubic spline interpolation.
    /*!
     * Initializes the cubic spline interpolation.
     * \param independentVariables Vector with the independent variables.
     * \param dependentVariables Vector with the dependent variables.
     */
    void initializeCubicSplineInterpolation( Eigen::VectorXd& independentVariables,
                                             Eigen::VectorXd& dependentVariables );

    //! Interpolate.
    /*!
     * Executes interpolation of data at a given target value of the independent variable, to
     * yield an interpolated value of the dependent variable.
     * \param targetIndependentVariableValue Target independent variable value at which point
     *          the interpolation is performed.
     * \return Interpolated dependent variable value.
     */
    double interpolate( double targetIndependentVariableValue );

protected:

private:

    //! The lower entry in the vector closest to the required point.
    /*!
     * The lower entry in the vector closest to the required point.
     */
    unsigned int lowerEntry_;

    //! The number of datapoints.
    /*!
     * The number of datapoints.
     */
    unsigned int numberOfDataPoints_;

    //! Target independent variable value.
    /*!
     * Target independent variable value.
     */
    double targetIndependentVariableValue_;

    //! The coefficient b2 ( math variable ).
    /*!
     * The coefficient b2 ( math variable ).
     */
    double coefficientb2_;

    //! The coefficient a ( math variable ).
    /*!
     * The coefficient a ( math variable ).
     */
    double coefficientA_;

    //! The coefficient b ( math variable ).
    /*!
     * The coefficient b ( math variable ).
     */
    double coefficientB_;

    //! The coefficient c( math variable ).
    /*!
     * The coefficient c( math variable ).
     */
    double coefficientC_;

    //! The coefficient d ( math variable ).
    /*!
     * The coefficient d ( math variable ).
     */
    double coefficientD_;

    //! Vector with dependent variables.
    /*!
     * Vector with dependent variables.
     */
    Eigen::VectorXd dependentVariables_;

    //! Vector with independent variables.
    /*!
     * Vector with independent variables.
     */
    Eigen::VectorXd independentVariables_;

    //! Vector filled with intermediate version of the second derivative of curvature.
    /*!
     *  Vector filled with intermediate version of the second derivative of curvature.
     */
    Eigen::VectorXd intermediateSecondDerivativeOfCurvature_;

    //! Vector filled with second derivative of curvature of each point.
    /*!
     *  Vector filled with second derivative of curvature of each point.
     */
    Eigen::VectorXd secondDerivativeOfCurvature_;

    //! Vector filled with the h coefficient ( math variable ).
    /*!
     *  Vector filled with the h coefficient ( math variable ).
     */
    Eigen::VectorXd hCoefficients_;

    //! Vector filled with the a coefficient ( math variable ).
    /*!
     *  Vector filled with the a coefficient ( math variable ).
     */
    Eigen::VectorXd aCoefficients_;

    //! Vector filled with the b coefficient ( math variable ).
    /*!
     *  Vector filled with the b coefficient ( math variable ).
     */
    Eigen::VectorXd bCoefficients_;

    //! Vector filled with the c coefficient ( math variable ).
    /*!
     *  Vector filled with the c coefficient ( math variable ).
     */
    Eigen::VectorXd cCoefficients_;

    //! Vector filled with the r coefficient ( math variable ).
    /*!
     *  Vector filled with the r coefficient ( math variable ).
     */
    Eigen::VectorXd rCoefficients_;

    //! Vector filled with the g coefficient ( math variable ).
    /*!
     *  Vector filled with the g coefficient ( math variable ).
     */
    Eigen::VectorXd gCoefficients_;

    //! Initialize tridiagonal matrix.
    /*!
     * Initializes the tridiagonal matrix.
     */
    void constructTridiagonalMatrix_( );

    //! Compute second derivative of curvature.
    /*!
     * Computes the second derivative of curvature.
     */
    void computeSecondDerivativeOfCurvature_( );
};

} // namespace interpolators
} // namespace mathematics
} // namespace tudat

#endif // TUDAT_CUBIC_SPLINE_INTERPOLATION_H