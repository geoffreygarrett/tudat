/*    Copyright (c) 2010-2011 Delft University of Technology.
 *
 *    This software is protected by national and international copyright.
 *    Any unauthorized use, reproduction or modification is unlawful and
 *    will be prosecuted. Commercial and non-private application of the
 *    software in any form is strictly prohibited unless otherwise granted
 *    by the authors.
 *
 *    The code is provided without any warranty; without even the implied
 *    warranty of merchantibility or fitness for a particular purpose.
 *
 *    Changelog
 *      YYMMDD    Author            Comment
 *      111103    S. Billemont      First creation of code.
 *      120326    D. Dirkx          Code checked, minor layout changes.
 */

#ifndef TEXTPARSER_H
#define TEXTPARSER_H

#include <iostream>
#include "Tudat/InputOutput/parser.h"

namespace tudat
{
namespace input_output
{

//! A TextParser is a Parser that specialized in parsing text files.
/*!
 * A TextParser provides a set of functions to streamline parsing to simplify the parsing
 * process. The inheriting parsers can chose there prefered way of processing by calling 
 * TextParser(bool).
 *
 * NOTE: This TextParser works with the FieldValue/FieldType architecture.
 * For simpler file reading, use, for instance, matrixTextFileReader.
 */
class TextParser : public Parser {

public:

    //! Create the default TextParser with proccesAsStream == false.
    /*!
     * The default constructor for TextParser causes the parser to behave as a line based parser.
     */
    TextParser( ) : parsedData( parsed_data_vector_utilities::ParsedDataVectorPtr(
                               new parsed_data_vector_utilities::ParsedDataVector ) ),
                               parseAsStream( false ){ }

    //! Create the TextParser in the given process mode.
    /*!
     * Create a Textparser that deligates parsing either as stream or on a line-by-line basis.
     * \param processAsStream Boolean that determines whether the parsing should be done as a
     * stream (string is default).
     */
    TextParser( bool proccesAsStream ) : parsedData(
            parsed_data_vector_utilities::ParsedDataVectorPtr(
                new parsed_data_vector_utilities::ParsedDataVector ) ),
        parseAsStream( proccesAsStream ) {}

    //! Default destructor, no new objects besides smart ones.
    virtual ~TextParser( ) { }

    //! Implementation of parse for strings.
    /*!
     * \param string String that is to be parsed.
     * \see Parser::parse(std::string& string).
     */
    parsed_data_vector_utilities::ParsedDataVectorPtr parse( std::string& string );

    //! Implementation of parse for an istream.
    /*!
     * \param stream Stream that is to be parsed.
     * \see Parser::parse(std::istream& stream).
     */
    parsed_data_vector_utilities::ParsedDataVectorPtr parse( std::istream& stream );
	
protected:

    //! Data container of the parsed data.
    /*!
     * Cleared and refilled on every call to parse(string) or parse(stream).
     */
    parsed_data_vector_utilities::ParsedDataVectorPtr parsedData;

    //! Flag to identify either stream parsing or line based parsing.
    /*!
     * Clients doing stream based parsing must override parseStream!
     * Clients doing line   based parsing must override parseLine!
     * Clients doing both parsing techniques must override parseLine and parseStream.
     */
    bool parseAsStream;
	
    //! Parse the given line content and append the resulting data lines to parsedData.
    /*!
     * Parse all lines/fields from the passed string and store (append) them to parsedData.
     * 
     * Needs to be overwritten if parseAsStream==false otherwise an exception will be thrown!
     * 
     * \param line String to parse
     */
    virtual void parseLine( std::string& line )
    {
        boost::throw_exception( boost::enable_error_info( std::runtime_error
                                                         ( "Must be overriden to be used" ) ) );
    }

    //! Parse the given stream content and append the resulting data lines to parsedData.
    /*!
     * Parse all lines/fields from the passed stream and store (append) them to parsedData.
     *
     * Needs to be overwritten if parseAsStream==true otherwise an exception will be thrown!
     *
     * \param stream Stream to parse
     */
    virtual void parseStream( std::istream& stream )
    {
        boost::throw_exception( boost::enable_error_info( std::runtime_error
                                                         ( "Must be overriden to be used" ) ) );
    }

};

} // namespace input_output
} // namespace tudat

#endif
