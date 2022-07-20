/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2022 AUTHOR,AFFILIATION
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::cylindricalSinusoidalVelocityProfile2DFvPatchVectorField

Group
    grpGenericBoundaryConditions

Description
    This boundary condition provides a cylindricalSinusoidalVelocityProfile2D condition,
    calculated as:

        \f[
            U = U_{avg}*pi/2*cos(pi/w*theta)
        \f]

    where
    \vartable
        U      | sinusoidal velocity profile on a 2D cylindrical arc-segment patch [m/s]
        U_{avg}| average velocity (flowrate/arc-area) [m/s]
        w      | cylindrical arcarc-angle [deg]
        theta  | [-w/2, w/2] [deg]
    \endtable

Usage
    \table
        Property             | Description                  | Req'd | Default
        origin_               | single vector value         | yes |
        axis_                 | single vector value         | yes |
        theta0_               | single scalar value         | yes |
        w_                    | single scalar value         | yes |
        volumetricFlowRate_   | single scalar value         | yes |
        volumetric_           | single bool value           | no  | true
    \endtable

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            cylindricalSinusoidalProfile2D;
        origin          ( 0 0 0.005 ); // origin of the cylindrical-arc patch 
        axis            ( 1 0 0.005 ); 
        theta0          90; // smallest angle (clockwise or anticlockwise) between axis and patch centre
        w               10; // arc-angle of patch
        volumetricFlowRate constant 1e-05; 
        value           uniform ( 0 0 0 );
    }
    \endverbatim

SourceFiles
    cylindricalSinusoidalVelocityProfile2DFvPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef cylindricalSinusoidalVelocityProfile2DFvPatchVectorField_H
#define cylindricalSinusoidalVelocityProfile2DFvPatchVectorField_H

#include "fixedValueFvPatchFields.H"
#include "Function1.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
     Class cylindricalSinusoidalVelocityProfile2DFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class cylindricalSinusoidalVelocityProfile2DFvPatchVectorField
:
    public fixedValueFvPatchVectorField
{
    // Private Data

        //- Origin of the cylinder which includes arc-shaped patch
        const vector origin_;

        //- Reference axis from which the centre-angle of arc-shaped patch is measured 
        const vector axis_;
        
        //- Width-angle of arc-shaped patch in deg(=360 for full cylinder)
        const scalar w_;

        //- Centre-angle of arc-shaped patch in deg(the smallest angle measured from reference axis)
        const scalar theta0_;

        //- Patch integral flow rate
        autoPtr<Function1<scalar>> flowRate_;

        //- Name of the density field used to normalize the mass flux
        word rhoName_;

        //- Rho initialisation value (for start; if value not supplied)
        scalar rhoInlet_;

        //- Is volumetric?
        bool volumetric_;


    // Private Member Functions

        //- Return current time
        scalar t() const;

        //- Update the patch values given the appropriate density type and value
        template<class RhoType>
        void updateValues(const RhoType& rho);

        template<class RhoType>
        Foam::vectorField evaluateVelocityProfile(const RhoType& rho);

        template<class RhoType>
        void correctVelocityProfile(Foam::vectorField&, const RhoType& rho);

public:

    //- Runtime type information
    TypeName("cylindricalSinusoidalProfile2D");

    // Constructors

        //- Construct from patch and internal field
        cylindricalSinusoidalVelocityProfile2DFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        cylindricalSinusoidalVelocityProfile2DFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping onto a new patch
        cylindricalSinusoidalVelocityProfile2DFvPatchVectorField
        (
            const cylindricalSinusoidalVelocityProfile2DFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Copy construct
        cylindricalSinusoidalVelocityProfile2DFvPatchVectorField
        (
            const cylindricalSinusoidalVelocityProfile2DFvPatchVectorField&
        );

        //- Construct and return a clone
        virtual tmp<fvPatchVectorField> clone() const
        {
            return tmp<fvPatchVectorField>
            (
                new cylindricalSinusoidalVelocityProfile2DFvPatchVectorField(*this)
            );
        }

        //- Construct as copy setting internal field reference
        cylindricalSinusoidalVelocityProfile2DFvPatchVectorField
        (
            const cylindricalSinusoidalVelocityProfile2DFvPatchVectorField&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct and return a clone setting internal field reference
        virtual tmp<fvPatchVectorField> clone
        (
            const DimensionedField<vector, volMesh>& iF
        ) const
        {
            return tmp<fvPatchVectorField>
            (
                new cylindricalSinusoidalVelocityProfile2DFvPatchVectorField(*this, iF)
            );
        }


    // Member Functions

        // Mapping functions

            //- Map (and resize as needed) from self given a mapping object
            virtual void autoMap
            (
                const fvPatchFieldMapper&
            );

            //- Reverse map the given fvPatchField onto this fvPatchField
            virtual void rmap
            (
                const fvPatchVectorField&,
                const labelList&
            );

            


        // Evaluation functions

            //- Update the coefficients associated with the patch field
            virtual void updateCoeffs();


        //- Write
        virtual void write(Ostream& os) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#endif

// ************************************************************************* //
