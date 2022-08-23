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

\*---------------------------------------------------------------------------*/

#include "cylindricalSinusoidalVelocityProfile2DFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "unitConversion.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

Foam::scalar Foam::cylindricalSinusoidalVelocityProfile2DFvPatchVectorField::t() const
{
    return db().time().timeOutputValue();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::cylindricalSinusoidalVelocityProfile2DFvPatchVectorField::
cylindricalSinusoidalVelocityProfile2DFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(p, iF),
    origin_(Zero),
    radius_(0.0),
    w_(0.0),
    theta0_(0.0),
    flowRate_(),
    rhoName_("rho"),
    rhoInlet_(0.0),
    volumetric_(true)
    
{}


Foam::cylindricalSinusoidalVelocityProfile2DFvPatchVectorField::
cylindricalSinusoidalVelocityProfile2DFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchVectorField(p, iF),
    origin_(dict.getCompat<vector>("origin", {{"centre", 1712}})),
    radius_(dict.get<scalar>("radius")),
    w_(dict.get<scalar>("arcWidth")),
    theta0_(dict.get<scalar>("arcCentre")),
    rhoName_("rho"),
    rhoInlet_(dict.getOrDefault<scalar>("rhoInlet", -VGREAT)),
    volumetric_(true)
{
    if (dict.found("volumetricFlowRate"))
    {
        volumetric_ = true;
        flowRate_ =
            Function1<scalar>::New("volumetricFlowRate", dict, &db());
    }
    else if (dict.found("massFlowRate"))
    {
        volumetric_ = false;
        flowRate_ = Function1<scalar>::New("massFlowRate", dict, &db());
        rhoName_ = dict.getOrDefault<word>("rho", "rho");
    }
    else
    {
        FatalIOErrorInFunction(dict)
            << "Please supply either 'volumetricFlowRate' or"
            << " 'massFlowRate' and 'rho'" << nl
            << exit(FatalIOError);
    }

    // Value field require if mass based
    if (dict.found("value"))
    {
        fvPatchField<vector>::operator=
        (
            vectorField("value", dict, p.size())
        );
    }
    else
    {
        evaluate(Pstream::commsTypes::blocking);
    }
}


Foam::cylindricalSinusoidalVelocityProfile2DFvPatchVectorField::
cylindricalSinusoidalVelocityProfile2DFvPatchVectorField
(
    const cylindricalSinusoidalVelocityProfile2DFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper),
    origin_(ptf.origin_),
    radius_(ptf.radius_),
    w_(ptf.w_),
    theta0_(ptf.theta0_),
    flowRate_(ptf.flowRate_.clone()),
    rhoName_(ptf.rhoName_),
    rhoInlet_(ptf.rhoInlet_),
    volumetric_(ptf.volumetric_)
{}


Foam::cylindricalSinusoidalVelocityProfile2DFvPatchVectorField::
cylindricalSinusoidalVelocityProfile2DFvPatchVectorField
(
    const cylindricalSinusoidalVelocityProfile2DFvPatchVectorField& ptf
)
:
    fixedValueFvPatchVectorField(ptf),
    origin_(ptf.origin_),
    radius_(ptf.radius_),
    w_(ptf.w_),
    theta0_(ptf.theta0_),
    flowRate_(ptf.flowRate_.clone()),
    rhoName_(ptf.rhoName_),
    rhoInlet_(ptf.rhoInlet_),
    volumetric_(ptf.volumetric_)
{}


Foam::cylindricalSinusoidalVelocityProfile2DFvPatchVectorField::
cylindricalSinusoidalVelocityProfile2DFvPatchVectorField
(
    const cylindricalSinusoidalVelocityProfile2DFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(ptf, iF),
    origin_(ptf.origin_),
    radius_(ptf.radius_),
    w_(ptf.w_),
    theta0_(ptf.theta0_),
    flowRate_(ptf.flowRate_.clone()),
    rhoName_(ptf.rhoName_),
    rhoInlet_(ptf.rhoInlet_),
    volumetric_(ptf.volumetric_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::cylindricalSinusoidalVelocityProfile2DFvPatchVectorField::autoMap
(
    const fvPatchFieldMapper& m
)
{}


void Foam::cylindricalSinusoidalVelocityProfile2DFvPatchVectorField::rmap
(
    const fvPatchVectorField& ptf,
    const labelList& addr
)
{}

template<class RhoType>
Foam::vectorField Foam::cylindricalSinusoidalVelocityProfile2DFvPatchVectorField::evaluateVelocityProfile(const RhoType& rho)
{
    // Get patch normal unit vectors
    const vectorField n(patch().nf());

    // Get patch centre
    point arcCtr(radius_*Foam::cos(Foam::degToRad(theta0_)), radius_*Foam::sin(Foam::degToRad(theta0_)), origin_.z());

    //- Magnitude of uniform velocity profile on the cylindrical-arc patch
    const scalar avgU = -flowRate_->value(t())/gSum(rho*patch().magSf());
    
    // Reference axis to compute theta
    vector d((arcCtr - origin_)/mag(arcCtr - origin_));
    
    const vectorField r((patch().Cf() - origin_)/mag(patch().Cf() - origin_));

    scalarField theta(Foam::acos(r & d));

    // Return sinusoidal velocity profile on the patch 
    return (avgU*Foam::constant::mathematical::piByTwo*Foam::cos(Foam::constant::mathematical::pi/Foam::degToRad(w_)*theta) * n);
}

template<class RhoType>
void Foam::cylindricalSinusoidalVelocityProfile2DFvPatchVectorField::correctVelocityProfile(Foam::vectorField& U_profile, const RhoType& rho)
{
    const vectorField n(patch().nf());

    // Calculate the flux (flow rate) through the patch
    const scalar Q_calc = gSum(U_profile & patch().Sf());
    // Calculate the difference between given flow-rate and computed flow-rate based on sinusoidal velocity profile
    const scalar Q_err = -flowRate_->value(t()) - Q_calc;
    const vectorField U_err = Q_err/gSum(rho*patch().magSf()) * n;

    // Correct sinusoidal velocity profile for the error
    U_profile += U_err;
}

template<class RhoType>
void Foam::cylindricalSinusoidalVelocityProfile2DFvPatchVectorField::updateValues
(
    const RhoType& rho
)
{
    if (flowRate_->value(t()) == 0.0) // No-slip BC for Q=0.0
    {
       vectorField U_profile(this->size(), Zero);
       operator==(U_profile);
    }
    else
    {
        vectorField U_profile = evaluateVelocityProfile(rho);
        correctVelocityProfile(U_profile, rho);
        operator==(U_profile);
    }
    
}


void Foam::cylindricalSinusoidalVelocityProfile2DFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    if (volumetric_ || rhoName_ == "none")
    {
        updateValues(one{});
    }
    else
    {
        // Mass flow-rate
        if (db().foundObject<volScalarField>(rhoName_))
        {
            const fvPatchField<scalar>& rhop =
                patch().lookupPatchField<volScalarField, scalar>(rhoName_);

            updateValues(rhop);
        }
        else
        {
            // Use constant density
            if (rhoInlet_ < 0)
            {
                FatalErrorInFunction
                    << "Did not find registered density field " << rhoName_
                    << " and no constant density 'rhoInlet' specified"
                    << exit(FatalError);
            }
            updateValues(rhoInlet_);
        }
    }

    fixedValueFvPatchVectorField::updateCoeffs();
}


void Foam::cylindricalSinusoidalVelocityProfile2DFvPatchVectorField::write
(
    Ostream& os
) const
{
    fvPatchVectorField::write(os);
    os.writeEntry("origin", origin_);
    os.writeEntry("radius", radius_);
    os.writeEntry("arcWidth", w_);
    os.writeEntry("arcCentre", theta0_);
    flowRate_->writeData(os);
    if (!volumetric_)
    {
        os.writeEntryIfDifferent<word>("rho", "rho", rhoName_);
        os.writeEntryIfDifferent<scalar>("rhoInlet", -VGREAT, rhoInlet_);
    }
    writeEntry("value", os);
}


// * * * * * * * * * * * * * * Build Macro Function  * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchVectorField,
        cylindricalSinusoidalVelocityProfile2DFvPatchVectorField
    );
}

// ************************************************************************* //
