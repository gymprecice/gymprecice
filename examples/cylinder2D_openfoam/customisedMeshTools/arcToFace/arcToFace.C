/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2017 OpenFOAM Foundation
    Copyright (C) 2018-2020 OpenCFD Ltd.
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

#include "arcToFace.H"
#include "polyMesh.H"
#include "addToRunTimeSelectionTable.H"
#include "unitConversion.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(arcToFace, 0);
    addToRunTimeSelectionTable(topoSetSource, arcToFace, word);
    addToRunTimeSelectionTable(topoSetSource, arcToFace, istream);
    addToRunTimeSelectionTable(topoSetFaceSource, arcToFace, word);
    addToRunTimeSelectionTable(topoSetFaceSource, arcToFace, istream);
    addNamedToRunTimeSelectionTable
    (
        topoSetFaceSource,
        arcToFace,
        word,
        arc
    );
    addNamedToRunTimeSelectionTable
    (
        topoSetFaceSource,
        arcToFace,
        istream,
        arc
    );
}


Foam::topoSetSource::addToUsageTable Foam::arcToFace::usage_
(
    arcToFace::typeName,
    "\n    Usage: arcToFace -w/2 to w/2 theta with arc-centre at theta0\n\n"
    "    Select all faces with face centre within bounding arc\n\n"
);


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::arcToFace::combine(topoSet& set, const bool add) const
{
    const pointField& ctrs = mesh_.faceCentres();
    point arcCtr(radius_*Foam::cos(Foam::degToRad(theta0_)), radius_*Foam::sin(Foam::degToRad(theta0_)), origin_.z());

    labelHashSet patchIDs = mesh_.boundaryMesh().patchSet
    (
        selectedPatches_,
        true,           // warn if not found
        true            // use patch groups if available
    );

    for (const label patchi : patchIDs)
    {
        const polyPatch& pp = mesh_.boundaryMesh()[patchi];

        if (verbose_)
        {
            Info<< "    Found matching patch " << pp.name()
                << " with " << pp.size() << " faces." << endl;
        }

        for
        (
            label facei = pp.start();
            facei < pp.start() + pp.size();
            ++facei
        )
        {
            const vector r((ctrs[facei] - origin_)/mag(ctrs[facei] - origin_));
            vector d((arcCtr - origin_)/mag(arcCtr - origin_));
            scalar theta(Foam::acos(r & d));

            if ((theta >= Foam::degToRad(-w_/2.0)) && (theta <= Foam::degToRad(w_/2.0)))
            {
                addOrDelete(set, facei, add);
            }
        }
    }

    if (patchIDs.empty())
    {
        WarningInFunction
            << "Cannot find any patches matching "
            << flatOutput(selectedPatches_) << nl
            << "Valid names are " << flatOutput(mesh_.boundaryMesh().names())
            << endl;
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::arcToFace::arcToFace
(
    const polyMesh& mesh,
    const wordRe& patchName,
    const vector& origin,
    const scalar radius,
    const scalar w,
    const scalar theta0
)
:
    topoSetFaceSource(mesh),
    selectedPatches_(one{}, patchName),
    origin_(origin),
    radius_(radius),
    w_(w),
    theta0_(theta0)
{}


Foam::arcToFace::arcToFace
(
    const polyMesh& mesh,
    const dictionary& dict
)
:
    topoSetFaceSource(mesh),
    selectedPatches_(),
    origin_(dict.lookup("origin")),
    radius_(dict.get<scalar>("radius")),
    w_(dict.get<scalar>("widthAngle")),
    theta0_(dict.get<scalar>("centreAngle"))
{
    // Look for 'patches' and 'patch', but accept 'name' as well
    if (!dict.readIfPresent("patches", selectedPatches_))
    {
        selectedPatches_.resize(1);
        selectedPatches_.first() =
            dict.getCompat<wordRe>("patch", {{"name", 1806}});
    }
}


Foam::arcToFace::arcToFace
(
    const polyMesh& mesh,
    Istream& is
)
:
    topoSetFaceSource(mesh),
    selectedPatches_(one{}, wordRe(checkIs(is))),
    origin_(checkIs(is)),
    radius_(readScalar(checkIs(is))),
    w_(readScalar(checkIs(is))),
    theta0_(readScalar(checkIs(is)))
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::arcToFace::applyToSet
(
    const topoSetSource::setAction action,
    topoSet& set
) const
{
    if (action == topoSetSource::ADD || action == topoSetSource::NEW)
    {
        if (verbose_)
        {
            Info<< "    Adding patch faces with centre within arc-angle"
            << " [" << -w_/2 << ", " << w_/2 << "]"<< " (degrees unit)"
            << ", centred at " << theta0_ << " (degrees unit)" << endl;
        }

        combine(set, true);
    }
    else if (action == topoSetSource::SUBTRACT)
    {
        if (verbose_)
        {
            Info<< "    Removing patch faces with centre within arc-angle"
            << " [" << -w_/2 << ", " << w_/2 << "]"<< " (degrees unit)"
            << ", centred at " << theta0_ << " (degrees unit)" << endl;
        }

        combine(set, false);
    }
}


// ************************************************************************* //
