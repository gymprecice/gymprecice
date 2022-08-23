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

Class
    Foam::arcToFace

Description
    A \c topoSetFaceSource to select all faces whose face centre
    inside a given bounding arc.

    Operands:
    \table
      Operand   | Type    | Location
      output    | faceSet | $FOAM_CASE/constant/polyMesh/sets/\<set\>
    \endtable

Usage
    Minimal example by using \c system/topoSetDict.actions:
    \verbatim
    {
        // Mandatory (inherited) entries
        name           <name>;
        type           faceSet;
        action         <action>;

        // Mandatory entries
        source         arcToFace;
        origin         (<oX> <oY> <oZ>); 
        radius         <radius>;
        arcCentre    <arcCentre>; 
        arcWidth     <arcWidth>; 

        // Conditional mandatory entries
        // Select either of the below

        // Option-1
        patches
        (
            <patchName1>
            <patchName2>
            ...
        );

        // Option-2
        patch    <patchName>;
    }
    \endverbatim

    where the entries mean:
    \table
      Property   | Description                         | Type | Req'd | Dflt
      name       | Name of faceSet                     | word |  yes  | -
      type       | Type name: faceSet                  | word |  yes  | -
      action     | Action applied on faces - see below | word |  yes  | -
      source     | Source name: arcToFace         | word |  yes  | -
      origin     | Coordinate of arc centre | vector |  yes  | -
      radius     | arc radius              | scalar |  yes  | -
      centreAngle     | arc centre angle in degrees- measured w.r.t (1 0 0) axis  | scalar |  yes  | -
      widthAngle    | arc width in degrees | scalar |  yes  | -
    \endtable

    Options for the \c action entry:
    \verbatim
      new      | Create a new faceSet from selected faces
      add      | Add selected faces into this faceSet
      subtract | Remove selected faces from this faceSet
    \endverbatim

See also
    - Foam::topoSetSource
    - Foam::topoSetFaceSource
    - Foam::patchToFace

SourceFiles
    arcToFace.C

\*---------------------------------------------------------------------------*/

#ifndef arcToFace_H
#define arcToFace_H

#include "topoSetFaceSource.H"
#include "wordRes.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class arcToFace Declaration
\*---------------------------------------------------------------------------*/

class arcToFace
:
    public topoSetFaceSource
{
    // Private Data

        //- Add usage string
        static addToUsageTable usage_;

        //- Matcher for patches
        wordRes selectedPatches_;

        //- arc origin
        const vector origin_;

        //- arc radius
        scalar radius_;
        
        //- arc angle
        const scalar w_;

        //- arc centre
        const scalar theta0_;




    // Private Member Functions

        void combine(topoSet& set, const bool add) const;


public:

    //- Runtime type information
    TypeName("arcToFace");


    // Constructors

        //- Construct from components
        arcToFace
        (
            const polyMesh& mesh,
            const wordRe& patchName,
            const vector& origin,
            const scalar radius,
            const scalar w,
            const scalar theta0
        );

        //- Construct from dictionary
        arcToFace(const polyMesh& mesh, const dictionary& dict);

        //- Construct from Istream
        arcToFace(const polyMesh& mesh, Istream& is);


    //- Destructor
    virtual ~arcToFace() = default;


    // Member Functions

        virtual void applyToSet
        (
            const topoSetSource::setAction action,
            topoSet& set
        ) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
