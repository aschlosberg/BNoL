#ifndef BNOL_THRESH_SPEC_H
#define BNOL_THRESH_SPEC_H

#include "Bnol.h"

namespace bnolThresh {

    class Specimen {
    public:
        Specimen(double v, bnol::group_t g) :value{v}, group{g} {};
        double value; // value and group should ideally be const values, but this causes issues with copy() when using sort - just don't change them!
        bnol::group_t group;
        bool operator < (const Specimen& s) const { return value < s.value; }; // for std::sort
    };

}

#endif
