#ifndef BNOL_THRESH_THRESH_H
#define BNOL_THRESH_THRESH_H

#include <vector>
#include "Specimen.h"
#include <cstdio>

namespace bnolThresh {

    typedef std::vector<Specimen> SpecimenVector;
    typedef SpecimenVector::iterator SpecimenIterator;

    class Threshold {
    public:
        Threshold (double g, double m, SpecimenIterator b) :gain{g}, mdlpCriterion{m}, cumulativeExcess(g-m), best{b} {};
        ~Threshold() {
            delete(left);
            delete(right);
        };
        double gain;
        double mdlpCriterion;
        double cumulativeExcess; // sum of (gain - mdlpCriterion) for this plus all child thresholds
        SpecimenIterator best;
        Threshold *left;
        Threshold *right;
    };

    Threshold* FindThreshold(SpecimenIterator begin, SpecimenIterator end, const bool recursive, const bool sorted);
    Threshold* FindThreshold(SpecimenVector specs, const bool recursive);

    void PrintThreshold(Threshold* thresh);

};

#endif
