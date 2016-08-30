#include "Bnol.h"
#include "Specimen.h"
#include "Cluster.cpp"
#include "Threshold.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace bnolThresh {

    void FindSubThreshold(Threshold* parent, Threshold** sub, SpecimenIterator begin, SpecimenIterator end){
        *sub = FindThreshold(begin, end, true, true); // given that we are in a sub threshold, must be recursive + already sorted
        if(*sub){
             parent->cumulativeExcess += (*sub)->cumulativeExcess;
        }
    }

    Threshold* FindThreshold(SpecimenIterator begin, SpecimenIterator end, const bool recursive = true, const bool sorted = false){
        if(!sorted){
            std::sort(begin, end);
        }
        printf("%.6f\n", begin->value);
        Cluster left, right;
        SpecimenIterator spec;
        int len = std::distance(begin, end);

        for(spec=begin; spec!=end; spec++){
            right.incrementGroup(spec->group);
        }

        double baseEntropy = right.entropy(),
            bestEntropy = baseEntropy,
            clusterEntropy;
        unsigned int totalGroups = right.groups();
        SpecimenIterator best = begin;

        for(spec=begin; spec!=end; spec++){
            left.incrementGroup(spec->group);
            right.decrementGroup(spec->group);
            clusterEntropy = (left.entropy()*left.specimens() + right.entropy()*right.specimens()) / len;

            if(clusterEntropy < bestEntropy){
                best = spec;
                bestEntropy = clusterEntropy;
            }
        }

        for(spec=end-1; spec!=best; spec--){
            left.decrementGroup(spec->group);
            right.incrementGroup(spec->group);
        }

        if(left.specimens() + right.specimens() != len){
            throw std::logic_error("Group counts do not add up");
        }

        double delta = log2(pow(3,totalGroups) - 2) - totalGroups*baseEntropy + left.entropy()*left.groups() + right.entropy()*right.groups();
        double mdlpCriterion = (log2(len-1) + delta) / len;
        double gain = baseEntropy - bestEntropy;

        if(gain > mdlpCriterion){
            auto thresh = new Threshold(gain, mdlpCriterion, best);
            if(recursive){
                FindSubThreshold(thresh, &(thresh->left), begin, best+1);
                FindSubThreshold(thresh, &(thresh->right), best+1, end);
            }
            return thresh;
        }
        return NULL;
    }

    Threshold* FindThreshold(SpecimenVector specs, const bool recursive = true){
        return FindThreshold(specs.begin(), specs.end(), recursive);
    }

    void PrintThreshold(Threshold* thresh){
        if(thresh){
            printf("%.6f > %.6f\n", thresh->gain, thresh->mdlpCriterion);
            printf("%.6f to %.6f; Avg: %.6f\n", thresh->best->value, (thresh->best+1)->value, (thresh->best->value + (thresh->best+1)->value)/2);
            printf("%.6f\n", thresh->cumulativeExcess);
            PrintThreshold(thresh->left);
            PrintThreshold(thresh->right);
        }
    }

    int main(){
        Specimen testData[119] = {{1095.16,0},{1238.69,0},{1278.04,0},{1695.13,0},{347.242,0},{369.702,0},{412.916,0},{417.726,0},{419.735,0},{444.955,0},{462.568,0},{462.666,0},{480.291,0},{495.48,0},{499.789,0},{500.095,0},{518.839,0},{521.453,0},{538.821,0},{557.57,0},{561.16,0},{565.075,0},{568.241,0},{583.709,0},{640.132,0},{744.258,0},{766.249,0},{773.487,0},{871.988,0},{902.472,0},{256.856,1},{287.787,1},{314.713,1},{320.781,1},{357.991,1},{386.941,1},{388.485,1},{393.501,1},{410.667,1},{422.779,1},{433.895,1},{434.632,1},{439.938,1},{445.932,1},{465.841,1},{481.057,1},{484.283,1},{497.969,1},{513.092,1},{517.431,1},{542.421,1},{547.854,1},{562.761,1},{578.796,1},{595.477,1},{627.698,1},{641.61,1},{675.837,1},{753.132,1},{104.923,2},{110.196,2},{127.259,2},{130.374,2},{148.086,2},{154.787,2},{158.858,2},{16.2389,2},{19.2193,2},{232.135,2},{257.676,2},{268.835,2},{31.1918,2},{31.289,2},{356.472,2},{35.8564,2},{38.0187,2},{38.6983,2},{39.6036,2},{45.1738,2},{45.949,2},{54.9288,2},{57.5129,2},{63.1694,2},{64.0939,2},{67.0307,2},{70.423,2},{81.1231,2},{89.6348,2},{93.4855,2},{100.559,3},{102.007,3},{106.292,3},{1115.3,3},{117.88,3},{122.115,3},{122.698,3},{166.562,3},{177.301,3},{180.534,3},{194.242,3},{217.396,3},{227.111,3},{227.462,3},{2385.38,3},{241.933,3},{247.756,3},{289.019,3},{300.946,3},{33.361,3},{47.2235,3},{50.5685,3},{64.8005,3},{70.9747,3},{77.2164,3},{78.4484,3},{81.0224,3},{91.5544,3},{97.037,3},{97.902,3}};
        // Specimen testData[16] = {{1,1},{2,1},{3,0},{4,0},{5,0},{6,0},{7,0},{8,0},{9,1},{10,1},{11,1},{12,1},{13,1},{14,1},{15,0},{16,0}};
        //Specimen testData[16] = {{1,0},{2,0},{3,0},{4,0},{5,0},{6,0},{7,0},{8,0},{9,1},{10,1},{11,1},{12,1},{13,1},{14,1},{15,1},{16,1}};
        SpecimenVector specs(testData, testData + sizeof(testData) / sizeof(testData[0]));
        auto thresh = FindThreshold(specs, false);
        PrintThreshold(thresh);
        delete(thresh);
        return 0;
    }
}

int main(){
    return bnolThresh::main();
}
