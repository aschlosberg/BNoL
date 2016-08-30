#ifndef BNOL_THRESH_CLUSTER_H
#define BNOL_THRESH_CLUSTER_H

#include "Bnol.h"
#include <unordered_map>

namespace bnolThresh {

    class Cluster {
        public:
            Cluster() :nSpecimens{0}, nGroups{0} {};

            void incrementGroup(bnol::group_t group){
                modifyGroup(group, true);
            };
            void decrementGroup(bnol::group_t group){
                modifyGroup(group, false);
            }

            unsigned int specimens() const { return nSpecimens; };
            unsigned int groups() const { return nGroups; };
            double entropy();

        private:
            void modifyGroup(bnol::group_t group, bool incr);
            unsigned int nSpecimens;
            unsigned int nGroups;
            std::unordered_map<bnol::group_t,unsigned int> groupCount;
            double _entropy;
    };

}

#endif
