#include "Cluster.h"
#include "Bnol.h"
#include <cmath>

namespace bnolThresh {

    void Cluster::modifyGroup(bnol::group_t group, bool incr){
        if(incr){
            groupCount[group]++;
            if(groupCount[group]==1){
                nGroups++;
            }
        }
        else { // decrement the group
            if(groupCount[group]==0){
                throw std::logic_error("Attempt to decrement empty group");
            }
            groupCount[group]--;
            if(groupCount[group]==0){
                nGroups--;
            }
        }
        nSpecimens += incr ? 1 : -1;
        _entropy = -1; // invalidate the value and force a recalculation
    }

    double Cluster::entropy() {
        if(_entropy < 0 ){ // has been invalidated elsewhere
            _entropy = 0;
            double prop;
            for(auto g = groupCount.cbegin(); g!=groupCount.cend(); g++){
                if(g->second > 0){
                    prop = ((double)g->second) / nSpecimens;
                    _entropy -= prop * log2(prop); // deliberate use of -=
                }
            }
        }
        return _entropy;
    }

}
