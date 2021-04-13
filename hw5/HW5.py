

at_establishment = [[1,0,0,0,],[0,1,0,0],[0,1,1,1],[0,1,1,1],[0,1,1,1],[0,0,0,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[0,1,1,1],[0,1,1,1],[1,1,1,1],[0,1,1,1],[0,1,1,1],[1,1,1,1],[0,1,1,1],[0,1,1,1],[0,0,0,1],[0,0,0,1],[1,1,1,1],[0,1,1,1],[0,1,1,1],[0,0,1,1],[0,1,1,1],[0,0,0,1],[0,0,1,1],[0,1,1,1],[0,0,1,1],[1,1,1,1]]
fight_occured = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0]
from collections import defaultdict
import numpy as np
s = ''
memory = defaultdict(int)

def get_instigator(at_establishment, fight_occured):
    is_fight = [ind for ind, v in enumerate(fight_occured) if v == 1]
    if not is_fight:
        return None
    at_establishment_fights = np.array(at_establishment)[is_fight].tolist()
    for fight in at_establishment_fights:
        if np.count_nonzero(np.array(fight)) == 1:
            return  [ind for ind,v in enumerate(fight) if v == 1][0]


instigator_ind = get_instigator(at_establishment, fight_occured)

for i in range(len(at_establishment)):
    if all(at_establishment[i]):
        s += '0'
        memory[str(at_establishment[i])] = fight_occured[i]
    elif all(ae == 0 for ae in at_establishment[i]):
        s += '0'
        memory[str(at_establishment[i])] = 0
    elif np.count_nonzero(np.array(at_establishment[i])) == 1:
        if str(at_establishment[i]) not in memory.keys():
            one_index = [ind for ind, v in enumerate(at_establishment[i]) if v == 1][0]
            if instigator_ind is None:
                s += '2'
                memory[str(at_establishment[i])] = fight_occured[i]
            elif instigator_ind == one_index:
                s += '2'
                memory[str(at_establishment[i])] = fight_occured[i]
            else:
                s += '0'
        else:
            one_index = [ind for ind, v in enumerate(at_establishment[i]) if v == 1][0]
            if instigator_ind == one_index:
                s += '1'
            else:
                s += '0'
    else:
        if str(at_establishment[i]) in memory.keys():
            s += str(memory[str(at_establishment[i])])
        else:
            s += '2'
            memory[str(at_establishment[i])] = fight_occured[i]

print(s)





# actual submission
################
# DO NOT REMOVE
# Versions
# numpy==1.18.0
################
import numpy as np

class Agent(object):
    def __init__(self):
        pass

    def _get_instigator(self, at_establishment, fight_occurred):
        is_fight = [ind for ind, v in enumerate(fight_occurred) if v == 1]
        if not is_fight:
            return None
        at_establishment_fights = np.array(at_establishment)[is_fight].tolist()
        for fight in at_establishment_fights:
            if np.count_nonzero(np.array(fight)) == 1:
                return [ind for ind, v in enumerate(fight) if v == 1][0]

    def solve(self, at_establishment, fight_occurred):

        from collections import defaultdict
        import numpy as np
        s = ''
        memory = defaultdict(int)
        instigator_ind = self._get_instigator(at_establishment, fight_occurred)

        for i in range(len(at_establishment)):
            if all(at_establishment[i]):
                s += '0'
                memory[str(at_establishment[i])] = fight_occurred[i]
            elif all(ae == 0 for ae in at_establishment[i]):
                s += '0'
                memory[str(at_establishment[i])] = 0
            elif np.count_nonzero(np.array(at_establishment[i])) == 1:
                if str(at_establishment[i]) not in memory.keys():
                    one_index = [ind for ind, v in enumerate(at_establishment[i]) if v == 1][0]
                    if instigator_ind is None:
                        s += '2'
                        memory[str(at_establishment[i])] = fight_occurred[i]
                    elif instigator_ind == one_index:
                        s += '2'
                        memory[str(at_establishment[i])] = fight_occurred[i]
                    else:
                        s += '0'
                else:
                    one_index = [ind for ind, v in enumerate(at_establishment[i]) if v == 1][0]
                    if instigator_ind == one_index:
                        s += '1'
                    else:
                        s += '0'
            else:
                if str(at_establishment[i]) in memory.keys():
                    s += str(memory[str(at_establishment[i])])
                else:
                    s += '2'
                    memory[str(at_establishment[i])] = fight_occurred[i]
        return s