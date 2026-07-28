# -*- coding: utf-8 -*-
# %% Header
"""
Created on Sun Nov 10 16:23:12 2024

Dictionary for ALL analysed experiments.

    Every new experiment is entered here and parameters for analysis are specified as needed.
    Don't delete any entry, just add new ones. Compile after each new entry.
    
    Use this format to comment new experiments:
    ## Project: xyz | cohort: xyz | paradigm: xyz  
        # any other information
    
    A JSON file is generated at the very end which lists an overview of all analysed experiments
    and their corresponding parameters.
    
Version 0 (for tinkering)
    Stores the following information: region, reference pairs, filter type and specifications, downsampling.
    Addition: Unassigned channel as noise reference (i.e. one where no wire was soldered to).
    

@author: Juliana Groß
"""

import json

# %% Functions

# Below are default values. Specific changes can be made in the line for the corresponding ID-folder info in the next section.

def experiment_add(
        experiments, 
        ID,
        region="not defined",
        noise_ref=None,       # channel that serves as a "noise" reference, i.e. an electrode
                              # not in the brain but outside to pick up environmental noise
        ref_pairs=None,       # list of channel pairs for differential referencing
                              # NOTE: do not use [] as default — mutable default arguments
                              # are shared across all calls in Python, which can cause
                              # subtle bugs if the list is ever mutated.
        filter_type="FIR", 
        delta1 = 0.00001, 
        delta2 = 0.0001, 
        transition_width = 50,
        cutoff = 100,
        num_taps = None, 
        order = 4,
        downsampling = 1000
        ):
    if ref_pairs is None:
        ref_pairs = []

    experiment_data = {
        "region": region,
        "noise_ref": noise_ref,   # noise reference channel
        "ref_pairs": ref_pairs,   # differential reference pairs
        "filter": {
            "type": filter_type,
            "delta1": delta1,
            "delta2": delta2,
            "transition_width": transition_width,
            "cutoff": cutoff,
            "num_taps": num_taps,
            "order": order
        },
        "downsampling_Hz": downsampling
    }
    # experiments[ID] = experiment_data # Add the new folder data to the dictionary
    # return experiments
#---This is new: store expe entries by region (since there are more regions with the same ID which would be overwritten otherwise)    
    # Ensure ID entry exists
    if ID not in experiments:
        experiments[ID] = {}
    # Store by region key
    experiments[ID][region] = experiment_data
    return experiments

# %% Data for experiments

# Usage:
# experiments = experiment_add(experiments, 'ID', 'region', 'noise reference', [channels], 'filter type', 
#                              delta 1, delta 2, transition width, cutoff, num taps, order, downsampling)
# "noise reference" can be any channel that is not assigned (i.e. no electrode connected).
# You don't have to specify everything. If any argument is not specified, it will get the default value defined above. 
    
def experiment_data():
    experiments = {} # init dictinary
    
# %% Project: Jills paper | cohort: ChR2 | paradigm: EZM
        # left CA1 (CA1_L) with the optical fiber:
    # experiments = experiment_add(experiments, 'ID0714-240826-134543', 'CA1_L', 8, [[1, 3], [5, 7],[2,4]]) # Layers: Pyr 1,3 | Rad 5,7 | Or 2,4
    # experiments = experiment_add(experiments, 'ID0895-240826-112301', 'CA1_L', 8, [[3,4],[5,6],[1,2]]) # Layers: Pyr/Rad 3,4,5,6 | Or 1,2
    # experiments = experiment_add(experiments, 'ID0896-240826-103104', 'CA1_L', 8, [[3,4],[5,6],[1,2]]) # Layers: Pyr/Rad 3,4,5,6 | Or 1,2
    # experiments = experiment_add(experiments, 'ID1098-250126-114609', 'CA1_L', 8, [[31,29],[27,25],[23,21]]) # Layers: Pyr/Or 31,29,27,25 | Rad 23,21
    # experiments = experiment_add(experiments, 'ID1099-250126-122443', 'CA1_L', 8, [[31,29],[27,25],[23,21]]) # Layers: Pyr/Or 31,29,27,25 | Rad 23,21
    # experiments = experiment_add(experiments, 'ID1101-250126-132021', 'CA1_L', 8, [[31,29],[27,25],[23,21]]) # Layers: Pyr/Or 31,29,27,25 | Rad 23,21

        # right CA1 (CA1_R), contralateral to optical fiber: 
        # (noise reference can be the same since it's only the other side of the same implant)
    # experiments = experiment_add(experiments, 'ID0714-240826-134543', 'CA1_R', 8, [[13,15], [9,11],[14,16]]) # Layers: Pyr 13,15 | Rad 9,11 | Or 14,16
    # experiments = experiment_add(experiments, 'ID0895-240826-112301', 'CA1_R', 8, [[27,28],[29,30],[31,32]]) # Layers: Pyr/Rad 27,28,29,30 | Or 31,32
    # experiments = experiment_add(experiments, 'ID0896-240826-103104', 'CA1_R', 8, [[27,28],[29,30],[31,32]]) # Layers: Pyr/Rad 27,28,29,30 | Or 31,32
    # experiments = experiment_add(experiments, 'ID1098-250126-114609', 'CA1_R', 8, [[1,3],[5,7],[9,11]]) # Layers: Pyr/Or 1,3,5,7 | Rad 9,11
    # experiments = experiment_add(experiments, 'ID1099-250126-122443', 'CA1_R', 8, [[1,3],[5,7],[9,11]]) # Layers: Pyr/Or 1,3,5,7 | Rad 9,11
    # experiments = experiment_add(experiments, 'ID1101-250126-132021', 'CA1_R', 8, [[1,3],[5,7],[9,11]]) # Layers: Pyr/Or 1,3,5,7 | Rad 9,11

# %% Project: Jills paper | cohort: ChR2 | paradigm: OFT
        # left CA1 (CA1_L) with the optical fiber:
    experiments = experiment_add(experiments, 'ID0714-240826-134543', 'CA1_L', 8, [[1, 3], [5, 7],[2,4]]) # Layers: Pyr 1,3 | Rad 5,7 | Or 2,4
    experiments = experiment_add(experiments, 'ID0895-240826-112301', 'CA1_L', 8, [[3,4],[5,6],[1,2]]) # Layers: Pyr/Rad 3,4,5,6 | Or 1,2
    experiments = experiment_add(experiments, 'ID0896-240826-103104', 'CA1_L', 8, [[3,4],[5,6],[1,2]]) # Layers: Pyr/Rad 3,4,5,6 | Or 1,2
    experiments = experiment_add(experiments, 'ID1098-250114-102002', 'CA1_L', 8, [[31,29],[27,25],[23,21]]) # Layers: Pyr/Or 31,29,27,25 | Rad 23,21
    experiments = experiment_add(experiments, 'ID1099-250114-121824', 'CA1_L', 8, [[31,29],[27,25],[23,21]]) # Layers: Pyr/Or 31,29,27,25 | Rad 23,21
    experiments = experiment_add(experiments, 'ID1100-250114-133830', 'CA1_L', 8, [[31,29],[27,25],[23,21]]) # Layers: Pyr/Or 31,29,27,25 | Rad 23,21   
    experiments = experiment_add(experiments, 'ID1101-250114-143507', 'CA1_L', 8, [[31,29],[27,25],[23,21]]) # Layers: Pyr/Or 31,29,27,25 | Rad 23,21  
        
        # right CA1 (CA1_R), contralateral to optical fiber:
    experiments = experiment_add(experiments, 'ID0714-240826-134543', 'CA1_R', 8, [[13,15], [9,11],[14,16]]) # Layers: Pyr 13,15 | Rad 9,11 | Or 14,16
    experiments = experiment_add(experiments, 'ID0895-240826-112301', 'CA1_R', 8, [[27,28],[29,30],[31,32]]) # Layers: Pyr/Rad 27,28,29,30 | Or 31,32
    experiments = experiment_add(experiments, 'ID0896-240826-103104', 'CA1_R', 8, [[27,28],[29,30],[31,32]]) # Layers: Pyr/Rad 27,28,29,30 | Or 31,32
    experiments = experiment_add(experiments, 'ID1098-250114-102002', 'CA1_R', 8, [[1,3],[5,7],[9,11]]) # Layers: Pyr/Or 31,29,27,25 | Rad 23,21
    experiments = experiment_add(experiments, 'ID1099-250114-121824', 'CA1_R', 8, [[1,3],[5,7],[9,11]]) # Layers: Pyr/Or 31,29,27,25 | Rad 23,21
    experiments = experiment_add(experiments, 'ID1100-250114-133830', 'CA1_R', 8, [[1,3],[5,7],[9,11]]) # Layers: Pyr/Or 31,29,27,25 | Rad 23,21   
    experiments = experiment_add(experiments, 'ID1101-250114-143507', 'CA1_R', 8, [[1,3],[5,7],[9,11]]) # Layers: Pyr/Or 31,29,27,25 | Rad 23,21  
    
# %% Project: Jills paper | cohort: vLWO | paradigm: EZM   
        # left CA1 with the optical fiber:
    experiments = experiment_add(experiments, 'ID1042-250126-144711', 'CA1_L', 8, [[23,27],[31,27]]) # Layers: Pyr/Or 23,27 | Rad 31 | two Pyr/Or wires (25,29) ripped off during surgery thus the remaining ones need to serve as reference for each other, one Rad wire during building
    experiments = experiment_add(experiments, 'ID1045-250126-152355', 'CA1_L', 8, [[23,29],[27,29]]) # Layers: Pyr/Or 23,27,29 | no Rad | one Pyr/Or wire ripped off, both Rad wires ripped off
    experiments = experiment_add(experiments, 'ID1065-250126-160154', 'CA1_L', 8, [[27,31],[29,31],[23,25]]) # Layers: Pyr/Or 27,29,31 | Rad 23,25
    experiments = experiment_add(experiments, 'ID1081-250126-170024', 'CA1_L', 8, [[25,27],[29,31],[21,23]]) # Layers: Pyr/Or 25,27,29,31 | Rad 21,23
        
        # right CA1 (CA1_R), contralateral to optical fiber:
    experiments = experiment_add(experiments, 'ID1042-250126-144711', 'CA1_R', 8, [[1,5],[3,5],[7,9]]) # Layers: Pyr/Or 1,3,5 | Rad 7,9 | one Pyr/Or wires ripped off during surgery
    experiments = experiment_add(experiments, 'ID1045-250126-152355', 'CA1_R', 8, [[1,5],[3,5],[7,9]]) # Layers: Pyr/Or 1,3,5 | Rad 7,9 | one Pyr/Or wires ripped off during surgery
    experiments = experiment_add(experiments, 'ID1065-250126-160154', 'CA1_R', 8, [[1,5],[3,5],[7,9]]) # Layers: Pyr/Or 1,3,5 | Rad 7,9 | one Pyr/Or wires ripped off during surgery
    experiments = experiment_add(experiments, 'ID1081-250126-170024', 'CA1_R', 8, [[1,5],[3,5],[7,9]]) # Layers: Pyr/Or 1,3,5 | Rad 7,9 | one Pyr/Or wires ripped off during surgery
    
# %% Project: Jills paper | cohort: vLWO | paradigm: OFT   
         # left CA1 with the optical fiber:
    experiments = experiment_add(experiments, 'ID1042-250114-164002', 'CA1_L', 8, [[23,27],[31,27]]) # Layers: Pyr/Or 23,27 | Rad 31 | two Pyr/Or wires (25,29) ripped off during surgery, one Rad wire during building
    experiments = experiment_add(experiments, 'ID1045-250114-172728', 'CA1_L', 8, [[23,29],[27,29]]) # Layers: Pyr/Or 23,27,29 | Rad - | one Pyr/Or wire ripped off, both Rad wires ripped off
    experiments = experiment_add(experiments, 'ID1065-250114-181132', 'CA1_L', 8, [[27,31],[29,31],[23,25]]) # Layers: Pyr/Or 27,29,31 | Rad 23,25
    experiments = experiment_add(experiments, 'ID1081-250114-191229', 'CA1_L', 8, [[25,27],[29,31],[21,23]]) # Layers: Pyr/Or 25,27,29,31 | Rad 21,23
    experiments = experiment_add(experiments, 'ID1118-250114-195909', 'CA1_L', 8, [[27,31],[29,31],[23,25]]) # Layers: Pyr/Or 27,29,31 | Rad 23,25
         
         # right CA1 (CA1_R), contralateral to optical fiber:
    experiments = experiment_add(experiments, 'ID1042-250114-164002', 'CA1_R', 8, [[1,5],[3,5],[7,9]]) # Layers: Pyr/Or 1,3,5 | Rad 7,9 | one Pyr/Or wires ripped off during surgery
    experiments = experiment_add(experiments, 'ID1045-250114-172728', 'CA1_R', 8, [[1,5],[3,5],[7,9]]) # Layers: Pyr/Or 1,3,5 | Rad 7,9 | one Pyr/Or wires ripped off during surgery
    experiments = experiment_add(experiments, 'ID1065-250114-181132', 'CA1_R', 8, [[1,5],[3,5],[7,9]]) # Layers: Pyr/Or 1,3,5 | Rad 7,9 | one Pyr/Or wires ripped off during surgery
    experiments = experiment_add(experiments, 'ID1081-250114-191229', 'CA1_R', 8, [[1,5],[3,5],[7,9]]) # Layers: Pyr/Or 1,3,5 | Rad 7,9 | one Pyr/Or wires ripped off during surgery
    experiments = experiment_add(experiments, 'ID1118-250114-195909', 'CA1_R', 8, [[1,3],[5,7],[9,11]]) # Layers: Pyr/Or 1,3,5,7 | Rad 9,11
     
# %% Project: Jills paper | cohort: vSWO | paradigm: EZM
        # left CA1 (CA1_L) with the optical fiber:
    experiments = experiment_add(experiments, 'ID1095-250215-133231', 'CA1_L', 8, [[27,31],[29,31],[23,25]]) # Layers: Pyr/Or 27,29,31 | Rad 23,25
    experiments = experiment_add(experiments, 'ID1096-250215-143746', 'CA1_L', 8, [[27,31],[29,31],[23,25]]) # Layers: Pyr/Or 27,29,31 | Rad 23,25
    experiments = experiment_add(experiments, 'ID1105-250215-115029', 'CA1_L', 8, [[25,27],[29,31],[21,23]]) # Layers: Pyr/Or 25,27,29,31 | Rad 21,23
    experiments = experiment_add(experiments, 'ID1106-250215-122429', 'CA1_L', 8, [[25,27],[29,31],[21,23]]) # Layers: Pyr/Or 25,27,29,31 | Rad 21,23 | one of Pyr wires kinked but could still be implanted
    experiments = experiment_add(experiments, 'ID1107-250215-140511', 'CA1_L', 8, [[27,31],[29,31],[21,23]]) # Layers: Pyr/Or 27,29,31 | Rad 23,25
        
        # right CA1 (CA1_R), contralateral to optical fiber:
    experiments = experiment_add(experiments, 'ID1095-250215-133231', 'CA1_R', 8, [[3,5],[7,9],[11,13]]) # Layers: Pyr/Or 3,5,7,9 | Rad 11,13
    experiments = experiment_add(experiments, 'ID1096-250215-143746', 'CA1_R', 8, [[1,5],[3,5],[7,9]]) # Layers: Pyr/Or 1,3,5 | Rad 7,9
    experiments = experiment_add(experiments, 'ID1105-250215-115029', 'CA1_R', 8, [[1,3],[5,7],[9,11]]) # Layers: Pyr/Or 1,3,5,7 | Rad 9,11
    experiments = experiment_add(experiments, 'ID1106-250215-122429', 'CA1_R', 8, [[1,3],[5,7],[9,11]]) # Layers: Pyr/Or 1,3,5,7 | Rad 9,11
    experiments = experiment_add(experiments, 'ID1107-250215-140511', 'CA1_R', 8, [[1,3],[5,7],[9,11]]) # Layers: Pyr/Or 1,3,5,7 | Rad 9,11
    
# %% Project: Jills paper | cohort: vSWO | paradigm: OFT
        # left CA1 (CA1_L) with the optical fiber:
            
    experiments = experiment_add(experiments, 'ID1095-250213-114208', 'CA1_L', 8, [[27,31],[29,31],[23,25]]) # Layers: Pyr/Or 27,29,31 | Rad 23,25
    experiments = experiment_add(experiments, 'ID1096-250213-123232', 'CA1_L', 8, [[27,31],[29,31],[23,25]]) # Layers: Pyr/Or 27,29,31 | Rad 23,25
    experiments = experiment_add(experiments, 'ID1105-250213-140003', 'CA1_L', 8, [[25,27],[29,31],[21,23]]) # Layers: Pyr/Or 25,27,29,31 | Rad 21,23
    experiments = experiment_add(experiments, 'ID1106-250213-144832', 'CA1_L', 8, [[25,27],[29,31],[21,23]]) # Layers: Pyr/Or 25,27,29,31 | Rad 21,23 | one of Pyr wires kinked but could still be implanted
    experiments = experiment_add(experiments, 'ID1107-250214-110509', 'CA1_L', 8, [[27,31],[29,31],[21,23]]) # Layers: Pyr/Or 27,29,31 | Rad 23,25
            
        # right CA1 (CA1_R), contralateral to optical fiber:
    experiments = experiment_add(experiments, 'ID1095-250213-114208', 'CA1_R', 8, [[3,5],[7,9],[11,13]]) # Layers: Pyr/Or 3,5,7,9 | Rad 11,13
    experiments = experiment_add(experiments, 'ID1096-250213-123232', 'CA1_R', 8, [[1,5],[3,5],[7,9]]) # Layers: Pyr/Or 1,3,5 | Rad 7,9
    experiments = experiment_add(experiments, 'ID1105-250213-140003', 'CA1_R', 8, [[1,3],[5,7],[9,11]]) # Layers: Pyr/Or 1,3,5,7 | Rad 9,11
    experiments = experiment_add(experiments, 'ID1106-250213-144832', 'CA1_R', 8, [[1,3],[5,7],[9,11]]) # Layers: Pyr/Or 1,3,5,7 | Rad 9,11
    experiments = experiment_add(experiments, 'ID1107-250214-110509', 'CA1_R', 8, [[1,3],[5,7],[9,11]]) # Layers: Pyr/Or 1,3,5,7 | Rad 9,11
     
# %%    
    return experiments

# %% Create Jason file

if __name__ == "__main__":
    experiments = experiment_data()
    with open("experiments.json", 'w') as json_file:
        json.dump(experiments, json_file, indent=4)  # `indent=4` for pretty formatting