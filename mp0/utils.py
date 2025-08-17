
from verse.analysis.analysis_tree import AnalysisTree 
from typing import List
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    RED = '\033[31m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def eval_safety(tree_list: List[AnalysisTree]):
    agent_id = 'air1_#007BFF'
    unsafe_init = []
    for tree in tree_list:
        assert agent_id in tree.root.init
        leaves = list(filter(lambda node: node.child == [], tree.nodes))
        unsafe = list(filter(lambda node: (node.assert_hits != None) and (node.assert_hits !={}), leaves))
        if len(unsafe) != 0:
            print(bcolors.RED + f"Unsafety Detected in Tree With Init {tree.root.init}😫" + bcolors.ENDC)
            unsafe_init.append(tree.root.init)
          
    if len(unsafe_init) == 0:
        print(bcolors.OKGREEN + f"No Unsafety detected!🥰" + bcolors.ENDC)
    else:
        print(bcolors.RED + f"Unsafety detected" + bcolors.ENDC)
    
    return unsafe_init




def tree_safe(tree: AnalysisTree):
    for node in tree.nodes:
        if node.assert_hits is not None:
            return False 
    return True