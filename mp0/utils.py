
from verse.analysis.analysis_tree import AnalysisTree 
from typing import List, Tuple
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

def is_refine_complete(ownship_init: List[float], intruder_init: List[float], partitions: Tuple[List[float], List[float]]):
    from z3 import Solver, Real, And, Or, Not, sat, unsat
    
    combined_init_lower = ownship_init[0] + intruder_init[0]  # concatenate lower bounds
    combined_init_upper = ownship_init[1] + intruder_init[1]  # concatenate upper bounds
    
    big_rect = np.array([combined_init_lower, combined_init_upper])
    
    combined_partitions = []
    for own_partition, intruder_partition in partitions:
        combined_lower = own_partition[0] + intruder_partition[0]
        combined_upper = own_partition[1] + intruder_partition[1]
        combined_partitions.append([combined_lower, combined_upper])
    
    if not combined_partitions:
        return False
    
    small_rects = np.array(combined_partitions)
    
    def check_coverage(small_rects: np.ndarray, big_rect: np.ndarray) -> bool:
        if not (small_rects.ndim == 3 and small_rects.shape[1] == 2 and
                big_rect.ndim == 2 and big_rect.shape[0] == 2 and
                small_rects.shape[2] == big_rect.shape[1]):
            raise ValueError("Input arrays have incorrect shapes. Expected (K, 2, N) and (2, N).")

        K, _, N = small_rects.shape

        if K == 0:
            return False

        solver = Solver()

        p = [Real(f'p_{i}') for i in range(N)]

        
        in_big_rect_conds = []
        for i in range(N):
            in_big_rect_conds.append(p[i] >= big_rect[0, i])
            in_big_rect_conds.append(p[i] <= big_rect[1, i])
        
        solver.add(And(in_big_rect_conds))

       
        in_any_small_rect_conds = []
        for k in range(K):
            in_small_k_conds = []
            for i in range(N):
                in_small_k_conds.append(p[i] >= small_rects[k, 0, i])
                in_small_k_conds.append(p[i] <= small_rects[k, 1, i])
            
            in_any_small_rect_conds.append(And(in_small_k_conds))

        in_union = Or(in_any_small_rect_conds)

        solver.add(Not(in_union))

        result = solver.check()

        if result == unsat:
            return True
        elif result == sat:
            return False
        else:
            raise RuntimeError(f"Z3 solver returned an unknown status: {result}")
    
    return check_coverage(small_rects, big_rect)

def tree_safe(tree: AnalysisTree):
    for node in tree.nodes:
        if node.assert_hits is not None:
            return False 
    return True