import numpy as np 
import pyvista as pv
from verse import  Scenario, ScenarioConfig
from dubin_agent import CarAgent, NPCAgent
from dubin_sensor import DubinSensor
from dubin_controller import AgentMode
from typing import List

from verse.plotter.plotter3D import *
from verse.plotter.plotter3D_new import *

from  dubin_controller import AgentMode
from utils import eval_safety, tree_safe, is_refine_complete

import copy
import time


def verify_refine(scenario : Scenario, ownship_aircraft_init : List[float], intruder_aircraft_init : List[float], time_horizon : float, time_step : float, ax : pv.Plotter):
    assert time_horizon > 0
    assert time_step > 0

    #Each safe trace from verify() function must be appended to trace list
    final_traces = []
   
    #Each partition of the initial set included in your final output must be appended to partitions lists
    partitions =[]

    # Example: 
    # trace = scenario.verify( modified own init, modified intruder init, time_horizon, time_step, ax=ax) 
    # if tree_safe(trace): 
    #   final_traces.append(trace)
    #   partitions.append( (modified own init, modified intruder init) )
    # else:
    #   ....


    ################# YOUR CODE STARTS HERE #################

   

    #########################################################


    assert is_refine_complete(ownship_aircraft_init, intruder_aircraft_init, partitions)


    return final_traces



if __name__ == "__main__":
    import os 
    
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "dubin_controller.py")
    ownship = CarAgent('air1_#007BFF', file_name=input_code_name)
    intruder = NPCAgent('air2_#FF0000')

    scenario = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))

    scenario.add_agent(ownship) 
    scenario.add_agent(intruder)
    scenario.set_sensor(DubinSensor())

    # # ----------- Different initial ranges -------------
    #R1:
    ownship_aircraft_init = [[-4100, -5000, 0, 120], [-3100, -4950, 0, 120]]
    intruder_aircraft_init =[[-1450, 1500, -np.pi/2, 100], [-1250, 2000, -np.pi/2, 100]]


    #R2
    # ownship_aircraft_init = [[-2200, -2000, -np.pi/2, 120], [-1200, -950, -np.pi/2, 120]]
    # intruder_aircraft_init =[[-100, 1500, (-3 * np.pi/4) +1.4, 100], [100, 2000, (-np.pi/12) + 1.4, 100]]


    # #R3
    # ownship_aircraft_init = [[-7500, -6000, 0, 120], [-7000, -5950, 0, 120]]
    # intruder_aircraft_init =[[-100, 1500, -np.pi/2, 100], [100, 2000, -np.pi/2, 100]]


    
 
    scenario.set_init_single(
        'air1_#007BFF', ownship_aircraft_init,(AgentMode.COC,)
    )
    scenario.set_init_single(
        'air2_#FF0000', intruder_aircraft_init, (AgentMode.COC,)
    )



    
    # ----------- Simulate: Uncomment this block to perform simulation n times-------------
    ax =  pv.Plotter()
    ax.set_scale(xscale=100.0)
    n= #change number of simulations
    traces = []
    for i in range(n):
        traces.append(scenario.simulate( ownship_aircraft_init, intruder_aircraft_init, 80, 0.1, ax=ax))

    eval_safety(traces)
    ax.add_legend([["Own Plane", 'blue'], ["Intruder Plane", 'red']])

    ax.show_grid(xtitle='time (1/100 s)', ytitle='x', ztitle='y', font_size=10)
    ax.show()
       
    # -----------------------------------------


    # ------------- simulate from select points -------------

    # ax =  pv.Plotter()
    # ax.set_scale(xscale=100.0)
    # traces = []

    # # You may change the initial states here
    # init_dict_list = #Format: [{ "air1_#007BFF": [-7500, -6000, 0, 120],  "air2_#FF0000": [-100, 1500, -np.pi/2, 100]}, {"air1_#007BFF": [-7200, -5975, 0, 120], "air2_#FF0000": [100, 2000, -np.pi/2, 100]}]
    
    # traces = scenario.simulate_multi(ownship_aircraft_init, intruder_aircraft_init, 80, 0.1, ax=ax, init_dict_list=init_dict_list)
    # eval_safety(traces)
    # ax.add_legend([["Own Plane", 'blue'], ["Intruder Plane", 'red']])

    # ax.show_grid(xtitle='time (1/100 s)', ytitle='x', ztitle='y', font_size=10)
    # ax.show()
    # -----------------------------------------


    # ----------- verify: Uncomment this block to perform verification without refinement ----------
    # ax =  pv.Plotter()
    # ax.set_scale(xscale=100.0)
    
    # trace = scenario.verify(ownship_aircraft_init, intruder_aircraft_init, 80, 0.1, ax=ax)
    # for node in trace.nodes:
    #     plot3dReachtubeSingleLive(node.trace, ax, assert_hits=node.assert_hits )
    # ax.add_legend([["Own Plane", 'blue'], ["Intruder Plane", 'red']])

    # ax.show_grid(xtitle='time (1/100 s)', ytitle='x', ztitle='y', font_size=10)
    # ax.show()
    # -----------------------------------------

 
    # ------------- Verify refine: Uncomment this block to perform verification with refinement -------------

    # ax = pv.Plotter()
    # ax.set_scale(xscale=100.0)

    # traces = verify_refine(scenario, ownship_aircraft_init, intruder_aircraft_init, 80, 0.1, ax)


    # for trace in traces:
    #     for node in trace.nodes:
    #         plot3dReachtubeSingleLive(node.trace, ax, node.assert_hits )


    # ax.add_legend([["Own Plane", 'blue'], ["Intruder Plane", 'red']])
    
    # ax.show_grid(xtitle='time (1/100 s)', ytitle='x', ztitle='y', font_size=10)
    # ax.show()

    # -----------------------------------------