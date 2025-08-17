import numpy as np 
import pyvista as pv
from verse import  Scenario, ScenarioConfig
from dubin_agent import CarAgent, NPCAgent
from dubin_sensor import DubinSensor
from dubin_controller import AgentMode

from verse.plotter.plotter3D import *
from verse.plotter.plotter3D_new import *

from  dubin_controller import AgentMode
from utils import eval_safety, tree_safe

import copy
import time


def verify_refine(scenario : Scenario, time_horizon : float, time_step : float, plotter : pv.Plotter):
    assert time_horizon > 0
    assert time_step > 0

    own_acas_plane = scenario.init_dict['air1_#007BFF']
    int_npc_plane = scenario.init_dict['air2_#FF0000']

    traces = []


    ################# YOUR CODE STARTS HERE #################

  
    return traces



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
    ownship_aircraft = [[-4100, -5000, 0, 120], [-3100, -4950, 0, 120]]
    intruder_aircraft =[[-1450, 1500, -np.pi/2, 100], [-1250, 2000, -np.pi/2, 100]]


    #R2
    # ownship_aircraft = [[-2200, -2000, -np.pi/2, 120], [-1200, -950, -np.pi/2, 120]]
    # intruder_aircraft =[[-100, 1500, (-3 * np.pi/4) +1.4, 100], [100, 2000, (-np.pi/12) + 1.4, 100]]


    # #R3
    # ownship_aircraft = [[-7500, -6000, 0, 120], [-7000, -5950, 0, 120]]
    # intruder_aircraft =[[-100, 1500, -np.pi/2, 100], [100, 2000, -np.pi/2, 100]]

    
 
    scenario.set_init_single(
        'air1_#007BFF', ownship_aircraft,(AgentMode.COC,)
    )
    scenario.set_init_single(
        'air2_#FF0000', intruder_aircraft, (AgentMode.COC,)
    )


   
    
    
    # ----------- Simulate: Uncomment this block to perform simulation n times-------------
    # traces = []
    # ax =  pv.Plotter()
    # ax.set_scale(xscale=100.0)
    # n= 1100
    # for i in range(n):
    #     traces.append(scenario.simulate(80, 0.1, ax=ax))

    # eval_safety(traces)
    
    # ax.show_grid(xlabel='time (1/100 s)', ylabel='x', zlabel='y', font_size=10)
    # ax.show()
       
    # -----------------------------------------


    # ------------- simulate from select points -------------

    # ax =  pv.Plotter()
    # ax.set_scale(xscale=100.0)
    # traces = []

    # # You may change the initial states here
    # init_dict_list = #Format: [{ "air1_#007BFF": [-7500, -6000, 0, 120],  "air2_#FF0000": [-100, 1500, -np.pi/2, 100]}, {"air1_#007BFF": [-7200, -5975, 0, 120], "air2_#FF0000": [100, 2000, -np.pi/2, 100]}]
    
    # traces = scenario.simulate_multi(80, 0.1, ax=ax, init_dict_list=init_dict_list)
    # eval_safety(traces)

    # ax.show_grid(xlabel='time (1/100 s)', ylabel='x', zlabel='y', font_size=10)
    # ax.show()
    # -----------------------------------------


    # ----------- verify: Uncomment this block to perform verification without refinement ----------
    # ax =  pv.Plotter()
    # ax.set_scale(xscale=100.0)
    
    # trace = scenario.verify(80, 0.1, ax=ax)
    # for node in trace.nodes:
    #     plot3dReachtubeSingleLive(node.trace, ax, assert_hits=node.assert_hits )
        
    # ax.show_grid(xlabel='time (1/100 s)', ylabel='x', zlabel='y', font_size=10)
    # ax.show()
    # -----------------------------------------

 
    # ------------- Verify refine: Uncomment this block to perform verification with refinement -------------

    # ax = pv.Plotter()
    # ax.set_scale(xscale=100.0)

    # traces = verify_refine(scenario, 80, 0.1, ax)


    # for trace in traces:
    #     for node in trace.nodes:
    #         plot3dReachtubeSingleLive(node.trace, ax, node.assert_hits )
    
    # ax.show_grid(xlabel='time (1/100 s)', ylabel='x', zlabel='y', font_size=10)
    # ax.show()

    # -----------------------------------------