import numpy as np
from dubin_agent import CarAgent

from verse.analysis.utils import wrap_to_pi
    


def dist(pnt1, pnt2):
    return np.linalg.norm(
        np.array(pnt1) - np.array(pnt2)
    )

def get_extreme(rect1, rect2):
    lb11 = rect1[0]
    lb12 = rect1[1]
    ub11 = rect1[2]
    ub12 = rect1[3]

    lb21 = rect2[0]
    lb22 = rect2[1]
    ub21 = rect2[2]
    ub22 = rect2[3]

    # Using rect 2 as reference
    left = lb21 > ub11 
    right = ub21 < lb11 
    bottom = lb22 > ub12
    top = ub22 < lb12

    if top and left: 
        dist_min = dist((ub11, lb12),(lb21, ub22))
        dist_max = dist((lb11, ub12),(ub21, lb22))
    elif bottom and left:
        dist_min = dist((ub11, ub12),(lb21, lb22))
        dist_max = dist((lb11, lb12),(ub21, ub22))
    elif top and right:
        dist_min = dist((lb11, lb12), (ub21, ub22))
        dist_max = dist((ub11, ub12), (lb21, lb22))
    elif bottom and right:
        dist_min = dist((lb11, ub12),(ub21, lb22))
        dist_max = dist((ub11, lb12),(lb21, ub22))
    elif left:
        dist_min = lb21 - ub11 
        dist_max = np.sqrt((lb21 - ub11)**2 + max((ub22-lb12)**2, (ub12-lb22)**2))
    elif right: 
        dist_min = lb11 - ub21 
        dist_max = np.sqrt((lb21 - ub11)**2 + max((ub22-lb12)**2, (ub12-lb22)**2))
    elif top: 
        dist_min = lb12 - ub22
        dist_max = np.sqrt((ub12 - lb22)**2 + max((ub21-lb11)**2, (ub11-lb21)**2))
    elif bottom: 
        dist_min = lb22 - ub12 
        dist_max = np.sqrt((ub22 - lb12)**2 + max((ub21-lb11)**2, (ub11-lb21)**2)) 
    else: 
        dist_min = 0 
        dist_max = max(
            dist((lb11, lb12), (ub21, ub22)),
            dist((lb11, ub12), (ub21, lb22)),
            dist((ub11, lb12), (lb21, ub12)),
            dist((ub11, ub12), (lb21, lb22))
        )
    return dist_min, dist_max

class DubinSensor():
    def sense(self, agent: CarAgent, state_dict, lane_map, simulate = False):
        len_dict = {}
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        ego_name = "air1_#007BFF"
        other_name = "air2_#FF0000"
        if simulate: #tmp.ndim < 2:
            if agent.id == "air1_#007BFF":
                ego_name = "air1_#007BFF"
                other_name = "air2_#FF0000"
            
            if agent.id == "air2_#FF0000":
                ego_name = "air2_#FF0000"
                other_name = "air1_#007BFF"

            len_dict['others'] = 1 
            
            curr_x = state_dict[ego_name][0][1]
            curr_y = state_dict[ego_name][0][2]
            curr_theta = state_dict[ego_name][0][3]
            obstacle_x = state_dict[other_name][0][1]
            obstacle_y = state_dict[other_name][0][2]
            
            # ego_time = state_dict[ego_name][0][5]
            
            
            # Calcs from Stanley Bak (acasxu_closed_loop_sim/acasxu_dubins/acasxu_dubins.py state7_to_state5() function)
            dy = obstacle_y - curr_y
            dx = obstacle_x - curr_x
                
            rho = np.sqrt((curr_x - obstacle_x)**2 + (curr_y - obstacle_y)**2)

                
            theta = np.arctan2(dy, dx) - curr_theta

            theta = np.arctan2(np.sin(theta),np.cos(theta))
                
            cont['ego.rho'] = rho
            cont['ego.theta'] = theta
            # cont['ego.timer_DL'] = ego_time
            
            disc['ego.agent_mode'] = state_dict[ego_name][1][0]
                
        else:
            if agent.id == "air1_#007BFF":
                ego_name = "air1_#007BFF"
                other_name = "air2_#FF0000"
            
            if agent.id == "air2_#FF0000":
                ego_name = "air2_#FF0000"
                other_name = "air1_#007BFF"
            
            len_dict['others'] = 1 
            curr_x_min = state_dict[ego_name][0][0][1]
            curr_y_min = state_dict[ego_name][0][0][2]
            curr_theta_min = state_dict[ego_name][0][0][3]
            curr_v_min = state_dict[ego_name][0][0][4]
            obstacle_x_min = state_dict[other_name][0][0][1]
            obstacle_y_min = state_dict[other_name][0][0][2]
            
            # Timer variable, no uncertainty
            # ego_time = state_dict[ego_name][0][0][5]
            curr_x_max = state_dict[ego_name][0][1][1]
            curr_y_max = state_dict[ego_name][0][1][2]
            curr_theta_max = state_dict[ego_name][0][1][3]

            obstacle_x_max = state_dict[other_name][0][1][1]
            obstacle_y_max = state_dict[other_name][0][1][2]


            ego_rect = [curr_x_min, curr_y_min, curr_x_max, curr_y_max]
            obstacle_rect = [obstacle_x_min, obstacle_y_min, obstacle_x_max, obstacle_y_max]
            rho_min, rho_max = get_extreme(ego_rect, obstacle_rect)
            
            # Angular ranges
            sign_dy_max = np.max([obstacle_y_max - curr_y_min, obstacle_y_min - curr_y_max])
            sign_dx_max = np.max([obstacle_x_max - curr_x_min, obstacle_x_min - curr_x_max])
            
            sign_dy_min = np.min([obstacle_y_max - curr_y_min, obstacle_y_min - curr_y_max])
            sign_dx_min = np.min([obstacle_x_max - curr_x_min, obstacle_x_min - curr_x_max])
            
            theta_max = np.max([np.arctan2(sign_dy_max, sign_dx_min), np.arctan2(sign_dy_min, sign_dx_max), np.arctan2(sign_dy_max, sign_dx_max), np.arctan2(sign_dy_min, sign_dx_min)]) - curr_theta_min
            theta_min = np.min([np.arctan2(sign_dy_max, sign_dx_min), np.arctan2(sign_dy_min, sign_dx_max), np.arctan2(sign_dy_max, sign_dx_max), np.arctan2(sign_dy_min, sign_dx_min)]) - curr_theta_max

            arho_min = np.inf
            arho_max = -np.inf
            
            own_ext = [(curr_x_min, curr_y_min), (curr_x_max, curr_y_max), (curr_x_min, curr_y_max), (curr_x_max, curr_y_min)]
            int_ext = [(obstacle_x_min, obstacle_y_min), (obstacle_x_max, obstacle_y_max), (obstacle_x_min, obstacle_y_max), (obstacle_x_max, obstacle_y_min)]
            for own_vert in own_ext:
                for int_vert in int_ext:
                    arho = np.arctan2(int_vert[1]-own_vert[1],int_vert[0]-own_vert[0]) % (2*np.pi)
                    arho_max = max(arho_max, arho)
                    arho_min = min(arho_min, arho)
            
            theta_min = wrap_to_pi((2*np.pi-curr_theta_max)+arho_min)
            theta_max = wrap_to_pi((2*np.pi-curr_theta_min)+arho_max) 


            cont['ego.theta'] = [theta_min, theta_max]
            cont['ego.rho'] = [
                rho_min, rho_max
            ]
            # cont['ego.timer_DL'] = [ego_time, ego_time]
            disc['ego.agent_mode'] = state_dict[ego_name][1][0]

        
        return cont, disc, len_dict