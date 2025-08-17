from enum import Enum, auto
import copy
from typing import List


class AgentMode(Enum):
    COC = auto()
    WL = auto()
    WR = auto()
    SL = auto()
    SR = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    M01 = auto()
    M12 = auto()
    M21 = auto()
    M10 = auto()


class State:
    '''x: float
    y: float
    theta: float
    v: float
    agent_mode: AgentMode'''
    rho: float
    theta: float
    psi: float
    v_own: float
    agent_mode: AgentMode

    def __init__(self, rho, theta, psi, v_own, timer_DL, agent_mode: AgentMode):#__init__(self, x, y, theta, v, agent_mode: AgentMode):
        pass


def decisionLogic(ego: State, others: List[State]):
    next = copy.deepcopy(ego)
    rho = ego.rho
    theta = ego.theta
   
    
    if rho <= 3000:
        if theta <= -0.3: 
            if ego.agent_mode != AgentMode.SR:
                next.agent_mode = AgentMode.SR
        if theta > 0.3:
            if ego.agent_mode != AgentMode.SL:
                next.agent_mode = AgentMode.SL
    if rho > 4000 and rho < 8000:
        if theta <= -0.3:
            if ego.agent_mode != AgentMode.WR:
                next.agent_mode = AgentMode.WR
        if theta > 0.3:
            if ego.agent_mode != AgentMode.WL:
                next.agent_mode = AgentMode.WL
                  
    assert rho > 25, "too close"

    return next
