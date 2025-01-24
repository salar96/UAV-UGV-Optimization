import numpy as np
from utils import *
from normalizers import *
from scipy.spatial.distance import cdist
from ObstacleAvoidance import *
from FLPO import *
import time

class UAV_Net:
    def __init__(self, drones, num_stations, init_ugv = None, blocks=None,
                 ugv_factor=0.0, fcr=25, distance='euclidean') -> None:
        super().__init__()
        self.den_drones = drones
        self.drones,lb_ub = normalize_drones(drones)
        self.lb,self.ub = lb_ub
        self.scale = self.ub-self.lb
        if init_ugv is None:
            self.stations=np.repeat([0.5,0.5],num_stations,axis=0)
        else:
            self.stations = init_ugv
        if blocks is not None:
          self.blocks=normalize_blocks(blocks,lb_ub)
        else:
          self.blocks=None
        self.N_drones= len(drones)
        self.N_stations=num_stations
        self.stage_horizon=self.N_stations+1
        self.gamma_k_length=self.N_stations+1
        self.fcr=fcr/self.scale
        self.distance=distance
        self.ugv_factor=ugv_factor
        self.bounds = [(0, 1)]*self.N_stations*2
        self.cost_normalizer = 1/(self.N_drones*(self.N_stations**self.N_stations))
        print("UAV Network was successfully created.")
        return

    def return_stagewise_cost(self,params,beta): #params is like stations
        d_F=cdist(params,params,self.distance)
        # strt = time.time()
        if not self.blocks==None:
            for block in self.blocks:
                obs = Obstacle(block)
                d_F=d_F+cdist(params,params,metric=add_block_dist(obs))
        # print('1',time.time()-strt)
        d_F=d_F+np.diag([my_inf]*self.N_stations)
        d_delta_to_f=np.array([my_inf]*self.N_stations).reshape(1,-1)
        d_df=np.concatenate((d_F,d_delta_to_f),axis=0)
        D_ss=[0]*self.N_drones
        for drone_id,drone in enumerate(self.drones):
            stage=np.concatenate((params,np.array(self.drones[drone_id][1]).reshape(1,-1)),axis=0)
            D_s=[0]*(self.stage_horizon+1)
            stage_0=np.array(self.drones[drone_id][0]).reshape(1,-1)
            D_s[0]=cdist(stage_0,stage,self.distance)
            # strt = time.time()
            if not self.blocks==None:
                for block in self.blocks:
                    obs = Obstacle(block)
                    D_s[0]=D_s[0]+cdist(stage_0,stage,metric=add_block_dist(obs))
            #print(D_s[0]-self.drones[drone_id][2]*self.fcr)
            # print('2',time.time()-strt)
            D_s[0]=D_s[0]+penalty(D_s[0]-self.drones[drone_id][2]*self.fcr)
            #D_s[0]=D_s[0]+my_exp(beta*(D_s[0]-self.drones[drone_id][2]*self.fcr))
            #D_s[0][0,-1]=D_s[0][0,-1]*(D_s[0][0,-1]>my_inf)

            delta_id= self.N_stations+drone_id
            # so far we have taken care of the first distance matrix

            d_f_to_delta=cdist(params,np.array(self.drones[drone_id][1]).reshape(1,-1),self.distance)
            # strt = time.time()
            if not self.blocks==None:
                for block in self.blocks:
                    obs = Obstacle(block)
                    d_f_to_delta=d_f_to_delta+cdist(params,np.array(self.drones[drone_id][1]).reshape(1,-1),metric=add_block_dist(obs))
            # print('3',time.time()-strt)
            d_last=np.concatenate((d_f_to_delta,np.array([0]).reshape(1,-1)),axis=0)
            d=np.concatenate((d_df,d_last),axis=1)


            d=d+(penalty(d-self.fcr))
            #d=d+my_exp(beta*(d-self.fcr))
            D_s[1:self.stage_horizon] = [d] * (self.stage_horizon - 1)
            d_l=[my_inf]*(self.gamma_k_length-1)
            d_l.append(0.0)
            D_s[-1]=np.array(d_l).reshape(-1,1)
            D_ss[drone_id]=D_s
        self.D_ss=D_ss
        return
    
    def objective(self,params,beta):
        self.return_stagewise_cost(params.reshape(-1,2),beta)
        cost = 0.0
        for i in range(len(self.D_ss)):
            cost += free_energy_Gibbs(self.D_ss[i],beta)
        if self.ugv_factor == 0.0:
            return cost
        else:
            return cost+self.ugv_factor*np.linalg.norm(params.reshape(-1,2)-self.stations)

        
    def return_total_cost(self):
        return self.cost_fun/self.cost_normalizer*self.scale
    def return_direct_cost(self):
        return np.sum([np.sum((np.array(drone[0])-np.array(drone[1]))**2)**0.5 for drone in self.den_drones])