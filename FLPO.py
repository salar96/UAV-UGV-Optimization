import numpy as np
from utils import *

def calc_associations(D_ss,beta):
        """ This function calculates the association probabilities iver all drones for all stages.
        The input D_ss is a list, each element is another list, where each element of the latter is cost matrix 
        from one stage to another.
        beta is the temperature parameter. """
        p=[]
        # self.return_stagewise_cost(self.params.reshape(-1,2),beta)
        # D_ss=self.D_ss
        for D_s in D_ss:
            K=len(D_s)
            D=D_s[::-1]
            out_D=[0]*(K+1)
            out_D[0]=np.array([0.0]).reshape(-1,1)
            out_p=[0]*(K+1)
            out_p[0]=np.array([1.0]).reshape(-1,1)
            out=[0]*(K+1)
            out[0]=np.array([1.0]).reshape(-1,1)
            for i in range(1,K+1):
                out_D[i]=(D[i-1]+np.repeat(np.transpose(out_D[i-1]),D[i-1].shape[0],axis=0))
                m=out_D[i].min(axis=1,keepdims=True)
                exp_D=np.exp(np.multiply(-beta,out_D[i]-m))
                out[i]=np.sum(np.multiply(exp_D,np.tile(out[i-1], (1,D[i-1].shape[0])).T),axis=1,keepdims=True)
                out_p[i]=np.divide(np.multiply(exp_D,out[i-1].T),out[i])
                out_D[i]=m
            p.append(out_p[::-1][:-1])
        # self.P_ss=p
        return p

def free_energy(D_s,P_s,beta):
        '''
        Implementation of the free energy function given the cost matrices and association matrices.
        input: D_s: a list of K numpy arrays corrosponding to distances between stages
        P_s: a list of K numpy arrays corrosponding to probabilities between stages

        output: out_c: K+1 numpy arrays with shape[1]=1, indicating the total cost of nodes or the cost of first node which
        gives you the free energy.
        '''

        K=len(D_s)
        D=D_s[::-1]
        P=P_s[::-1]
        out_P=[0]*(K+1)
        out_C=[0]*(K+1)
        out_H=[0]*(K+1)
        out_P[0]=np.array([1.0]).reshape(-1,1)
        out_C[0]=np.array([0.0]).reshape(-1,1)
        out_H[0]=np.array([0.0]).reshape(-1,1)
        for i in range(1,K+1):
          # assigning P of each node for calculating C in the next i
          out_P[i]=(P[i-1]*np.repeat(np.transpose(out_P[i-1]),P[i-1].shape[0],axis=0)).sum(axis=1).reshape(-1,1)
          out_C[i]=(P[i-1]*(D[i-1]*np.repeat(np.transpose(out_P[i-1]),D[i-1].shape[0],axis=0)+np.repeat(np.transpose(out_C[i-1]),D[i-1].shape[0],axis=0))).sum(axis=1).reshape(-1,1)
          out_H[i]=-(P[i-1]*(my_log(P[i-1])*np.repeat(np.transpose(out_P[i-1]),D[i-1].shape[0],axis=0)-np.repeat(np.transpose(out_H[i-1]),D[i-1].shape[0],axis=0))).sum(axis=1).reshape(-1,1)
        # D-1/beta*H
        return (out_C[-1].T).sum() + (-1/beta)*(out_H[-1].T).sum()

def free_energy_Gibbs(D_s,beta):
        '''
        Implementation of the free energy function directly given the cost matrices (assuming optimal associations).
        input: D_s: a list of K numpy arrays corrosponding to distances between stages
     

        output: out_c: K+1 numpy arrays with shape[1]=1, indicating the total cost of nodes or the cost of first node which
        gives you the free energy.
        '''
        K=len(D_s)
        D=D_s[::-1]
        out_D=[0]*(K+1)
        out_D[0]=np.array([0.0]).reshape(-1,1)
        out=[0]*(K+1)
        out[0]=np.array([1.0]).reshape(-1,1)
        for i in range(1,K+1):

            out_D[i]=(D[i-1]+np.repeat(np.transpose(out_D[i-1]),D[i-1].shape[0],axis=0))

            m=out_D[i].min(axis=1,keepdims=True)
            exp_D=np.exp(np.multiply(-beta,D[i-1]))
            out[i]=np.sum(np.multiply(exp_D,np.tile(out[i-1], (1,D[i-1].shape[0])).T),axis=1,keepdims=True)
            out_D[i]=m
        if np.isclose(out[-1],0.0).all():
            return m.sum()
        else:
            return (-1/beta*np.log(out[-1]).sum())


def calc_routs(P_ss):
        """ Given the associations for each drone, return the routs (facility ids) for each. """
        O=[]
        for i in range(len(P_ss)):
          m=0
          o=[]
          for p in P_ss[i]:
              m=np.argmax(p[m,:])
              o.append(m)
          o.pop()
          O.append(o)
        return O

def print_routs(routs , N_stations):
        print("")
        for i,o in enumerate(routs):
          print(f'\nDrone {i+1} --->', end='')
          for j in o:
            if j < N_stations:
              print(f'f{j+1} --->', end='')
            else:
              print(f'[D{i+1}]', end='')
              break