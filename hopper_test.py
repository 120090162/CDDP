import numpy as np

import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")
warnings.filterwarnings("ignore", message="UserWarning: Singular Jacobian matrix. Using SVD decomposition to perform the factorizations.")


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

from systems import DynamicalSystem

class hopper(DynamicalSystem):
    def __init__(self):
        super().__init__(7, 2)
        # self.dt = 0.01
        self.dt = 0.05
        self.control_bound = np.ones(2) * 10 # [100, 100]
        self.goal = np.zeros(4)

        self.mass = 1.0
        self.inertia = 0.1
        self.leg_len_max = 1
        self.leg_len_min = 0.25
        self.gravity = 9.81
        self.friction = 0.5
        
        self.M = np.zeros((3,3))
        self.M[0:2,0:2] = self.mass * np.eye(2)
        self.M[2,2] = self.inertia
        
        self.h = np.array([0,self.mass*self.gravity,0]).T
        
    def foot_J(self,x):
        return np.array([[1,0,x[6]*np.cos(x[2])],
                      [0,1,x[6]*np.sin(x[2])]])
    def foot_J_x(self,x):
        result = np.zeros((self.state_size,2,3))
        result[3,:,:] = np.array([[0,0,-x[6]*np.sin(x[2])],
                      [0,0,x[6]*np.cos(x[2])]])
        # result[6,:,:] = np.array([[0,0,np.cos(x[2])],
        #               [0,0,np.sin(x[2])]])
        return result
    
    def contact_dynamics(self, x, u):
        foot_J = self.foot_J(x)
        M1 = np.linalg.inv(foot_J @ np.linalg.inv(self.M) @ foot_J.T)
        c = foot_J @ np.linalg.inv(self.M) @ ((self.input_B(x) @ u - self.h) * self.dt + self.M @ x[3:6])
        contact_v = lambda force: c + np.linalg.inv(M1) @ force
        obj = lambda force: contact_v(force).T @ M1 @ contact_v(force)
        def constraint(force):
            v = contact_v(force)
            return [v[1],
                    force[1],
                    self.friction * force[1] - np.abs(force[0]),
                    v[1]*force[1]]
        
        cons = (LinearConstraint(np.linalg.inv(M1)[1],lb=-c[1]),
                LinearConstraint(np.array([0,1]),lb=0),
                LinearConstraint(np.array([[1, self.friction],[-1, self.friction]]),lb=np.zeros(2)),
                NonlinearConstraint(lambda force: contact_v(force)[1]*force[1],lb=0,ub=0))
        result = minimize(obj,np.array([0,10]),method='trust-constr',constraints=cons,options={'verbose':0})
        if not result.success:
            raise 'contact dynamics fails'
        return (contact_v(result.x),result.x)
    
    def contact_dynamics_relaxed(self, x, u, rho):
        assert rho >= 0
        foot_J = self.foot_J(x)
        M1 = np.linalg.inv(foot_J @ np.linalg.inv(self.M) @ foot_J.T)
        c = foot_J @ np.linalg.inv(self.M) @ ((self.input_B(x) @ u - self.h) * self.dt + self.M @ x[3:6])
        contact_v = lambda force: c + np.linalg.inv(M1) @ force
        obj = lambda force: contact_v(force).T @ M1 @ contact_v(force)
        def constraint(force):
            v = contact_v(force)
            return [v[1],
                    force[1],
                    self.friction * force[1] - np.abs(force[0]),
                    v[1]*force[1]-rho]
        
        cons = (LinearConstraint(np.linalg.inv(M1)[1],lb=-c[1]),
                LinearConstraint(np.array([0,1]),lb=0),
                LinearConstraint(np.array([[1, self.friction],[-1, self.friction]]),lb=np.zeros(2)),
                NonlinearConstraint(lambda force: contact_v(force)[1]*force[1]-rho,lb=0,ub=0))
        result = minimize(obj,np.array([0,10]),method='trust-constr',constraints=cons,options={'verbose':0})
        if not result.success:
            raise 'relaxed contact dynamics fails'
        return (contact_v(result.x),result.x)
        
    def input_B(self, x):
        return np.array([[-np.sin(x[2]),0],
                        [np.cos(x[2]),0],
                        [0,1]])
    
    def transition(self, x, u):
        end = np.array([x[0]+x[6]*np.sin(x[2]), x[1]-x[6] * np.cos(x[2])])
        if np.isclose(end[1], 0): # contact ?
            contact_v,contact_f = self.contact_dynamics(x,u)
            foot_J = self.foot_J(x)
            result = np.zeros(7)
            
            result[3:6] = x[3:6] + np.linalg.inv(self.M) @ ((self.input_B(x) @ u - self.h) * self.dt + foot_J.T @ contact_f)
            if np.isclose(contact_v[1],0):
                result[4] = -x[4]
            result[0:3] = x[0:3] + self.dt * result[3:6]
            if result[1] < 0:
                result[1] = np.max([-result[1],x[1]])
                # result[4] = -result[4]
                
            if np.isclose(contact_v[1],0):
                end = end + contact_v * self.dt
                result[6] = np.clip(np.sqrt((result[0]-end[0])**2 + (result[1]-end[1])**2),0,self.leg_len_max)
                if result[6] < self.leg_len_min:
                    result[6] = self.leg_len_min
                    result[0] = end[0] - result[6]*np.sin(result[2])
                    result[1] = end[1] + result[6]*np.cos(result[2])
            # elif not np.isclose(contact_f[1],0):
                    
            else:
                result[6] = np.clip(np.sqrt((result[0]-end[0])**2 + (result[1]-end[1])**2),self.leg_len_min,self.leg_len_max)
                # result[4] = -result[4]
           
        else:
            u[0] = 0
            result = np.zeros(7)
            # self.M @ (xx-x) = (-self.h+self.input_B(x) @ u)*self.dt + foot_J.T.dot(contact_f)
            result[3:6] = x[3:6] + np.linalg.inv(self.M) @ (self.input_B(x) @ u - self.h) * self.dt
            result[0:3] = x[0:3] + self.dt * result[3:6]
            result[6] = x[6]
            
            # if result[1] < 0 and end_new[1] < 0:
            #     raise 'Unpredictable contact'
            if result[1] < 0:
                result[1] = -result[1]
                result[4] = -result[4]
                # result[4] = 0
            end_new = np.array([result[0]+result[6]*np.sin(result[2]), result[1]-result[6] * np.cos(result[2])])
            
            if end_new[1] < 0:
                result[1] -= end_new[1]
        return result
    def transition_J(self, x, u):
        #return matrix A, B, so that x = Ax + Bu
        A = np.zeros((self.state_size, self.state_size))
        B = np.zeros((self.state_size, self.control_size))
        end = np.array([x[0]+x[6]*np.sin(x[2]), x[1]-x[6] * np.cos(x[2])])
        
        # q
        A[0:3, 0:3] = np.identity(3)
        A[0:3, 3:6] = np.identity(3) * self.dt
        B[0:3, :] = np.linalg.inv(self.M) @ self.input_B(x) * self.dt ** 2
        # dot q
        A[3:6, 3:6] = np.identity(3)
        A[3:6, 0:3] = 0
        B[3:6, :] = np.linalg.inv(self.M) @ self.input_B(x) * self.dt
            
        if np.isclose(end[1], 0): # contact ?
        # if False:
            contact_v,contact_f = self.contact_dynamics(x,u)
            foot_J = self.foot_J(x)
            foot_J_x = self.foot_J_x(x) # 7,2,3
            tensor_invM = np.linalg.inv(self.M)
            tensor_invM = tensor_invM[np.newaxis,:] # 1,3,3
            tensor_contact_f=contact_f.reshape((-1,1))[np.newaxis,:] # 1,2,1
            tensor_J = foot_J[np.newaxis,:] # 1,2,3
            qdot = (np.linalg.inv(self.M)@(self.input_B(x)@u-self.h)*self.dt+x[3:6]).reshape((-1,1)) # 3,1
            tensor_qdot = qdot[np.newaxis,:] # 1,3,1
            tensor_qdot_x = A[3:6,:].T.reshape((self.state_size,3,1)) # 7,3,1
            tensor_qdot_u = B[3:6,:].T.reshape((self.control_size,3,1)) # 2,3,1
            
            lambda_x = np.zeros((self.state_size,2,1))
            lambda_u = np.zeros((self.control_size,2,1))
            if np.isclose(contact_f[1]*self.friction-np.abs(contact_f[0]), 0): # slide ?
                Jt = foot_J[0,:]
                Jn = foot_J[1,:] # 1,3
                Jt_x = foot_J_x[:,[0],:] # 7,1,3
                Jn_x = foot_J_x[:,[1],:]
                Jtn = (Jn+Jt*self.friction*np.sign(contact_v[0])) # 1,3
                Jtn = Jtn[np.newaxis,:]
                Jtn_x = (Jn_x+Jt_x*self.friction*np.sign(contact_v[0])) # 7,1,3
                tensor_Jt = tensor_J[:,[0],:] # 1,1,3
                tensor_Jn = tensor_J[:,[1],:]
                tensor_Jtn = Jtn[np.newaxis,:] # 1,1,3
                
                Acc = Jn@np.linalg.inv(self.M)@Jtn.T
                bcc = Jn@qdot
                
                Acc_x = Jn_x@tensor_invM@tensor_Jtn.transpose((0,2,1))+tensor_Jn@tensor_invM@Jtn_x.transpose((0,2,1)) # 7,1,1
                
                bcc_x = (Jn_x@tensor_qdot+tensor_Jn@tensor_qdot_x) # 7,1,1
                bcc_u = (tensor_Jn@tensor_qdot_u) # 2,1,1
                                
                invAcc = 1/Acc # 1
                
                lambda_x[:,0,:] = -(invAcc*(Acc_x.squeeze(2)*contact_f[1]+bcc_x.squeeze(2))) # 7,1
                lambda_u[:,0,:] = -(invAcc*bcc_u.squeeze(2)) # 2,1
                
                lambda_x[:,1,:] = self.friction*np.sign(contact_v[0])*lambda_x[:,0,:]
                lambda_u[:,1,:] = self.friction*np.sign(contact_v[0])*lambda_u[:,0,:]
                
            else: # slide很少有
                Jcc = foot_J # 2,3
                tensor_Jcc = tensor_J # 1,2,3
                Acc_x = foot_J_x@tensor_invM@tensor_Jcc.transpose((0,2,1))+tensor_Jcc@tensor_invM@foot_J_x.transpose((0,2,1)) # 7,2,2
                
                
                bcc_x = (foot_J_x@tensor_qdot+tensor_Jcc@tensor_qdot_x) # 7,2,1
                bcc_u = (tensor_Jcc@tensor_qdot_u) # 2,2,1
                
                invAcc = np.linalg.inv(Jcc@np.linalg.inv(self.M)@Jcc.T) # 2,2
                bcc = Jcc@qdot # 2,1
                tensor_Acc=invAcc[np.newaxis,:]
                tensor_bcc=bcc[np.newaxis,:]
                
                lambda_x = -(tensor_Acc@(Acc_x@tensor_contact_f+bcc_x)) # 7,2,1
                lambda_u = -(tensor_Acc@bcc_u) # 2,2,1
            
            Aincr = (tensor_invM@(foot_J_x.transpose((0,2,1))@tensor_contact_f+tensor_J.transpose((0,2,1))@lambda_x)).squeeze().T # 3,7
            Bincr = (tensor_invM@(tensor_J.transpose((0,2,1))@lambda_u)).squeeze().T # 3,2
            A[0:3,:] += Aincr * self.dt
            A[3:6,:] += Aincr
            B[0:3,:] += Bincr * self.dt
            B[3:6,:] += Bincr
            # B[:,0] = 0
            
        else:
            # no contact
            B[:,0] = 0
        return A, B
    def transition_J_relaxed(self,x,u,rho):
        # return self.transition_J(x,u)
        #return matrix A, B, so that x = Ax + Bu
        A = np.zeros((self.state_size, self.state_size))
        B = np.zeros((self.state_size, self.control_size))
        end = np.array([x[0]+x[6]*np.sin(x[2]), x[1]-x[6] * np.cos(x[2])])
        
        # q
        A[0:3, 0:3] = np.identity(3)
        A[0:3, 3:6] = np.identity(3) * self.dt
        B[0:3, :] = np.linalg.inv(self.M) @ self.input_B(x) * self.dt ** 2
        # dot q
        A[3:6, 3:6] = np.identity(3)
        A[3:6, 0:3] = 0
        B[3:6, :] = np.linalg.inv(self.M) @ self.input_B(x) * self.dt
            
        if np.isclose(end[1], 0): # contact ?
        # if False:
            contact_v,contact_f = self.contact_dynamics_relaxed(x,u,rho)
            foot_J = self.foot_J(x)
            foot_J_x = self.foot_J_x(x) # 7,2,3
            tensor_invM = np.linalg.inv(self.M)
            tensor_invM = tensor_invM[np.newaxis,:] # 1,3,3
            tensor_contact_f=contact_f.reshape((-1,1))[np.newaxis,:] # 1,2,1
            tensor_J = foot_J[np.newaxis,:] # 1,2,3
            qdot = (np.linalg.inv(self.M)@(self.input_B(x)@u-self.h)*self.dt+x[3:6]).reshape((-1,1)) # 3,1
            tensor_qdot = qdot[np.newaxis,:] # 1,3,1
            tensor_qdot_x = A[3:6,:].T.reshape((self.state_size,3,1)) # 7,3,1
            tensor_qdot_u = B[3:6,:].T.reshape((self.control_size,3,1)) # 2,3,1
            
            lambda_x = np.zeros((self.state_size,2,1))
            lambda_u = np.zeros((self.control_size,2,1))
            if np.isclose(contact_f[1]*self.friction-np.abs(contact_f[0]), 0): # slide ?
                Jt = foot_J[0,:]
                Jn = foot_J[1,:] # 1,3
                Jt_x = foot_J_x[:,[0],:] # 7,1,3
                Jn_x = foot_J_x[:,[1],:]
                Jtn = (Jn+Jt*self.friction*np.sign(contact_v[0])) # 1,3
                Jtn = Jtn[np.newaxis,:]
                Jtn_x = (Jn_x+Jt_x*self.friction*np.sign(contact_v[0])) # 7,1,3
                tensor_Jt = tensor_J[:,[0],:] # 1,1,3
                tensor_Jn = tensor_J[:,[1],:]
                tensor_Jtn = Jtn[np.newaxis,:] # 1,1,3
                
                Acc = Jn@np.linalg.inv(self.M)@Jtn.T
                bcc = Jn@qdot
                
                Acc_x = Jn_x@tensor_invM@tensor_Jtn.transpose((0,2,1))+tensor_Jn@tensor_invM@Jtn_x.transpose((0,2,1)) # 7,1,1
                
                bcc_x = (Jn_x@tensor_qdot+tensor_Jn@tensor_qdot_x) # 7,1,1
                bcc_u = (tensor_Jn@tensor_qdot_u) # 2,1,1
                                
                invAcc = 1/(Acc+rho/contact_f[1]**2) # 1
                
                lambda_x[:,0,:] = -(invAcc*(Acc_x.squeeze(2)*contact_f[1]+bcc_x.squeeze(2))) # 7,1
                lambda_u[:,0,:] = -(invAcc*bcc_u.squeeze(2)) # 2,1
                
                lambda_x[:,1,:] = self.friction*np.sign(contact_v[0])*lambda_x[:,0,:]
                lambda_u[:,1,:] = self.friction*np.sign(contact_v[0])*lambda_u[:,0,:]
                
            else: # slide很少有
                Jcc = foot_J # 2,3
                tensor_Jcc = tensor_J # 1,2,3
                Acc_x = foot_J_x@tensor_invM@tensor_Jcc.transpose((0,2,1))+tensor_Jcc@tensor_invM@foot_J_x.transpose((0,2,1)) # 7,2,2
                
                
                bcc_x = (foot_J_x@tensor_qdot+tensor_Jcc@tensor_qdot_x) # 7,2,1
                bcc_u = (tensor_Jcc@tensor_qdot_u) # 2,2,1
                
                invAcc = np.linalg.inv(Jcc@np.linalg.inv(self.M)@Jcc.T+rho*np.diag(1/contact_f)) # 2,2
                bcc = Jcc@qdot # 2,1
                tensor_Acc=invAcc[np.newaxis,:]
                tensor_bcc=bcc[np.newaxis,:]
                
                lambda_x = -(tensor_Acc@(Acc_x@tensor_contact_f+bcc_x)) # 7,2,1
                lambda_u = -(tensor_Acc@bcc_u) # 2,2,1
            
            Aincr = (tensor_invM@(foot_J_x.transpose((0,2,1))@tensor_contact_f+tensor_J.transpose((0,2,1))@lambda_x)).squeeze().T # 3,7
            Bincr = (tensor_invM@(tensor_J.transpose((0,2,1))@lambda_u)).squeeze().T # 3,2
            A[0:3,:] += Aincr * self.dt
            A[3:6,:] += Aincr
            B[0:3,:] += Bincr * self.dt
            B[3:6,:] += Bincr
            # B[:,0] = 0
            
        else:
            # no contact
            B[:,0] = 0
        return A, B
    def draw_trajectories(self, traj):
        N = traj.shape[1]
        fig, ax = plt.subplots(1,1)
        def anim(i):
            ax.clear()
            # trace = slice(i-10,i)
            trace = slice(0,i)
            plt.plot(traj[0, trace], traj[1, trace],color='b')
            plt.plot(traj[0, trace]+traj[6,trace]*np.sin(traj[2,trace]), traj[1, trace]-traj[6,trace] * np.cos(traj[2,trace]),color='g')
            plt.arrow(traj[0, i]+traj[6,i]*np.sin(traj[2,i]), traj[1, i]-traj[6,i] * np.cos(traj[2,i]),
                    -traj[6,i]*np.sin(traj[2,i]), traj[6,i] * np.cos(traj[2,i]), color='k', width=traj[6,i]/50)
            ax.set_aspect("equal")
            ax.set_xlim(-5, 10)
            ax.set_ylim(-1, 4)
            plt.axhline(0,color='k')
        ani = FuncAnimation(fig,anim,frames=range(10,N),interval=100,repeat=False)
        
        plt.show()
        
    def draw_u_trajectories(self, u_trajectories):
        x = plt.subplot(111)
        plt.scatter(u_trajectories[0, 0::5], u_trajectories[1, 0::5], 4,color='r')
        plt.show()

if __name__ == '__main__':
    
    # print('Test for the contact force.......')
    # system = hopper()
    # x = np.array([0,0.9,0,0,0,0,0.9]).T
    # u = np.array([10,0])
    # contact = system.contact_dynamics(x,u)
    
    # u1 = np.linspace(1,10,20)
    # u2 = np.concat((np.linspace(-10,-2,10),np.linspace(2,10,10)),axis=0)
    # v = np.zeros((400,2))
    # f = np.zeros((400,2))
    # for i in range(20):
    #     for j in range(20):
    #         u = np.array([u1[i],u2[j]])
    #         contact = system.contact_dynamics(x,u)
    #         # print(contact)
    #         v[i*20+j,:],f[i*20+j,:] = contact[0],contact[1]
    
    # ax = plt.subplot(111)
    # plt.scatter(f[:, 0], f[:, 1], 4,color='r')
    # # plt.scatter(v[:,1],f[:,1],4,color='r')
    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(0, 1)
    # plt.show()
    
    # u=np.array([1,5])
    # print(system.contact_dynamics(x,u))
    
    # print('Test for the transition.......')
    # system = hopper()
    # x = np.array([0,1,0,0,0,0,0.8]).T
    # N=80
    # traj = np.zeros((7,N))
    # u = np.zeros((2,N))
    # u[0,:] = 10
    # t = np.arange(N) * system.dt
    # traj[:,0] = x
    # for i in range(1,N):
    #     traj[:,i] = system.transition(traj[:,i-1],u[:,i-1])
    
    # fig, ax = plt.subplots(1,1)
    # # plt.scatter(t, traj[1, :], 4,color='b')
    # # plt.scatter(t, traj[1, :]-traj[6,:] * np.cos(traj[2,:]), 4,color='g')
    # # plt.scatter(t, traj[6, :], 4,color='b')
    # def anim(i):
    #     ax.clear()
    #     plt.scatter(traj[0, i], traj[1, i], 20,color='b')
    #     plt.scatter(traj[0, i]+traj[6,i]*np.sin(traj[2,i]), traj[1, i]-traj[6,i] * np.cos(traj[2,i]), 20,color='g')
    #     plt.arrow(traj[0, i]+traj[6,i]*np.sin(traj[2,i]), traj[1, i]-traj[6,i] * np.cos(traj[2,i]),
    #               -traj[6,i]*np.sin(traj[2,i]), traj[6,i] * np.cos(traj[2,i]), color='k', width=traj[6,i]/100)
    #     ax.set_aspect("equal")
    #     ax.set_xlim(-2, 2)
    #     ax.set_ylim(-1, 3)
    # ani = FuncAnimation(fig,anim,frames=N,interval=10,repeat=False)
    # plt.show()
    
    # print('Test for the transition1.......')
    # system = hopper()
    # x = np.array([0,1,-0.5,0,0,0,0.9]).T
    # N=200
    # traj = np.zeros((7,N))
    # u = np.zeros((2,N))
    # u[0,:] = 10
    # u[1,:] = 0 # 磕头
    # # u[1,:] = 1.5
    # t = np.arange(N) * system.dt
    # traj[:,0] = x
    # for i in range(1,N):
    #     traj[:,i] = system.transition(traj[:,i-1],u[:,i-1])
    # system.draw_trajectories(traj)
    
    # plt.scatter(t, traj[1, :], 4,color='b')
    # plt.scatter(t, traj[1, :]-traj[6,:] * np.cos(traj[2,:]), 4,color='g')
    
    # plt.scatter(t, traj[6, :], 4,color='b')
    # ax.set_aspect("equal")
    # ax.set_xlim(-1, 5)
    # ax.set_ylim(-1, 2)
    
    # def anim(i):
    #     ax.clear()
    #     # trace = slice(i-10,i)
    #     trace = slice(0,i)
    #     plt.plot(traj[0, trace], traj[1, trace],color='b')
    #     plt.plot(traj[0, trace]+traj[6,trace]*np.sin(traj[2,trace]), traj[1, trace]-traj[6,trace] * np.cos(traj[2,trace]),color='g')
    #     plt.arrow(traj[0, i]+traj[6,i]*np.sin(traj[2,i]), traj[1, i]-traj[6,i] * np.cos(traj[2,i]),
    #               -traj[6,i]*np.sin(traj[2,i]), traj[6,i] * np.cos(traj[2,i]), color='k', width=traj[6,i]/50)
    #     ax.set_aspect("equal")
    #     ax.set_xlim(-1, 25)
    #     ax.set_ylim(-1, 2)
    #     plt.axhline(0,color='k')
    # ani = FuncAnimation(fig,anim,frames=range(10,N),interval=100,repeat=False)
    
    # plt.show()
    
    print('Test for the relaxed contact force.......')
    system = hopper()
    x = np.array([0,0.9,0,0,0,0,0.9]).T
    u = np.array([10,0])
    rho=0.1
    
    u1 = np.linspace(1,10,20)
    u2 = np.concat((np.linspace(-10,-2,10),np.linspace(2,10,10)),axis=0)
    v = np.zeros((400,2))
    f = np.zeros((400,2))
    for i in range(20):
        for j in range(20):
            u = np.array([u1[i],u2[j]])
            contact = system.contact_dynamics_relaxed(x,u,rho)
            # print(contact)
            v[i*20+j,:],f[i*20+j,:] = contact[0],contact[1]
    
    ax = plt.subplot(111)
    # plt.scatter(f[:, 0], f[:, 1], 4,color='r')
    plt.scatter(v[:,1],f[:,1],4,color='r')
    t = np.linspace(0.1,0.5,40)
    # plt.plot(t,rho/t,color='g')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1)
    plt.show()