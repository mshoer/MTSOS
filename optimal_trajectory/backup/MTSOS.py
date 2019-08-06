
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from pycubicspline import *
import cvxpy as cv
from cvxpy import Minimize, power
from numpy.linalg import inv
import random
import time
from numpy import loadtxt


# In[2]:


# Vehicle and Problem Parameters

m = 10 # vehicle mass [kg]
Wf = .5 # percent of weight on front tires 
ms = .8 # coefficient of friction between tires and ground
g = 9.81 # gravity
u_max = ms*m*g # max allowed force
L = .28 # vehicle length [m]
w = .15 # vehicle width [m]
V0 = 0 # initial speed
s_size = 2 # number of states
u_size = 2 # number of control inputs

class params:
    pass

params.variables = m, u_max, Wf, ms, L, w
params.s_size = s_size
params.u_size = u_size
params.initial_velocity = V0
params.C1 = u_max
params.C2 = u_max*Wf


# In[19]:


def Path(track, alpha, beta):
    
    q = layout(track)
        
    x = q[0]
    y = q[1]
    
    num_wpts = np.size(x) 
    upsample = round(alpha*num_wpts)

    L = np.ceil(num_wpts/upsample)
    
    idx = np.linspace(0,num_wpts-1,L+1, dtype=int)

    X = []
    Y = []
    for i in idx:
        X.append(x[i])
        Y.append(y[i])

    sp = Spline2D(X,Y)
    s = np.linspace(0, sp.s[-1],num=(beta), endpoint=False)
    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))
    
    test_length = np.size(s)
    
    S = {}
    theta = np.linspace(0,len(rx),len(rx))/(len(rx))
    theta_mid = (theta[0:len(theta)-1] + theta[1:])/2
    
    i = 0
    for t in theta:
        S[t] = np.asarray([rx[i],ry[i]])
        i += 1
        
    S_length = len(S)
    
    dtheta = 1/(test_length-1)
    
    S_middle = {}
    S_prime = {}
    S_dprime= {}
    
    j = 0
    for i in theta_mid:
        if j < S_length-1:
            S_middle[theta_mid[j]] = (S[theta[j+1]] + S[theta[j]])/2
            S_prime[theta_mid[j]] = (S[theta[j+1]]-S[theta[j]])/dtheta
            if j == S_length-1:
                S_dprime[theta_mid[j]] = (S[theta[j]]/2-S[theta[j+1]]+S[theta[j+2]]/2)/(dtheta*dtheta)
            elif S_length == 2:
                S_dprime[step] = 0
            elif j==0:
                S_middle[theta_mid[j]] = (S[theta[j+1]] + S[theta[j]])/2
                S_dprime[theta_mid[j]] = ((S[theta[j]]/2-S[theta[j+1]]+S[theta[j+2]]/2)/(dtheta*dtheta))
            elif j==S_length-2:
                S_dprime[theta_mid[j]] = ((S[theta[j-1]]/2-S[theta[j]]+S[theta[j+1]]/2)/(dtheta*dtheta))
            elif j == 1 or j==S_length-3:
                S_dprime[theta_mid[j]] = ((S[theta[j-1]]-S[theta[j]]-S[theta[j+1]]+S[theta[j+2]])/(2*dtheta*dtheta))
            else:
                S_dprime[theta_mid[j]] = ((-S[theta[j-2]]*5/48+S[theta[j-1]]*13/16-S[theta[j]]*17/24-S[theta[j+1]]*17/24+S[theta[j+2]]*13/16-S[theta[j+3]]*5/48)/(dtheta*dtheta))
        j += 1

        
    flg, ax = plt.subplots(1)
    plt.plot(x, y, "xb", label="waypoints")
    plt.plot(X, Y, "oy", label="discretized")
    plt.plot(rx, ry, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()
    plt.title('Generate Path from Waypoints')
    plt.show()
    
    
    class path:
        pass
    path = path()

    path.S = S
    path.S_middle = S_middle
    path.S_prime = S_prime
    path.S_dprime = S_dprime
    path.test_length = test_length
    path.dtheta = dtheta
    path.theta = theta
    path.theta_mid = theta_mid
    path.alpha = alpha
    path.beta = beta
    
    return path


# In[4]:


#Create Waypoints

def corner():
    x = np.linspace(0,50,num=50)
    x2 = np.ones(50)*50
    x = np.concatenate((x,x2),axis=0)
    y = np.zeros(50)
    y2 = np.linspace(0,10,num=50)
    y = np.concatenate((y,y2),axis=0)
    q = x,y
    return q

def ellipse():
    t = np.linspace(0, 2*np.pi, num=100)
    x = 10*np.cos(t)
    y = 5*np.sin(t)
    q = x,y
    return q

def wave():
    t = np.linspace(0, 6*np.pi, num=200)
    y = 5*np.sin(t+np.pi/2)
    q = t,y
    return q

def qph():
    q = loadtxt("qph.txt", comments="#", delimiter=",", unpack=False)
    return q

def layout(track):  
    if track == 'corner':
        q = corner()
    elif track == 'ellipse':
        q = ellipse()
    elif track == 'wave':
        q = wave() 
    elif track == 'qph':
        q = qph()
    return q

def friction_circle(C1):
    t = np.linspace(0, 2*np.pi, num=100)
    x = C1*np.cos(t)
    y = C1*np.sin(t)
    q = x,y
    return q


# In[5]:


def simulate(B,A,U):
    
    b = B.value
    a = A.value
    uu = U.value

    b = abs(b)

    S = np.array(list(path.S.values()))
    S_prime = np.array(list(path.S_prime.values()))
    S_dprime = np.array(list(path.S_dprime.values()))
    
    b_s = np.sqrt(b)
    dt = 2*path.dtheta/(b_s[0:path.test_length-1]+b_s[1:path.test_length])
    
    t = {}
    time = 0
    t[0] = 0
    for i in range(len(dt)):
        time = time + dt[i]
        t[i+1] = time
    t = np.array(list(t.values()))
    
    q_dot0 = (S_prime[0]*b_s[0])
    q0 = S[0]
    
    u = {}
    X = {}
    V = {}

    # Initial Guess
    x1 = np.matrix(q0).T  
    v1 = 0*np.matrix(q_dot0).T
    vz1 = np.matrix(S_prime[0]).T

    X[0] = x1
    V[0] = np.linalg.norm(v1)
    
    
    for i in range(path.test_length-1): 
        u[i] = np.matrix(uu[:,i+1]).T

        RR, MM, CC, dd = dynamics_sim(vz1, 0, params.variables)
        k1v = inv(MM)*(RR*u[i])*dt[i]
        k1p = v1*dt[i]

        RR, MM, CC, dd = dynamics_sim(vz1+k1v/4, 0, params.variables)
        k2v = inv(MM)*(RR*u[i])*dt[i]
        k2p = (v1+k1v/4)*dt[i]

        RR, MM, CC, dd = dynamics_sim(vz1+3/32*k1v+9/32*k2v, 0, params.variables)
        k3v = inv(MM)*(RR*u[i])*dt[i]
        k3p = (v1+3/32*k1v+9/32*k2v)*dt[i]

        RR, MM, CC, dd = dynamics_sim(vz1+1932/2197*k1v-7200/2197*k2v+7296/2197*k3v, 0, params.variables)
        k4v = inv(MM)*(RR*u[i])*dt[i]
        k4p = (v1+1932/2197*k1v-7200/2197*k2v+7296/2197*k3v)*dt[i]

        RR, MM, CC, dd = dynamics_sim(vz1+439/216*k1v-8*k2v+3680/513*k3v-845/4104*k4v, 0, params.variables)
        k5v = inv(MM)*(RR*u[i])*dt[i]
        k5p = (v1+439/216*k1v-8*k2v+3680/513*k3v-845/4104*k4v)*dt[i]

        RR, MM, CC, dd = dynamics_sim(vz1-8/27*k1v+2*k2v-3544/2565*k3v+1859/4104*k4v-11/40*k5v, 0, params.variables)
        k6v = inv(MM)*(RR*u[i])*dt[i]
        k6p = (v1-8/27*k1v+2*k2v-3544/2565*k3v+1859/4104*k4v-11/40*k5v)*dt[i]

        x1 = x1 + 16/135*k1p + 6656/12825*k3p + 28561/56430*k4p - 9/50*k5p + 2/55*k6p
        v1 = v1 + 16/135*k1v + 6656/12825*k3v + 28561/56430*k4v - 9/50*k5v + 2/55*k6v
        vz1 = v1

        X[i+1] = x1
        V[i+1] = np.linalg.norm(v1)

        
    X = list(X.values()) 
    X = np.reshape(X,(path.test_length,2))
    V = list(V.values()) 
    
    x_error_RK6 = X[:,0]-S[:,0]
    y_error_RK6 = X[:,1]-S[:,1]

    flg, ax = plt.subplots(1)
    plt.plot(S[:,0], S[:,1], '-b.', label="Original Path")
    plt.plot(X[:,0],X[:,1],'r.', label="Simulated")
    plt.plot(S[0,0], S[0,1], 'yo')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Global Position')
    plt.legend()
    plt.show()

    c1 = friction_circle(params.C1)

    flg, ax = plt.subplots(1)
    u_long = uu[0,1::]
    u_lat = uu[1,1::]

    u_max = np.amax(u_lat[:])
    plt.plot(u_long[:path.test_length-1],u_lat[:path.test_length-1],'b.',label='Control Inputs')
    plt.plot(c1[0],c1[1],'r-',label='Friction Circle')
    plt.xlabel('F_long')
    plt.ylabel('F_lat')
    plt.title('Control Inputs on Friction Circle')
    plt.legend()
    plt.show()

    flg, ax = plt.subplots(1)
    plt.plot(t,V,'-r.')
    plt.title('Speed vs. Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Speed')
    plt.show()
    
    return
        

    


# In[6]:


def dynamics(S_prime, S_dprime, variables):
    
    m = variables[0]
    
    phi = np.arctan2(S_prime[0],S_prime[1])
    
    R = np.zeros((2,2))
    R[0,0] = np.cos(phi)
    R[0,1] = -np.sin(phi)
    R[1,0] = np.sin(phi)
    R[1,1] = np.cos(phi)

    M = np.zeros((2,2))
    M[0,0] = m
    M[1,1] = m
    
    C = np.zeros((2,2))
    d = np.zeros((2))
    
#     S_prime = np.matrix(S_prime).T
#     S_dprime = np.matrix(S_dprime).T
    
#     M = np.matmul(M,S_prime)
#     C = np.matmul(M,S_dprime) + np.matmul(C,S_prime**2)
    
    C = m*S_dprime 
    M = m*S_prime
    
    return R, M, C, d

def dynamics_sim(S_prime, S_dprime, variables):

    m = variables[0]
    phi = np.arctan2(S_prime[0],S_prime[1])
    
    R = np.zeros((2,2))
    R[0,0] = np.cos(phi)
    R[0,1] = -np.sin(phi)
    R[1,0] = np.sin(phi)
    R[1,1] = np.cos(phi)

    M = np.zeros((2,2))
    M[0,0] = m
    M[1,1] = m
    
    C = np.zeros((2,2))
    d = np.zeros((2))

    
    return R, M, C, d



# In[14]:


def optimize(path, params, viz):
    
    S = path.S
    S_prime = path.S_prime
    S_dprime = path.S_dprime
    
    A = cv.Variable((path.test_length))
    B = cv.Variable((path.test_length))
    U = cv.Variable((2,path.test_length))

    cost = 0
    constr = []
    
    j = 0
    for i in path.theta_mid:
        if j < path.test_length:
            cost += path.dtheta*(power(B[j+1],-.5) + power(B[j],-.5))
            R, M, C, d = dynamics(S_prime[i], S_dprime[i], params.variables)   
            constr += [B[0] == 0 , A[0] == 0, U[0,0] == 0, U[1,0] == 0]
            constr += [R*U[:,j+1] == M*A[j+1] + C*((B[j+1] + B[j])/2) + d]
            constr += [B[j+1] >= 0]
            constr += [cv.norm(U[:,j+1],2)<= params.C1]
            constr += [U[0,j+1] <= params.C2]
            constr += [B[j+1] == B[j]+2*A[j+1]*path.dtheta]
        j += 1
        
    problem = cv.Problem(Minimize(2*cost), constr)
    solution = problem.solve(solver=cv.ECOS, max_iters=400)
    
    if viz == True:
        simulate(B, A, U)
        
#     B = abs(B.value)
#     A = A.value
#     U = U.value
    
    return B, A, U


# In[24]:


path = Path('corner',.03, 100)
b,a,u = optimize(path,params,True)

