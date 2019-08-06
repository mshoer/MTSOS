
import numpy as np
import matplotlib.pyplot as plt
from pycubicspline import *
from numpy.linalg import norm
import cvxpy as cv
from cvxpy import Minimize, power
# from scipy.integrate import solve_ivp, odeint
from numpy.linalg import inv
# from scipy.io import loadmat
import random


# In[22]:


def make_path(waypoints):
    x = waypoints[0]
    y = waypoints[1]
    num_wpts = np.size(x) 

    alpha = 2
    beta = 100

    L = round(num_wpts/alpha)    
    idx = np.linspace(0,num_wpts-1,L, dtype=int)

    X = []
    Y = []
    for i in idx:
        X.append(x[i])
        Y.append(y[i])
#         
    sp = Spline2D(X,Y)
    # s = np.linspace(0, sp.s[-1],num=(test_length+1), endpoint=False)
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
        # if j < S_length-1:
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
    # plt.plot(X, Y, "oy", label="discretized")
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
    path.length = test_length
    path.dtheta = dtheta
    path.theta = theta
    path.theta_mid = theta_mid
    
    return path


# In[3]:


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



# In[20]:


#Create Waypoints

# sharp 90 degree turn
def sharp_corner():
    x = np.linspace(0,50,num=50)
    x2 = np.ones(50)*50
    x = np.concatenate((x,x2),axis=0)
    y = np.zeros(50)
    y2 = np.linspace(0,10,num=50)
    y = np.concatenate((y,y2),axis=0)
    q = x,y
    return q

# ellipse
def ellipse():
    t = np.linspace(0, 2*np.pi, num=100)
    x = 10*np.cos(t)
    y = 5*np.sin(t)
    q = x,y
    return q

def wave():
    t = np.linspace(0, 6*np.pi, num=200)
#     x = 10*np.cos(t)
    y = 5*np.sin(t+np.pi/2)
    q = t,y
    return q


# output 
def layout(ind):  
    if ind == 1:
        q1 = sharp_corner()
        path = make_path(q1)
    elif ind == 0:
        q2 = ellipse()
        path = make_path(q2)
    else:
        q3 = wave()
        path = make_path(q3)
            
    return path


def friction_circle(C1):
    t = np.linspace(0, 2*np.pi, num=100)
    x = C1*np.cos(t)
    y = C1*np.sin(t)
    q = x,y
    return q



# CONSTANTS
m = 10 # vehicle mass [kg]
Wf = .5 # percent of weight on front tires 
ms = .1 # coefficient of friction between tires and ground
g = 9.81 # gravity
u_max = ms*m*g # max allowed force
L = .28 # vehicle length [m]
w = .15 # vehicle width [m]

class parameters:
    pass

params = parameters()

params.kappa = 0
params.epsilon = .01
params.alpha = 0
params.beta = 0
params.initial_velocity = 0
params.variables = m, u_max, Wf, ms, L, w
params.max_iterations = 1000
params.U_size = 2
params.State_size = 2

def optimize(path):
    
    test_length = path.length
    dtheta = path.dtheta
    S = path.S
    S_prime = path.S_prime
    S_dprime = path.S_dprime
    theta = path.theta
    theta_mid = path.theta_mid
    
    C1 = params.variables[1] # U_max
    C2 = params.variables[1]*params.variables[2] # F_long_max
#     print('C1 = ',C1)
#     print('C2 = ',C2)
    
    theta_mid = path.theta_mid

    A = cv.Variable((test_length))
    B = cv.Variable((test_length))
    U = cv.Variable((2,test_length))

    cost = 0
    constr = []
    
    j = 0
    for i in theta_mid:
        # if j < test_length:
        cost += dtheta*(power(B[j+1],-.5) + power(B[j],-.5))
        R, M, C, d = dynamics(S_prime[i], S_dprime[i], params.variables)   
        constr += [B[0] == 0 , A[0] == 0, U[0,0] == 0, U[1,0] == 0]
        constr += [R*U[:,j+1] == M*A[j+1] + C*((B[j+1] + B[j])/2) + d]
        constr += [B[j+1] >= 0]
        constr += [cv.norm(U[:,j+1],2)<= C1]
        constr += [U[0,j+1] <= C2]
        constr += [B[j+1] == B[j]+2*A[j+1]*dtheta]
        j += 1
        
    problem = cv.Problem(Minimize(2*cost), constr)
    solution = problem.solve(solver=cv.ECOS, max_iters=200)
    
    return B, A, U, C1, C2


def simulate(order):
    
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
    
    if order == 4:
        for i in range(test_length-1): 
            u[i] = np.matrix(uu[:,i+1]).T
            # Runge-Kutta 4th Oder 
            RR, MM, CC, dd = dynamics_sim(vz1, 0, params.variables)
            k1v = inv(MM)*(RR*u[i])*dt[i]
            k1p = v1*dt[i]

            RR, MM, CC, dd = dynamics_sim(vz1+k1v/2, 0, params.variables)
            k2v = inv(MM)*(RR*u[i])*dt[i]
            k2p = (v1+k1v/2)*dt[i]

            RR, MM, CC, dd = dynamics_sim(vz1+k2v/2, 0, params.variables)
            k3v = inv(MM)*(RR*u[i])*dt[i]
            k3p = (v1+k2v/2)*dt[i]

            RR, MM, CC, dd = dynamics_sim(vz1+k3v, 0, params.variables)
            k4v = inv(MM)*(RR*u[i])*dt[i]
            k4p = (v1+k3v)*dt[i]

            x1 = x1+k1p/6+(k2p+k3p)/3+k4p/6
            v1 = v1+k1v/6+(k2v+k3v)/3+k4v/6
            vz1 = v1

            X[i+1] = x1
            V[i+1] = np.linalg.norm(v1)
    else:
        for i in range(test_length-1): 
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
    X = np.reshape(X,(test_length,2))
    V = list(V.values()) 
    
    return X, V
        

path = layout(1)

test_length = path.length
dtheta = path.dtheta
S = path.S
S_prime = path.S_prime
theta = path.theta
theta_mid = path.theta_mid

B, A, U, C1, C2 = optimize(path)

b = B.value
a = A.value
uu = U.value

b = abs(b)

S = np.array(list(path.S.values()))
S_prime = np.array(list(path.S_prime.values()))
S_dprime = np.array(list(path.S_dprime.values()))

b_s = np.sqrt(b)
dt = 2*dtheta/(b_s[0:test_length-1]+b_s[1:test_length])

T = np.sum(dt)
print('Total Time = ', T)

t = {}
time = 0
t[0] = 0
for i in range(len(dt)):
    time = time + dt[i]
    t[i+1] = time
t = np.array(list(t.values()))

q_dot0 = (S_prime[0]*b_s[0])
q0 = S[0]

# Simulate Contol Inputs
X4, V4 = simulate(4) # Runge-Kutta 4th Order
X, V = simulate(6)   # Runge-Kutta 6th Order

# RK4 Position Error
x_error_RK4 = abs(X4[:,0]-S[:,0])
y_error_RK4 = abs(X4[:,1]-S[:,1])

# RK6 Position Error
x_error_RK6 = abs(X[:,0]-S[:,0])
y_error_RK6 = abs(X[:,1]-S[:,1])


flg, ax = plt.subplots(1)
plt.plot(S[:,0], S[:,1], '-b.', label="Original Path")
plt.plot(X[:,0],X[:,1],'r.', label="Simulated")
plt.plot(S[0,0], S[0,1], 'yo')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Global Position')
plt.legend()
plt.show()

c1 = friction_circle(C1)

flg, ax = plt.subplots(1)
u_long = uu[0,1::]
u_lat = uu[1,1::]

u_max = np.amax(u_lat[:])
plt.plot(u_long[:test_length-1],u_lat[:test_length-1],'b.',label='Control Inputs')
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

flg, ax = plt.subplots(1)
plt.plot(x_error_RK4,'-r.', label='RK4')
plt.plot(x_error_RK6,'-b.', label='RK6')
plt.title('Position Error vs. Time')
plt.xlabel('Time [s]')
plt.ylabel('Error')
plt.legend()
plt.show()

flg, ax = plt.subplots(1)
plt.plot(t,X[:,0],'-r.', label='RK6')
plt.plot(t,S[:,0],'-b.', label='Desired')
plt.title('X Position vs. Time')
plt.xlabel('Time [s]')
plt.ylabel('X Position')
plt.legend()


flg, ax = plt.subplots(1)
plt.plot(t,X[:,1],'-r.', label='RK6')
plt.plot(t,S[:,1],'-b.', label='Desired')
plt.title('Y Position vs. Time')
plt.xlabel('Time [s]')
plt.ylabel('Y Position')
plt.legend()

plt.show()

print('DONE')
    

