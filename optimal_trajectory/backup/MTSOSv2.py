import numpy as np
import cvxpy as cv
from pycubicspline import *
from utils import generate_way_points
import matplotlib.pyplot as plt

# vehicle and problem parameters
m = 3       # vehicle mass [kg]
Wf = .5     # percent of weight on front tires 
ms = .8     # coefficient of friction between tires and ground
g = 9.81    # gravity
L = .28     # vehicle length [m]
w = .15     # vehicle width [m]
V0 = 0      # initial speed
s_size = 2  # number of states
u_size = 2  # number of control inputs


class define_params:

    def __init__(self, m, Wf, ms, L, w, s_size, V0):
        u_max = ms*m*g      # max allowed force
        self.variables = m, u_max, Wf, ms, L, w
        self.s_size = s_size
        self.u_size = u_size
        self.initial_velocity = V0
        self.C1 = u_max
        self.C2 = u_max*Wf


class define_path:
    def __init__(self, S, S_middle, S_prime, S_dprime, test_length, 
                    dtheta, theta, theta_mid, beta):
        self.S = S
        self.S_middle = S_middle
        self.S_prime = S_prime
        self.S_dprime = S_dprime
        self.test_length = test_length
        self.dtheta = dtheta
        self.theta = theta
        self.theta_mid = theta_mid
        self.beta = beta
        

def Path(track, beta, plot_results=True):
    
    x, y = generate_way_points(track)

    num_wpts = np.size(x)
    L = num_wpts-1

    # alpha = 0.03
    # upsample = round(alpha*num_wpts)
    # L = np.ceil(num_wpts/upsample)
    
    idx = np.linspace(0, num_wpts-1, L+1, dtype=int)

    X = []
    Y = []
    for i in idx:
        X.append(x[i])
        Y.append(y[i])

    sp = Spline2D(X,Y)
    s = np.linspace(0, sp.s[-1], num=(beta), endpoint=False)
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

    if plot_results:
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
    
    path = define_path(S, S_middle, S_prime, S_dprime, test_length, 
                    dtheta, theta, theta_mid, beta)
    
    return path


def friction_circle(C1):
    t = np.linspace(0, 2*np.pi, num=100)
    x = C1*np.cos(t)
    y = C1*np.sin(t)
    return x, y


# simulate control inputs
def simulate(B, A, U, params, plot_results=True):
    
    b = B
    a = A
    uu = U

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
        k1v = np.linalg.inv(MM)*(RR*u[i])*dt[i]
        k1p = v1*dt[i]

        RR, MM, CC, dd = dynamics_sim(vz1+k1v/4, 0, params.variables)
        k2v = np.linalg.inv(MM)*(RR*u[i])*dt[i]
        k2p = (v1+k1v/4)*dt[i]

        RR, MM, CC, dd = dynamics_sim(vz1+3/32*k1v+9/32*k2v, 0, params.variables)
        k3v = np.linalg.inv(MM)*(RR*u[i])*dt[i]
        k3p = (v1+3/32*k1v+9/32*k2v)*dt[i]

        RR, MM, CC, dd = dynamics_sim(vz1+1932/2197*k1v-7200/2197*k2v+7296/2197*k3v, 0, params.variables)
        k4v = np.linalg.inv(MM)*(RR*u[i])*dt[i]
        k4p = (v1+1932/2197*k1v-7200/2197*k2v+7296/2197*k3v)*dt[i]

        RR, MM, CC, dd = dynamics_sim(vz1+439/216*k1v-8*k2v+3680/513*k3v-845/4104*k4v, 0, params.variables)
        k5v = np.linalg.inv(MM)*(RR*u[i])*dt[i]
        k5p = (v1+439/216*k1v-8*k2v+3680/513*k3v-845/4104*k4v)*dt[i]

        RR, MM, CC, dd = dynamics_sim(vz1-8/27*k1v+2*k2v-3544/2565*k3v+1859/4104*k4v-11/40*k5v, 0, params.variables)
        k6v = np.linalg.inv(MM)*(RR*u[i])*dt[i]
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

    if plot_results:
        flg, ax = plt.subplots(1)
        plt.plot(S[:,0], S[:,1], '-b.', label="Original Path")
        plt.plot(X[:,0],X[:,1], 'r.', label="Simulated")
        plt.plot(S[0,0], S[0,1], 'yo')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Global Position')
        plt.legend()

        c1 = friction_circle(params.C1)

        flg, ax = plt.subplots(1)
        u_long = uu[0,1::]
        u_lat = uu[1,1::]

        u_max = np.amax(u_lat[:])
        plt.plot(u_long[:path.test_length-1],u_lat[:path.test_length-1], 'b.', label='Control Inputs')
        plt.plot(c1[0],c1[1], 'r-', label='Friction Circle')
        plt.xlabel('F_long')
        plt.ylabel('F_lat')
        plt.title('Control Inputs on Friction Circle')
        plt.legend()

        flg, ax = plt.subplots(1)
        plt.plot(t,V, '-r.')
        plt.title('Speed vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Speed')

        flg, ax = plt.subplots(1)
        plt.plot(t[1:], u_long, '-b.', label='Force long')
        plt.plot(t[1:], u_lat, '-r.', label='Force lat')
        plt.title('Force vs time')
        plt.xlabel('time [s]')
        plt.ylabel('force [N]')
        plt.legend()

        plt.show()
    
    return


# dynamics (convexified)
def dynamics(S_prime, S_dprime, variables):

    m = variables[0]
    R, M, C, d = dynamics_sim(S_prime, S_dprime, variables)
    C = m*S_dprime 
    M = m*S_prime
    return R, M, C, d


# dynamics (non-linear)
def dynamics_sim(S_prime, S_dprime, variables):

    m = variables[0]
    phi = np.arctan2(S_prime[0], S_prime[1])
    
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


# optimize trajectory
def optimize(path, params, viz=True):
    
    S = path.S
    S_prime = path.S_prime
    S_dprime = path.S_dprime
    
    # opt vars
    A = cv.Variable((path.test_length))
    B = cv.Variable((path.test_length))
    U = cv.Variable((2, path.test_length))

    cost = 0
    constr = []
    
    j = 0
    for i in path.theta_mid:
        if j < path.test_length:
            cost += path.dtheta*(cv.power(B[j+1],-.5) + cv.power(B[j],-.5))
            R, M, C, d = dynamics(S_prime[i], S_dprime[i], params.variables)   
            constr += [B[0] == 0, A[0] == 0, U[0,0] == 0, U[1,0] == 0]
            constr += [R*U[:,j+1] == M*A[j+1] + C*((B[j+1] + B[j])/2) + d]
            constr += [B[j+1] >= 0]
            constr += [cv.norm(U[:,j+1],2)<= params.C1]
            constr += [U[0,j+1] <= params.C2]
            constr += [B[j+1] == B[j]+2*A[j+1]*path.dtheta]
        j += 1
        
    problem = cv.Problem(cv.Minimize(2*cost), constr)
    solution = problem.solve(solver=cv.ECOS, max_iters=400)
    
    B, A, U = B.value, A.value, U.value
    if viz:
        simulate(B, A, U, params)

    return B, A, U


if __name__ == "__main__":
    
    # path can be 'qph', 'ellipse', 'wave', 'corner', rectangular'
    path = Path('map0', 275)
    params = define_params(m, Wf, ms, L, w, s_size, V0)

    B, A, U = optimize(path=path, params=params)
