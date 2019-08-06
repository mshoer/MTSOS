import numpy as np
import cvxpy as cv
from scipy.integrate import ode, odeint
from utils import generate_way_points
import matplotlib.pyplot as plt
from rk6 import odeintRK6

# sim fails for large area because dynamics is non ode, but ode solver is used

# vehicle parameters
mass = 1            # vehicle mass [kg]
Wf = .5             # percent of weight on front tires 
muf = .8            # coefficient of friction between tires and ground
gravity = 9.81      # gravity
rho = 1.225         # density of air [kg/m^3]
Cd = 0.4            # drag coefficient
area = 0.2          # front area


def define_params(mass, Wf, muf, gravity):
    """
    params derived from vehicle config
    """

    params = {}
    Fmax = muf*mass*gravity
    params['mass'] = mass
    params['rho'] = rho
    params['Cd'] = Cd
    params['area'] = area
    params['Fmax'] = Fmax
    params['Flongmax'] = Fmax*Wf
    return params
        

def define_path(track, plot_results=True):
    """
    calculate s, s_prime, s_dprime using way points
    """

    x, y = generate_way_points(track)
    num_wpts = np.size(x)

    theta = np.linspace(0, 1, num_wpts)
    dtheta = 1/(num_wpts-1)
    S = np.array([x, y])

    S_middle = np.zeros([2,num_wpts-1])
    S_prime = np.zeros([2,num_wpts-1])
    S_dprime= np.zeros([2,num_wpts-1])

    for j in range(num_wpts-1):

        S_middle[:,j] = (S[:,j] + S[:,j+1])/2
        S_prime[:,j] = (S[:,j+1] - S[:,j])/dtheta

        if j==0:
            S_dprime[:,j] = (S[:,j]/2 - S[:,j+1] + S[:,j+2]/2)/(dtheta**2)
        elif j==1 or j==num_wpts-3:
            S_dprime[:,j] = (S[:,j-1] - S[:,j] - S[:,j+1] + S[:,j+2])/2/(dtheta**2)
        # elif j==num_wpts-3:
        #     S_dprime[:,j] = (S[:,j-2] - S[:,j-1] - S[:,j] + S[:,j+1])/(2*dtheta**2)
        elif j==num_wpts-2:
            S_dprime[:,j] = (S[:,j-1]/2 - S[:,j] + S[:,j+1]/2)/(dtheta**2)
        else:
            S_dprime[:,j] = (- 5/48*S[:,j-2] + 13/16*S[:,j-1] - 17/24*S[:,j] - 17/24*S[:,j+1] + 13/16*S[:,j+2] - 5/48*S[:,j+3])/(dtheta**2)

    path = {
            'theta': theta, 
            'dtheta': dtheta, 
            'S': S, 
            'S_middle': S_middle, 
            'S_prime': S_prime, 
            'S_dprime': S_dprime,
            }

    return path


def dynamics(velocity, phi, params):
    """
    dynamics (non-linear)
    """

    mass = params['mass']
    rho = params['rho']
    Cd = params['Cd']
    area = params['area']

    xdot = velocity[0]
    ydot = velocity[1]
    
    R = np.zeros((2,2))
    R[0,0] = np.cos(phi)
    R[0,1] = -np.sin(phi)
    R[1,0] = np.sin(phi)
    R[1,1] = np.cos(phi)

    M = np.zeros((2,2))
    M[0,0] = mass
    M[1,1] = mass
    
    C = np.zeros((2,2))
    C[0,0] = 1/2*rho*Cd*area*xdot
    C[1,1] = 1/2*rho*Cd*area*ydot

    d = np.zeros((2))

    return R, M, C, d


def dynamics_cvx(S_prime, S_dprime, params):
    """
    dynamics (convexified)
    """

    phi = np.arctan2(S_prime[1], S_prime[0])
    xdot = S_prime[0]
    ydot = S_prime[1]
    
    R, M, C, d = dynamics([xdot, ydot], phi, params)

    C = np.dot(M, S_dprime) + np.dot(C, S_prime)
    M = np.dot(M, S_prime)
    return R, M, C, d


def friction_circle(Fmax):
    t = np.linspace(0, 2*np.pi, num=100)
    x = Fmax*np.cos(t)
    y = Fmax*np.sin(t)
    return x, y


def diffequation(t, x, u, R, M, C, d):
    """
    write as first order ode
    """
    x0dot = x[2:]
    x1dot = np.dot(np.linalg.inv(M), np.dot(R, u) - np.dot(C, x[2:]) - d)
    return np.concatenate([x0dot, x1dot], axis=0)


# simulate control inputs
def simulate(b, a, u, params, int_method ='rk6', plot_results=True):
    """
    integrate using ode solver
    """

    theta = path['theta']
    dtheta = path['dtheta']
    S = path['S']
    S_prime = path['S_prime']
    S_dprime = path['S_dprime']
    num_wpts = theta.size

    # initialize position, velocity
    x, y = np.zeros([num_wpts]), np.zeros([num_wpts])
    x[0], y[0] = S[0,0], S[1,0]
    vx, vy = np.zeros([num_wpts]), np.zeros([num_wpts])

    # calculate time for each index
    bsqrt = np.sqrt(b)
    dt = 2*dtheta/(bsqrt[0:num_wpts-1]+bsqrt[1:num_wpts])
    t = np.zeros([num_wpts])
    for j in range(1, num_wpts):
        t[j] = t[j-1] + dt[j-1]
    print('The optimal time to traverse is {:.4f}s'.format(t[-1]))

    # integrate
    if int_method == 'odeint':
        print('using Runge Kutta sixth order integration')
        for j in range(num_wpts-1):
            phi = np.arctan2(S_prime[1,j], S_prime[0,j])
            R, M, C, d = dynamics([vx[j], vy[j]], phi, params)
            odesol = odeint(diffequation, [x[j], y[j], vx[j], vy[j]], [t[j], t[j+1]], 
                            args=(u[:,j], R, M, C, d), tfirst=True)
            x[j+1], y[j+1], vx[j+1], vy[j+1] = odesol[-1,:]

    elif int_method == 'rk6':
        print('using Runge Kutta sixth order integration')
        for j in range(num_wpts-1):
            phi = np.arctan2(S_prime[1,j], S_prime[0,j])
            R, M, C, d = dynamics([vx[j], vy[j]], phi, params)
            odesol = odeintRK6(diffequation, [x[j], y[j], vx[j], vy[j]], np.linspace(t[j], t[j+1], 10),
                            args=(u[:,j], R, M, C, d))
            x[j+1], y[j+1], vx[j+1], vy[j+1] = odesol[-1,:]

    else:
        integrator = ode(diffequation).set_integrator('dopri5')
        print('using Runge Kutta fourth order integration')
        for j in range(num_wpts-1):
            phi = np.arctan2(S_prime[1,j], S_prime[0,j])
            R, M, C, d = dynamics([vx[j], vy[j]], phi, params)
            integrator.set_initial_value([x[j], y[j], vx[j], vy[j]], t[j]).set_f_params(u[:,j], R, M, C, d)
            x[j+1], y[j+1], vx[j+1], vy[j+1] = integrator.integrate(t[j+1])


    if plot_results:

        flg, ax = plt.subplots(1)
        plt.plot(S[0,:], S[1,:], '-b.', label='original')
        plt.plot(x, y, 'r.', label='simulated')
        plt.plot(S[0,0], S[1,0], 'g*', label='start')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('global position')
        plt.legend()

        flg, ax = plt.subplots(1)
        c1 = friction_circle(params['Fmax'])
        plt.plot(u[0,:], u[1,:], 'b.', label='optimal inputs')
        plt.plot(c1[0], c1[1], 'r-', label='friction circle')
        plt.plot(u[0,0], u[1,0], 'g*', label='start')
        plt.xlabel('force long [N]')
        plt.ylabel('force lat [N]')
        plt.title('optimal inputs on friction circle')
        plt.legend(loc='right')

        flg, ax = plt.subplots(1)
        # plt.plot(t, vx, '-b.', label='speed x')
        # plt.plot(t, vy, '-r.', label='speed y')
        plt.plot(t, np.sqrt(vx**2+vy**2), '-g.', label='speed abs')
        plt.title('speed vs time')
        plt.xlabel('time [s]')
        plt.ylabel('speed [m/s]')
        plt.legend()

        flg, ax = plt.subplots(1)
        plt.plot(t[1:], u[0,:], '-b.', label='force long')
        plt.plot(t[1:], u[1,:], '-r.', label='force lat')
        plt.title('force vs time')
        plt.xlabel('time [s]')
        plt.ylabel('force [N]')
        plt.legend()

        plt.show()
    
    return


def optimize(path, params, plot_results=True):
    """
    main function to optimize trajectory
    solves convex optimization
    """
    
    theta = path['theta']
    dtheta = path['dtheta']
    S = path['S']
    S_prime = path['S_prime']
    S_dprime = path['S_dprime']
    num_wpts = theta.size
    
    # opt vars
    A = cv.Variable((num_wpts-1))
    B = cv.Variable((num_wpts))
    U = cv.Variable((2, num_wpts-1))

    cost = 0
    constr = []

    # no constr on A[0], U[:,0], defined on mid points
    constr += [B[0] == 0]

    for j in range(num_wpts-1):

        cost += 2*dtheta*cv.inv_pos(cv.power(B[j],0.5) + cv.power(B[j+1],0.5))

        R, M, C, d = dynamics_cvx(S_prime[:,j], S_dprime[:,j], params)
        constr += [R*U[:,j] == M*A[j] + C*((B[j] + B[j+1])/2) + d]
        constr += [B[j] >= 0]
        constr += [cv.norm(U[:,j],2) <= params['Fmax']]
        constr += [U[0,j] <= params['Flongmax']]
        constr += [B[j+1] - B[j] == 2*A[j]*dtheta]
        
    problem = cv.Problem(cv.Minimize(cost), constr)
    solution = problem.solve(solver=cv.ECOS, verbose=False)

    B, A, U = B.value, A.value, U.value
    B = abs(B)
    if plot_results:
        simulate(B, A, U, params)

    return B, A, U


if __name__ == "__main__":
    
    # path can be 'qph', 'ellipse', 'wave', 'corner', rectangular2D', 'random', 'map0', 'levine'
    path = define_path('map0')
    params = define_params(mass, Wf, muf, gravity)

    B, A, U = optimize(path=path, params=params)
