#!/usr/bin/env python3
import scipy.integrate as spi
import matplotlib.pyplot as plt
import numpy as np
import warnings
#from scipy.integrate import quad


'''
sys_const: systems constants
Type: np.array

init_cond: initial conditions
'''
class drone():
    g = 9.81 # Gravity constant
    def __init__(self, sys_const, time_step=0.001):

        # Vehicle Properties
        self.m = float(sys_const[0]) # Mass of drone
        self.g = 9.81
        self.Ixx = float(sys_const[1])
        self.Iyy = float(sys_const[2])
        self.Izz = float(sys_const[3])
        self.R_LD = float(sys_const[4])
        self.d = float(sys_const[5])
        self.r_D = float(sys_const[6])
        self.config = str(sys_const[8])
        self.sys_const = sys_const

        # Forces
        self.X = 0  # Force magnitude in X direction
        self.Y = 0  # Force magnitude in Y direction
        self.Z = 0  # Force magnitude in Z direction

        # Moment
        self.L = 0  # Rolling Moment
        self.M = 0  # Pitching Moment
        self.N = 0  # Yawing Moment

        # Velocity
        self.U = 0  # Velocity in X direction
        self.V = 0  # Velocity in Y direction
        self.W = 0  # Velocity in Z direction

        # Acceleration
        self.Udot = 0  # Acceleration in X direction
        self.Vdot = 0  # Acceleration in Y direction
        self.Wdot = 0  # Acceleration in Z direction

        # Angulare Velocity
        self.P = 0  # Roll Rate
        self.Q = 0  # Pitch Rate
        self.R = 0  # Yaw Rate

        # Angulare Acceleration
        self.Pdot = 0  # Roll Acceleration
        self.Qdot = 0  # Pitch Acceleration
        self.Rdot = 0  # Yaw Acceleration

        # Thrust of each motor
        self.T1 = 0
        self.T2 = 0
        self.T3 = 0
        self.T4 = 0

        self.T1dot = 0
        self.T2dot = 0
        self.T3dot = 0
        self.T4dot = 0

        # Set  Thrust Point
        self.set_T1 = 0
        self.set_T2 = 0
        self.set_T3 = 0
        self.set_T4 = 0

        # time constant for each motor
        self.tau1 = float(sys_const[7])
        self.tau2 = float(sys_const[7])
        self.tau3 = float(sys_const[7])
        self.tau4 = float(sys_const[7])

        self.t = 0  #
        self.dt = time_step


        # Position in the NED plane
        self.xPos = 0
        self.yPos = 0
        self.zPos = 0


        self.theta_dot = 0
        self.phi_dot = 0
        self.psi_dot = 0

        # Body angles of Drone
        self.theta = 0
        self.phi = 0
        self.psi = 0

        self.error = 0

        self.DCM = np.matrix([\
                    [np.cos(self.psi)*np.cos(self.theta),np.sin(self.psi)*np.cos(self.theta),-np.sin(self.theta)],\
                    [np.cos(self.psi)*np.sin(self.theta)*np.sin(self.phi)-np.sin(self.psi)*np.cos(self.phi),np.sin(self.psi)*np.sin(self.theta)*np.sin(self.phi)+np.cos(self.psi)*np.cos(self.phi), np.cos(self.theta)*np.sin(self.phi)],\
                    [np.cos(self.psi)*np.sin(self.theta)*np.cos(self.phi)+np.sin(self.psi)*np.sin(self.phi),np.sin(self.psi)*np.sin(self.theta)*np.cos(self.phi)-np.cos(self.psi)*np.sin(self.phi), np.cos(self.theta)*np.cos(self.phi)]\
                    ])


        # self.gravity_vec =np.matrix([ [-1*np.sin(self.theta)],[np.cos(self.theta)*np.sin(self.phi)],[np.cos(self.theta)*np.cos(self.phi)]])*self.m*self.g
        self.gravity_vec = np.array([-1*np.sin(self.theta), np.cos(self.theta)*np.sin(self.phi),np.cos(self.theta)*np.cos(self.phi)])*self.m*self.g

        #self.solver = scipy.integrate.RK4()

# Direct Cosine Matrix to execute axis system transformation through all three
# Euler angles
    def updateDCM():
        self.DCM = np.matrix([\
                    [np.cos(self.psi)*np.cos(self.theta),np.sin(self.psi)*np.cos(self.theta),-np.sin(self.theta)],\
                    [np.cos(self.psi)*np.sin(self.theta)*np.sin(self.phi)-np.sin(self.psi)*np.cos(self.phi),np.sin(self.psi)*np.sin(self.theta)*np.sin(self.phi)+np.cos(self.psi)*np.cos(self.phi), np.cos(self.theta)*np.sin(self.phi)],\
                    [np.cos(self.psi)*np.sin(self.theta)*np.cos(self.phi)+np.sin(self.psi)*np.sin(self.phi),np.sin(self.psi)*np.sin(self.theta)*np.cos(self.phi)-np.cos(self.psi)*np.sin(self.phi), np.cos(self.theta)*np.cos(self.phi)]\
                    ])

# Transpose of the Direct Cosine Matrix
    def transpose_DCM():
        return np.transpose(self.DCM)


    def bodyAngularRatesToEulerAngler(self):
        return np.matrix( [ [1, np.sin(self.phi)*np.tan(self.theta), np.cos(self.phi)*np.tan(self.theta)], \
                    [0, np.cos(self.phi), -1*np.sin(self.phi)], \
                    [0, np.sin(self.phi)/np.cos(self.theta), np.cos(self.phi)/np.cos(self.theta)] ] )*np.matrix([ [self.P],[self.Q],[self.R]])

    def setThrust(self,thrust):
        self.set_T1 = thrust[0]
        self.set_T2 = thrust[1]
        self.set_T3 = thrust[2]
        self.set_T4 = thrust[3]


    def setTimeConstants(self,taus):
        self.tau1 = taus[0]
        self.tau2 = taus[1]
        self.tau3 = taus[2]
        self.tau4 = taus[3]


# Returns the dynamic drag in Newton
# p: density of the fluid
# v: linear velocity of the object to the fluid
# Cd: Drag Coefficient
# A: Reference Area
    def aerodrag(p, v, Cd, A):
        return 0.5*p*np.square(v)*Cd*A

# Model for the wind disturbances
    def wind():
        pass

# Returns the components of (m*g)
    def update_gravity_vec(self):
        self.gravity_vec = np.array([-1*np.sin(self.theta), np.cos(self.theta)*np.sin(self.phi),np.cos(self.theta)*np.cos(self.phi)])*self.m*self.g


# Acceleration in the X direction
    def update_Udot(self):
        self.Udot = (np.divide(self.X, self.m) + self.V*self.R - self.W*self.Q)

# Acceleration in the Y direction
    def update_Vdot(self):
        self.Vdot = np.divide(self.Y, self.m) - self.U*self.R + self.W*self.P

# Acceleration in the Z direction
    def update_Wdot(self):
        self.Wdot = np.divide(self.Z, self.m) + self.U*self.Q - self.V*self.P

    def update_Pdot(self):
        self.Pdot = np.divide(self.L, self.Ixx) - np.divide( (self.Izz - self.Iyy), self.Ixx)*self.Q*self.R

    def update_Qdot(self):
        self.Qdot = np.divide(self.M,self.Izz) - np.divide((self.Ixx - self.Izz), self.Ixx)*self.Q*self.R

    def update_Rdot(self):
        self.Rdot = np.divide(self.N, self.Izz) - np.divide( (self.Iyy - self.Ixx ), self.Izz)*self.P*self.Q

    def update_phi_dot(self):
        self.phi_dot = self.P + np.sin(self.phi)*np.tan(self.theta)*self.Q + np.cos(self.phi)*np.tan(self.theta)*self.R

    def update_theta_dot(self):
        self.theta_dot = np.cos(self.phi)*self.Q + -np.sin(self.phi)*self.R

    def update_psi_dot(self):
        self.psi_dot = np.sin(self.phi)/( np.cos(self.theta))*self.Q + np.cos(self.phi)/(np.cos(self.theta))*self.R

    def update_T1(self):
        self.T1dot = self.set_T1/self.tau1 - self.T1/self.tau1

    def update_T2(self):
        self.T2dot = self.set_T2/self.tau2 - self.T2/self.tau2

    def update_T3(self):
        self.T3dot = self.set_T3/self.tau3 - self.T3/self.tau3

    def update_T4(self):
        self.T4dot = self.set_T4/self.tau4 - self.T4/self.tau4



    def sumForces_X(self):
        self.X = self.gravity_vec[0]

    def sumForces_Y(self):
        self.Y = self.gravity_vec[1]

    def sumForces_Z(self):
        if(self.config == '+'):
            self.Z = -1*(self.T1 + self.T2 + self.T3 + self.T4) + self.gravity_vec[2]


        elif(self.config == 'x'):
            self.Z = -1*(self.T1 + self.T2 + self.T3 + self.T4) + self.gravity_vec[2]



    def sumMoment_l(self):
        if(self.config == '+'):
            self.L = self.d*(self.T4-self.T2)

        elif(self.config == 'x'):
            # print("test")
            self.L = (self.d)/(np.sqrt(2))*(self.T1 + self.T2 + self.T3 - self.T4)

    def sumMoment_m(self):
        if(self.config == '+'):
            self.M = self.d*(self.T1-self.T3)

        elif(self.config == 'x'):
            self.M = (self.d)/(np.sqrt(2))*(self.T1 - self.T2 + self.T3 - self.T4)


    def sumMoment_n(self):
        if(self.config == '+'):
            self.N = (self.r_D*(-self.T1+self.T2-self.T3+self.T4))/self.R_LD

        elif(self.config == 'x'):
            self.N = (self.r_D)/(self.R_LD)*(self.T1 + self.T2 - self.T3 -self.T4)


    def thrust(self):
        pass

    def integration(self,x):
        # from scipy.integrate import quad
        # f = lambda t: x
        # return quad(f,self.t, self.t+self.dt)
        return x*self.dt


# Determine the next state conditions
    def step(self):

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                self.update_T1()
                self.update_T2()
                self.update_T3()
                self.update_T4()


                # Update all new forces and moments on the drone
                self.sumForces_X()
                self.sumForces_X()
                self.sumForces_Y()
                self.sumForces_Z()
                self.sumMoment_l()
                self.sumMoment_m()
                self.sumMoment_n()

                # Determine the new body Acceleration and angular rotaions
                self.update_Udot()
                self.update_Vdot()
                self.update_Wdot()
                self.update_Pdot()
                self.update_Qdot()
                self.update_Rdot()

                self.update_theta_dot()
                self.update_phi_dot()
                self.update_psi_dot()
                self.update_gravity_vec()
            except RuntimeWarning:
                print('RuntimeWarning: Generating new response')
                return True

        # self.update_gravity_vec()

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                self.T1 += self.integration(self.T1dot)
                self.T2 += self.integration(self.T2dot)
                self.T3 += self.integration(self.T3dot)
                self.T4 += self.integration(self.T4dot)


                self.U += self.integration(self.Udot)
                self.V += self.integration(self.Vdot)
                self.W += self.integration(self.Wdot)


                self.P += self.integration(self.Pdot)
                self.Q += self.integration(self.Qdot)
                self.R += self.integration(self.Rdot)



                self.xPos += self.integration(self.U)
                self.yPos += self.integration(self.V)
                self.zPos += self.integration(self.W)

                self.theta += self.integration(self.theta_dot)
                self.phi += self.integration(self.phi_dot)
                self.psi += self.integration(self.psi)
                self.t += self.dt
                return False
            #
            except:
                print("IntegrationWarning: Generating New Response")
                return True



    def getStates(self):
        return np.array([ self.xPos, self.yPos, self.zPos, self.theta_dot, self.phi_dot, self.psi_dot, self.theta, self.phi, self.psi])



    def getAllStates(self):
        return np.array([self.T1, self.T2,self.T3, self.T4, self.xPos, self.yPos,self.zPos,\
                    self.U, self.V,self.W, self.Udot,self.Vdot,self.Wdot, \
                    self.Pdot, self.Qdot, self.Rdot, self.P, self.Q, self.R,\
                    self.L,self.M,self.N,self.X,self.Y,self.Z])


    def getEstimatedStates(self):
        return np.array([self.xPos, self.yPos,self.zPos, self.U, self.V,self.W, \
                    self.Udot,self.Vdot,self.Wdot, self.Pdot, self.Qdot, self.Rdot,\
                    self.P, self.Q, self.R, self.theta, self.phi, self.psi])
