import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DynamicalSystem:
	def __init__(self, state_size, control_size):
		self.state_size = state_size
		self.control_size = control_size
	def set_cost(self, Q, R):
		# one step cost = x.T * Q * x + u.T * R * u
		# refer to iLQR formulation
		self.Q = Q
		self.R = R
	def set_final_cost(self, Q_f):
		self.Q_f = Q_f
	def calculate_cost(self, x, u):
		return 0.5*((x-self.goal).T.dot(self.Q).dot(x-self.goal) + u.T.dot(self.R).dot(u))
	def calculate_final_cost(self, x):
		return 0.5*(x-self.goal).T.dot(self.Q_f).dot(x-self.goal)
	def set_goal(self, x_goal):
		self.goal = x_goal

class DoubleIntegrator(DynamicalSystem):
	def __init__(self):
		super().__init__(4, 2)
		# self.dt = 0.01
		self.dt = 0.05
		self.control_bound = np.ones(2) * 100 # [100, 100]
		self.goal = np.zeros(4)
	def transition(self, x, u):
		result = np.zeros(4)
		result[0:2] = x[0:2] + self.dt * x[2:4]
		result[2:4] = x[2:4] + self.dt * u
		return result
	def transition_J(self, x, u):
		#return matrix A, B, so that x = Ax + Bu
		A = np.zeros((self.state_size, self.state_size))
		B = np.zeros((self.state_size, self.control_size))
		A[0:self.state_size, 0:self.state_size] = np.identity(self.state_size)
		A[0, 2] = self.dt
		A[1, 3] = self.dt
		B[2, 0] = self.dt
		B[3, 1] = self.dt
		return A, B
	def draw_trajectories(self, x_trajectories):
		ax = plt.subplot(111)
		circle1 = plt.Circle((1, 1), 0.5, color=(0, 0.8, 0.8))
		circle2 = plt.Circle((1.5, 2.2), 0.5, color=(0, 0.8, 0.8))
		ax.add_artist(circle1)
		ax.add_artist(circle2)
		plt.scatter(x_trajectories[0, 0::5], x_trajectories[1, 0::5], 4,color='r')
		ax.set_aspect("equal")
		ax.set_xlim(0, 3)
		ax.set_ylim(0, 3)
		plt.show()
	def draw_u_trajectories(self, u_trajectories):
		x = plt.subplot(111)
		plt.scatter(u_trajectories[0, 0::5], u_trajectories[1, 0::5], 4,color='r')
		plt.show()

class Car(DynamicalSystem):
	def __init__(self):
		super().__init__(4, 2)
		self.dt = 0.05
		self.control_bound = np.array([np.pi/2, 10])
		self.goal = np.zeros(4)
	def transition(self, x, u):
		x_next = np.zeros(4)
		x_next[0] = x[0] + self.dt * x[3] * np.sin(x[2])
		x_next[1] = x[1] + self.dt * x[3] * np.cos(x[2])
		x_next[2] = x[2] + self.dt * u[1] * x[3]
		x_next[3] = x[3] + self.dt * u[0]
		return x_next
	def transition_J(self, x, u):
		A = np.identity(4)
		B = np.zeros((4, 2))
		A[0, 3] = np.sin(x[2]) * self.dt
		A[0, 2] = x[3] * np.cos(x[2]) * self.dt
		A[1, 3] = np.cos(x[2]) * self.dt
		A[1, 2] = -x[3] * np.sin(x[2]) * self.dt
		A[2, 3] = u[1] * self.dt
		B[2, 1] = x[3] * self.dt
		B[3, 0] = self.dt
		return A, B
	def draw_trajectories(self, x_trajectories):
		ax = plt.subplot(111)
		circle1 = plt.Circle((1, 1), 0.5, color=(0, 0.8, 0.8))
		circle2 = plt.Circle((2, 2), 1, color=(0, 0.8, 0.8))
		ax.add_artist(circle1)
		ax.add_artist(circle2)
		for i in range(0, x_trajectories.shape[1]-1, 5):
			circle_car = plt.Circle((x_trajectories[0, i], x_trajectories[1, i]), 0.1, facecolor='none')
			ax.add_patch(circle_car)
			ax.arrow(x_trajectories[0, i], x_trajectories[1, i], 0.1*np.sin(x_trajectories[2, i]), 0.1 * np.cos(x_trajectories[2, i]), head_width=0.05, head_length=0.1, fc='k', ec='k')
		ax.set_aspect("equal")
		ax.set_xlim(-1, 4)
		ax.set_ylim(-1, 4)
		plt.show()

class OneleggedHoppingRobot(DynamicalSystem):
	def __init__(self):
		super().__init__(3, 2)
		self.dt = 0.025
		self.control_bound = np.array([np.pi/2, 10, 10, 10])
		self.goal = np.zeros(12)
	def transition(self, x, u):
		x_next = np.zeros(4)
		x_next[0] = x[0] + self.dt * x[3] * np.sin(x[2])
		x_next[1] = x[1] + self.dt * x[3] * np.cos(x[2])
		x_next[2] = x[2] + self.dt * u[1] * x[3]
		x_next[3] = x[3] + self.dt * u[0]
		return x_next
	def transition_J(self, x, u):
		A = np.identity(4)
		B = np.zeros((4, 2))
		A[0, 3] = np.sin(x[2]) * self.dt
		A[0, 2] = x[3] * np.cos(x[2]) * self.dt
		A[1, 3] = np.cos(x[2]) * self.dt
		A[1, 2] = -x[3] * np.sin(x[2]) * self.dt
		A[2, 3] = u[1] * self.dt
		B[2, 1] = x[3] * self.dt
		B[3, 0] = self.dt
		return A, B