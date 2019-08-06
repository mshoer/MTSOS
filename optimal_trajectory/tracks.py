import sys
import numpy as np
import matplotlib.pyplot as plt

class rectangular:
	"""base class for rectangular tracks"""

	def __init__(self, length, breadth, width):
		super(rectangular, self).__init__()
		
		self.length = length
		self.breadth = breadth
		self.width = width
		self.track_length = 2*self.length + 2*self.breadth

		self.parametric()


	def parametric(self):

		line = lambda l,b: [( l/2, -b/2), ( l/2,  b/2),
					   (-l/2,  b/2), (-l/2, -b/2)]

		l, b = self.length, self.breadth
		self.center = line(self.length, self.breadth)
		self.outer = line(self.length+self.width, self.breadth+self.width)
		self.inner = line(self.length-self.width, self.breadth-self.width)
		self.theta_start = [l/2+0, l/2+b, l/2+l+b, l/2+l+2*b]
		self.theta_end = [l/2+b, l/2+l+b, l/2+l+2*b, l/2+2*l+2*b]
		

class rectangular2D(rectangular):
	"""
	track is parametrized by its length theta: [0,1]->R^2 (x, y)
	"""

	def __init__(self, length, breadth, width):
		super(rectangular2D, self).__init__(length, breadth, width)


	def param_to_xy(self, theta, **kwargs):

		if "length" not in kwargs:
			length = self.length
		else:
			length = kwargs["length"]
		if "breadth" not in kwargs:
			breadth = self.breadth
		else:
			breadth = kwargs["breadth"]

		theta = theta%(2*(length+breadth))
		if theta<=length/2:
			x = theta
			y = -breadth/2

		elif theta>length/2 and theta<=length/2+breadth:
			x = length/2
			y = -breadth/2 + (theta - length/2)

		elif theta>length/2+breadth and theta<=3/2*length+breadth:
			x = length/2 - (theta - length/2 - breadth)
			y = breadth/2

		elif theta>3/2*length+breadth and theta<=3/2*length+2*breadth:
			x = -length/2
			y = breadth/2 - (theta - 3/2*length - breadth)

		elif theta>3/2*length+2*breadth and theta<=2*length+2*breadth:
			x = -length/2 + (theta - 3/2*length - 2*breadth)
			y = -breadth/2
		return x, y


	def trajectory(self, num=100, **kwargs):
		
		if "length" not in kwargs:
			length = self.length
		else:
			length = kwargs["length"]
		if "breadth" not in kwargs:
			breadth = self.breadth
		else:
			breadth = kwargs["breadth"]

		s = np.linspace(0,2*(length+breadth)-1e-2,num)
		x = np.empty([num])
		y = np.empty([num])
		for ids, theta in enumerate(s):
			x[ids], y[ids] = self.param_to_xy(theta, **kwargs)
		return x, y


class rectangular3D(rectangular):
	"""
	track is parametrized by its length theta: [0,1]->R^3 (x, y, psi)
	"""

	def __init__(self, length, breadth, width):
		super(rectangular3D, self).__init__(length, breadth, width)


	def param_to_xy(self, theta, **kwargs):

		if "length" not in kwargs:
			length = self.length
		else:
			length = kwargs["length"]
		if "breadth" not in kwargs:
			breadth = self.breadth
		else:
			breadth = kwargs["breadth"]

		theta = theta%(2*(length+breadth))
		if theta<=length/2:
			x = theta
			y = -breadth/2
			psi = 0

		elif theta>length/2 and theta<=length/2+breadth:
			x = length/2
			y = -breadth/2 + (theta - length/2)
			psi = np.pi/2

		elif theta>length/2+breadth and theta<=3/2*length+breadth:
			x = length/2 - (theta - length/2 - breadth)
			y = breadth/2
			psi = np.pi

		elif theta>3/2*length+breadth and theta<=3/2*length+2*breadth:
			x = -length/2
			y = breadth/2 - (theta - 3/2*length - breadth)
			psi = 3*np.pi/2

		elif theta>3/2*length+2*breadth and theta<=2*length+2*breadth:
			x = -length/2 + (theta - 3/2*length - 2*breadth)
			y = -breadth/2
			psi = 0
		return x, y, psi


	def trajectory(self, num=100, **kwargs):
		
		if "length" not in kwargs:
			length = self.length
		else:
			length = kwargs["length"]
		if "breadth" not in kwargs:
			breadth = self.breadth
		else:
			breadth = kwargs["breadth"]

		s = np.linspace(0,2*(length+breadth)-1e-2,num)
		x = np.empty([num])
		y = np.empty([num])
		psi = np.empty([num])
		for ids, theta in enumerate(s):
			x[ids], y[ids], psi[ids] = self.param_to_xy(theta, **kwargs)
		return x, y, psi