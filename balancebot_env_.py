import os
import random
import numpy as np
import gym
from gym import spaces,logger
from gym.utils import seeding
import pybullet as p
import pybullet_data
gravity = -9.81
timeStep = 0.02
urdf_root = "urdf/robot.urdf"

class BalanceBot:
	def __init__(self, render=True):
		self._gravity = gravity
		self._urdf = urdf_root
		self._timeStep = timeStep
		self._pybulletClient = p
		self._max_speed = 20.94 					# Max rotational speed of wheel in rad/s
		self._theta_threshold_radians = .16			# Angle at which to fail the episode
		self._x_threshold = .5
		self._cubeStartPos = [0, 0, 0]
		self._maxEnvSteps = 2000
		self._cubeStartOrientation = self._pybulletClient.getQuaternionFromEuler([.01,0,0])

		self.seed()
		self.state = None
		
		if render:
			self.physicsClient = self._pybulletClient.connect(self._pybulletClient.GUI)
		else:
			self.physicsClient = self._pybulletClient.connect(self._pybulletClient.DIRECT)  # non-graphical version

		self._pybulletClient.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

	def assign_throttle(self, action):
		self.vt = action
		self._pybulletClient.setJointMotorControlArray(bodyUniqueId=self.botId,
													   jointIndices=np.arange(2),
													   controlMode=self._pybulletClient.VELOCITY_CONTROL,
													   targetVelocities=[self.vt, -self.vt])
		return True

	def check_termination(self):
		done = self.bot_position[2] < self._theta_threshold_radians\
			or self.state[0] <= -self._x_threshold\
			or self.state[0] >= self._x_threshold\
			or self._envStepCounter >= self._maxEnvSteps
		return bool(done)

	def close():
		return self._pybulletClient.disconnect()

	def compute_reward(self):
		return .1 - abs(self.vt - self.vd) * 0.005

	def getActionDimensions(self):
		return (1,)

	def getActionUpperBound(self):
		return self._max_speed

	def getActionLowerBound(self):
		return -self.getActionUpperBound()

	def getObservationDimensions(self):
		return 4

	def getObservationUpperBound(self):
		# Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
		return np.array([self._x_threshold * 2,
						 np.finfo(np.float32).max,
						 self._theta_threshold_radians * 2,
						 np.finfo(np.float32).max])

	def getObservationLowerBound(self):
		return -self.getObservationUpperBound()

	def getBasePositionAndOrientation(self):
		return self._pybulletClient.getBasePositionAndOrientation(self.botId)

	def getBasePosition(self):
		return self._pybulletClient.getBasePositionAndOrientation(self.botId)[0]

	def getBaseOrientation(self):
		return self._pybulletClient.getBasePositionAndOrientation(self.botId)[1]			

	def getBaseVelocity(self):
		return self._pybulletClient.getBaseVelocity(self.botId)

	def getState(self):
		self.bot_position, bot_orientation = self.getBasePositionAndOrientation()
		bot_orientation_euler = self._pybulletClient.getEulerFromQuaternion(bot_orientation)
		linear_velocity, angular_velocity = self.getBaseVelocity()
		# Taking negative of bot_orientation as third value of state so that +ve y-axis becomes +ve angle and vice-versa.
		return np.array([self.bot_position[1], linear_velocity[1], -bot_orientation_euler[0], angular_velocity[0]])

	def _render(self, mode='human'):
		pass
	
	def _reset(self):
		self.vt = 0
		self.vd = 0
		# self.maxV = 20.9 # 235RPM = 24,609142453 rad/sec

		self._envStepCounter =0
		
		self._pybulletClient.resetSimulation()
		self._pybulletClient.setGravity(0, 0, self._gravity) # m/s^2
		
		self.setTimeSteps()
		self._pybulletClient.loadURDF("plane.urdf")

		path = os.path.abspath(os.path.dirname(__file__))
		self.botId = self._pybulletClient.loadURDF(os.path.join(path, self._urdf),
						   self._cubeStartPos,
						   self._cubeStartOrientation)
		
		self.steps_beyond_done = None
		self.state = self.getState()
		return self.state

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def setTimeSteps(self):
		self._pybulletClient.setTimeStep(self._timeStep)
	
	def _step(self, action):
		action_taken = self.assign_throttle(action)
		self._pybulletClient.stepSimulation()

		self.previous_state = self.state
		self.state = self.getState()
		done = self.check_termination()
		reward = self.compute_reward()
		# elif self.steps_beyond_done is None:
		# 	# Pole just fell!
		# 	self.steps_beyond_done = 0				# Vinit: What is steps_beyond_done?
		# 	reward = self.compute_reward()
	  
		self._envStepCounter += 1
		return self.state, reward, done, {}


class BalanceBotEnv(gym.Env, BalanceBot):
	def __init__(self):
		super().__init__()
		self.action_space = spaces.Box(low=self.getActionLowerBound(), high=self.getActionUpperBound(), shape=self.getActionDimensions(), dtype=np.float32)
		self.observation_space = spaces.Box(self.getObservationLowerBound(), self.getObservationUpperBound(), dtype=np.float32)

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		return self._step(action)

	def reset(self):
		return self._reset()


if __name__ == '__main__':
	bb = BalanceBotEnv()
	state = bb.reset()
	print(state)
	count = 0
	while True:
		action = bb.action_space.sample()
		action = np.array([0.0])
		count += 1
		state, reward, done, _ = bb.step(action)
		print('State:', state)
		print('Reward: ', reward)
		print('Done: ', done)
		# if count%500 == 0:
			# break