# BalanceBot Reinforcement Learning Environment

Source Code Author: Vinit Sarode

This repository contains the pybullet environment for a 2 wheeled balancing robot. Along with this, we have provided a gym environment with it to integrate into RL framework. We also provide a code in tensorflow 2 and Keras for A2C algorithm.

## The Robot:
<p align="center">
	<img src="https://github.com/vinits5/BalanceBot_A2C/blob/main/images/without_gravity.png" height="300">
</p>
<p align="center">
	<img src="https://github.com/vinits5/BalanceBot_A2C/blob/main/images/with_gravity.png" height="300">
</p>
All the stl files and urdf file are defined in urdf directory.

> python balancebot_env.py

## Train the A2C algorithm:
> python train_a2c_tfKeras.py

## Test the policy:
> python train_a2c_tfKeras.py -t -w 'path of pretrained policy'