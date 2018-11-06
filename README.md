# Project 3 - Continuous Control of Collaborative Multiple Agents Using Deep Deterministic Policy Gradients (DDPG) 

## Introduction

The third project in Udacity Deep Reinforcement Learning nanodegree consists on solving a continuous problem named "Tennis" - two agents control rackets to bounce a ball over a net - using reinforcement learning. 

Our solution uses a standard Deep Deterministic Policy Gradients (DDPG) implementation as described in the original [research paper](https://arxiv.org/pdf/1509.02971.pdf), to solve the *multi-agent* version of the problem. 

The implementation was based on the sample provided as an exercise in the course, modified to use the [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) environment, and use the [PyTorch](https://www.pytorch.org/) framework.

## The Environment

In this project our algorithm must control two intependent agents, each one responsible for moving a racket (in 2D space), bouncing a ball over the net. When an agent hits the ball over the net, it receives a reward of *+1*; conversely, if the agent isn't succesful (the ball hits the ground or goes out of bounds), it receives a reward of *-0.01*. 

The state space perceived by each agent is a vector with *8 continuous dimensions*, representing the position and velocity of the ball and racket. Each agent receives its own, local observation. After each observation of the state space, the agent may produce an action consisting of a vector with two numbers between -1 and 1, representing the movement toward (or away from) the net, and jumping. 

This environment is a variant created for the nanodegree and provided as a compiled Unity binary. The image below was part of the problem description and illustrates the multi-agent version of the problem. 

![Tennis](tennis.png)

## Getting Started

All the work was performed on a Windows 10 laptop, with a GeForce GTX 970M GPU. The training was performed using CUDA. 

After cloning the project, download and extract the pre-built "Tennis" environment using the link adequate to your operational system, in the same directory of the project:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Tennis/one_agent/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Tennis/one_agent/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Tennis/one_agent/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Tennis/one_agent/Tennis_Windows_x86_64.zip)

It is also necessary to install [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), [unityagents](https://pypi.org/project/unityagents/) and [NumPy](http://www.numpy.org/). Our development used an Anaconda (Python 3.6) environment to install all packages.  

## Training the agent

Run `python ./train.py` to train the agent using DDPG. The average rewards over 100 consecutive episodes will be printed to the standard output. 

At the end, the plot showing the agent progress will be saved in the image `training.png`, and the model (weights learned by the agent) will be saved in files `checkpoint_actor.pth` and `checkpoint_critic.pth` . 

## Running a trained agent

The repository already contains weights trained using DDPG (files `checkpoint_actor.pth` and `checkpoint_critic.pth`).

Run `python ./test.py checkpoint_actor.pth checkpoint_critic.pth` to see the agent in action! 

## Report 

Please refer to file `Report.md` for a detailed description of the solution, including neural network architecture and hyperparameters used. 

