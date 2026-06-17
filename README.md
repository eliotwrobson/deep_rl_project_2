[//]: # "Image References"
[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

# Deep Reinforcement Learning - Project 2: Continuous Control

## Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes. **Note:** We are use the multi-agent version of the environment.

## Environment Setup

This project requires Python 3.6 due to dependencies on `unityagents==0.4.0` and `tensorflow==1.7.1`.
Conda or micromamba is used for managing the environment.

### Unity Environment

The unity environment for this project is available here: https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started

Download the reacher environment appropriate for your operating system. Then be sure to set the path to the environment executable in `main.py`.

### Using Micromamba/Conda

Create the environment from the environment specification:

```bash
conda env create -f environment.yml
```

Then activate the environment:

```bash
conda activate deep_rl_p3
```

## Running the Project

Once the environment is activated, you can run the main training script:

```bash
python main.py
```

As the training runs, it will print once the environment is solved and keep training,
saving the best trained agents.
