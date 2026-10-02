# rl_drone_research_assignment
Research assignment where a drone is taught to land on a tilted, collapsable tower through reinforcement learning. Each directory contains a different subtask as follows:

1. drone_rl
This directory will contain the full, functioning landing task.

2. hover
This directory contains the environment used to train the drone to hover at a stationary point. The starting point of the drone is set to be its goal position, such that it must learn to maintain its position, rather than learn to also approach a defined goal point far away.

This environment was done as an initial test as to verify the drone was capable of learning a simple challange.

3. get_to_point:
This directory contains the environment used to train the drone to reach a defined goal position and stop there. It was designed as a direct follow-up to the hovering task, where the drone must now not only learn to sustain flight, but also to move towards its desired position.

This environment can play a larger role in the final landing task, as teaching the drone to first place itself above the tower could be an important first step to teach landing, for example in the context of curriculum learning.


# Code Scripts
Each directory includes 3 scripts:
1. test_env.py: This script defines the simulation environment, from the MuJoCo simulation to RL parameters (action space, observation space, reward policy, termination conditions)

2. train.py: This script defines the training conditions, such as the total number of timesteps and number of environments. Running it will start the training of a new agent, which will by default save within the "ppo_models" directory. It will save both the final model at the end of the timesteps (FINAL), as well as the model at the timestep with the highest average reward (BEST).

3. evaluate.py: This script allows for the trained agents to be visualized and evaluated. It runs 10 episodes of the selected agent, reporting back its positions over time as it runs, and compiles the distribution of termination reasons from those 10 attempts.


# Running the Code
Here is a brief explanation on how to run the scripts:

1. Training a new Model

To train a new model, simply enter the directory of the task you wish to train, and run the train.py script with the following command:

```
python3 train.py
```

Once finished, the "FINAL" and "BEST" models will both the saved within the "ppo_models" directory.

2. Evaluating a Model

To visualize a trained model, simply run the following command:

```
python3 evaluate.py
```

From there, a list of all saved models will be displayed, from which you may choose which model you would like to visualize. Once a model is selected, 10 episodes will be visualized within MuJoCo, whilst some evaluation details will be printed to the console. Each directory should come with pretrained demo models that may be evaluated to visualize the task working.