# Quick Start Guide

You can run the Animal AI environment in three different ways:
- running the the standalone `AnimalAI` executable
- running a configuration file via `visualizeArena.py`
- start training using `train.py`

## Running the standalone arena

Running the executable `AnimalAI` that you should have separately downloaded and added to the `envs` folder starts a 
playable environment with default configurations in a single arena. You can toggle the camera between First Person and 
Bird's eye view using the `C` key on your keyboard. The agent can then be controlled using `W,A,S,D` on your keyboard. 
The objects present in the configuration are randomly sampled from the list of objects that can be spawned, their 
location is random too. Hitting `R` or collecting rewards will reset the arena.

**Note**: on some platforms, running the standalone arena in full screen makes the environment slow, keep the 
environment in window mode for better performance.

## Running a specific configuration file

The `visualizeArena.py` script found in the main folder allows you to visualize an arena configuration file. We provide 
sample configuration files for you to experiment with. To make your own environment configuration file we advise to read 
thoroughly the [configuration file documentation page](configFile.md). You will find a detailed list of all the objects on the [definitions of objects page](definitionsOfObjects.md). Running this script only allows for a single arena to be visualized at once, as there can only be a single agent you control.

For example, to run an environment that contains the agent, a goal, and some randomly placed walls use:

```
python visualizeArena.py configs/obstacles.yaml
```


## Start training your agent

Once you're happy with your arena configuration you can start training your agent. This can be done in a way very similar 
to a regular [gym](https://github.com/openai/gym) environment. We provide a template training file `train.py` you can run 
out of the box, it uses the [ML agents' PPO](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-PPO.md) 
for training. We added the ability for participants to **change the environment configuration between episodes**. You can 
find more details about that in the [training documentation](training.md).
