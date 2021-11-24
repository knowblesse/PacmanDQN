# PacmanEnv

Modified Pac-man environment for dynamic navigation task

## Quickstart

Run following command on kernel with default parameters below

* Pacman Agent : None -> keyboard Input
* Ghost Agent : Random movement ghost
* Layout : 16 x 16 open space with three reward and one ghost

```shell
$ python3 pacman.py
```

## Environment Structure

### Game

All rules, valid movements, score information are store in three files. 

* ***game.py*** : rules
* ***pacman.py*** : main function
* ***utils.py*** : helper functions

DETAIL @SY

### Agents

There are two agent classes in the environment all inherited from the **Agent**(*game.py*) class.

**GhostAgent** : All ghost agent is defind in the *ghostAgents.py* file and must inherit **GhoastAgent**(*ghostAgent.py*) class.

Ghost agent is indexed by the class name and this name can be used by inserting '-g' option upon calling the *pacman.py*.
(ex. python3 pacman.py -g PredatorGhost)

**PacmanAgent** : Pacman agent is defind in the *pacmanAgents.py* file.

Default agent is KeyboardAgent were it receives user input for movement selection.

See *pacmanDQN_Agents.py* for self-behaving agent example.

###Directions

###Configuration

Get agents' coordinate from self.pos as (x, y) and travelling directions from self.direction

* getPosition(self):

* getDirection(self):

###AgentState

Hold the state of an agent (configuration, speed, etc)

* getPosition(self): Get agent position from Configuration

* getDirection(self): Get agent travelling direction from Configuration

###Grid

Return positions of Pacman map as [x][y]. x horizontal and y vertical.

###Actions

Manipulate move actions

Define moving directions of agents as tuple

* getPossibleActions(config, walls): Get agents' position and wall position. Return every possible position of agent

* getLegalNeighbors(position, walls): Get 

###GameStateData

Get new state data from previous event. 



### Layout

Layout file is defined in the *./layouts/* and has the **.lay** file extension.

DETAIL @SY

### Graphics

DETAIL @SY


## Citation

This repository is folked from tychovdo/PacmanDQN and original work is done by UC Berkeley AI class project (http://ai.berkeley.edu).

Citation below is from the forked repo.

```
@article{van2016deep,
  title={Deep Reinforcement Learning in Pac-man},
  subtitle={Bachelor Thesis},
  author={van der Ouderaa, Tycho},
  year={2016},
  school={University of Amsterdam},
  type={Bachelor Thesis},
  pdf={https://esc.fnwi.uva.nl/thesis/centraal/files/f323981448.pdf},
}

```

* [van der Ouderaa, Tycho (2016). Deep Reinforcement Learning in Pac-man.](https://esc.fnwi.uva.nl/thesis/centraal/files/f323981448.pdf)

## Acknowledgements

DQN Framework by  (made for ATARI / Arcade Learning Environment)
* [deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow) ([https://github.com/mrkulk/deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow))

Pac-man implementation by UC Berkeley:
* [The Pac-man Projects - UC Berkeley](http://ai.berkeley.edu/project_overview.html) ([http://ai.berkeley.edu/project_overview.html](http://ai.berkeley.edu/project_overview.html))
