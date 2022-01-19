# pacmanAgent
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import Agent
from game import Directions
import util
import random
import numpy as np

class RandomAgent(Agent):
    """
    Random movement Agent
    """
    def __init__(self, index=0):
        self.index = index

    def initialize(self):
        pass

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist

class KeyboardAgent(Agent):
    """
    An agent controlled by the keyboard.
    """
    # NOTE: Arrow keys also work.
    WEST_KEY = 'a'
    EAST_KEY = 'd'
    NORTH_KEY = 'w'
    SOUTH_KEY = 's'
    STOP_KEY = 'q'

    def __init__(self, index=0):

        self.lastMove = Directions.STOP
        self.index = index
        self.keys = []

    def initialize(self):
        self.lastMove = Directions.STOP

    def getAction(self, state):
        from graphicsUtils import keys_waiting
        from graphicsUtils import keys_pressed
        keys = keys_waiting() + keys_pressed()
        if keys != []:
            self.keys = keys

        legal = state.getLegalActions(self.index)
        move = self.getMove(legal)

        if move == Directions.STOP:
            # Try to move in the same direction as before
            if self.lastMove in legal:
                move = self.lastMove

        if (self.STOP_KEY in self.keys) and Directions.STOP in legal:
            move = Directions.STOP

        if move not in legal:
            move = random.choice(legal)

        self.lastMove = move
        return move

    def getMove(self, legal):
        move = Directions.STOP
        if (self.WEST_KEY in self.keys or 'Left' in self.keys) and Directions.WEST in legal:
            move = Directions.WEST
        if (self.EAST_KEY in self.keys or 'Right' in self.keys) and Directions.EAST in legal:
            move = Directions.EAST
        if (self.NORTH_KEY in self.keys or 'Up' in self.keys) and Directions.NORTH in legal:
            move = Directions.NORTH
        if (self.SOUTH_KEY in self.keys or 'Down' in self.keys) and Directions.SOUTH in legal:
            move = Directions.SOUTH
        return move

class ReflexAgent(Agent):
    """
    Run away from ghosts, run towards to foods
    """
    def __init__(self, index=0):
        self.index = index

    def initialize(self):
        pass

    def getAction(self, state):
        possibleActions = state.getLegalActions(self.index)

        if len(possibleActions) == 1:
            return possibleActions

        state_size = 7
        visual_range = int((state_size - 1) / 2)
        visualField = np.array(util.getState(state, state_size))

        if np.any(visualField == 2): # ghost detected
            if np.any(visualField[0:visual_range-1,:] == 2): # ghost at left
                if 'East' in possibleActions:
                    return Directions.EAST
                else:
                    if 'West' in possibleActions:
                        possibleActions.remove('West')
                    return np.random.choice(possibleActions)
            if np.any(visualField[visual_range+1:,:] == 2): # ghost at right
                if 'West' in possibleActions:
                    return Directions.WEST
                else:
                    if 'East' in possibleActions:
                        possibleActions.remove('East')
                    return np.random.choice(possibleActions)
            if np.any(visualField[:,0:visual_range-1] == 2): # ghost at bottom
                if 'North' in possibleActions:
                    return Directions.NORTH
                else:
                    if 'South' in possibleActions:
                        possibleActions.remove('South')
                    return np.random.choice(possibleActions)
            if np.any(visualField[:,visual_range+1:] == 2): # ghost at top
                if 'South' in possibleActions:
                    return Directions.SOUTH
                else:
                    if 'North' in possibleActions:
                        possibleActions.remove('North')
                    return np.random.choice(possibleActions)
        else:
            if np.any(visualField == 3):  # food detected
                if np.any(visualField[0:visual_range-1, :] == 3):
                    if 'West' in possibleActions:
                        return Directions.WEST
                if np.any(visualField[visual_range+1:, :] == 3):
                    if 'East' in possibleActions:
                        return Directions.EAST
                if np.any(visualField[:, 0:visual_range] == 3):
                    if 'South' in possibleActions:
                        return Directions.SOUTH
                if np.any(visualField[:, visual_range:] == 3):
                    if 'North' in possibleActions:
                        return Directions.NORTH
            else: # go random
                if (state.getPacmanState().getDirection() is not 'Stop') and (state.getPacmanState().getDirection() in possibleActions):
                    if np.random.rand() < 0.7:
                        return state.getPacmanState().getDirection()
                    else:
                        return np.random.choice(possibleActions)
        possibleActions.remove('Stop')
        return np.random.choice(possibleActions)
