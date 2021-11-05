# ghostAgents1.py
# --------------
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
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util
import numpy as np


class GhostAgent(Agent):

    def __init__(self, index):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()

class PredatorGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, index):
        self.index = index
        self.chaseTime = 0
        self.isChase = False
        self.maxChaseTime = 50
        self.bestActionProb = 0.8
        self.initChaseDistance = 5
        self.isRoam = True
        self.roamTime = 0
        self.maxRoamTime = 50

    def getDistribution(self, state):
        # Read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        ghostPosition = state.getGhostPosition(self.index)
        speed = 0.8

        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(ghostPosition[0] + a[0], ghostPosition[1] + a[1]) for a in actionVectors] # test all ghostPositionsible next Positions

        pacmanPosition = state.getPacmanPosition()
        startPosition = ghostState.start.getPosition()

        # Check the distances
        distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in newPositions]
        distancesFromStart = [manhattanDistance(pos, startPosition) for pos in newPositions]

        # Check if there is a wall between pacman and ghost
        walls = state.getWalls()
        bestActionIndex = np.argmin(distancesFromStart)

        if self.isChase: # if in chase, follow
            if self.chaseTime > self.maxChaseTime:
                self.isChase = False
                self.isRoam = True
                bestActionIndex = np.argmin(distancesFromStart)
                self.chaseTime = 0
                print(f'Ghost{self.index} stops the chase!')
            else:
                bestActionIndex = np.random.choice(np.where(distancesToPacman == np.min(distancesToPacman))[0])
            self.chaseTime += 1
        else: #not chasing, decide whether to follow or not;
            isPacmanClose = False
            wallFound = False
            if min(distancesToPacman) < self.initChaseDistance:# pacman is in range
                isPacmanClose = True
                if int(ghostPosition[0]) == pacmanPosition[0]:
                    for j in range(int(min(ghostPosition[1], pacmanPosition[1]))+1, int(max(ghostPosition[1], pacmanPosition[1]))):
                        if walls[pacmanPosition[0]][j] == True:
                            print("wall")
                            wallFound = True
                            break
                elif int(ghostPosition[1]) == pacmanPosition[1]:
                    for j in range(int(min(ghostPosition[0], pacmanPosition[0]))+1, int(max(ghostPosition[0], pacmanPosition[0]))):
                        if walls[j][pacmanPosition[1]] == True:
                            print("wall")
                            wallFound = True
                            break

            if isPacmanClose and (not wallFound):
                bestActionIndex = np.random.choice(np.where(distancesToPacman == np.min(distancesToPacman))[0])
                self.isChase = True
                self.isRoam = False
                print(f'Ghost{self.index} is now on chase!')

            else: # if not in chase, stay closer to the start position
                if self.isRoam:
                    dist = util.Counter()
                    for a in state.getLegalActions(self.index):
                        dist[a] = 1.0
                    dist.normalize()
                    self.roamTime += 1
                    if self.roamTime == self.maxRoamTime:
                        self.isRoam = False
                    return dist

                bestActionIndex = np.argmin(distancesFromStart)
                if ghostPosition == startPosition:
                    self.isRoam = True
                    self.roamTime = 0

    # Construct distribution
        dist = util.Counter()
        for a in legalActions:
            dist[a] += (1 - self.bestActionProb) / len(legalActions)
        dist[legalActions[bestActionIndex]] += self.bestActionProb
        dist.normalize()
        return dist
