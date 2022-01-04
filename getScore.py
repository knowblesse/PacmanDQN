import numpy as np
import textDisplay
import game
from pacman import runGames
import pacmanAgents
import ghostAgents
import layout

Layout = layout.getLayout('smallGrid.lay')#layout to use in game
#Set parameters to run game. Do not choose KeyboardAgent when Display is off.
layout, pacman, ghosts, display, numGames, record = [Layout, pacmanAgents.RandomAgent(), [ghostAgents.PredatorGhost(i + 1) for i in range(4)], textDisplay.NullGraphics(), 10, False]
#Run pacman.py
pacmanrun = runGames(layout, pacman, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=30)
#Extract score data from game data
score_list = [game.state.getScore() for game in pacmanrun]
average_score = np.mean(score_list)

print("*****************************************"
      "\n", "Random agent got", "\n", score_list, "points!",
      "\n","Average score is", average_score, "!","\n"
      "*****************************************")