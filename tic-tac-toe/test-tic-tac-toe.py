"""Script for test TicTacToe.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
from builtins import range

from TicTacToe import TicTacToe


# get a game
game = TicTacToe(opponent_mode="random")

num_eposide = 100000
success_rate = 0
for episode_idx in range(1, num_eposide+1):
    win_flag = game.play_game(verbose=False)
    game.reinitialize()

    if win_flag:
        success_rate += 1
    if episode_idx % 1000 == 0:
        print (success_rate/float(episode_idx))

game.play_game(is_train=False, verbose=True)
print (game.value_function)
