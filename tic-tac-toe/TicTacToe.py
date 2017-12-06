"""A Tic-Tac-Toe player class.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
from builtins import range
import random

import numpy as np


class TicTacToe(object):
    """A Tic-Tac-Toe player."""
    def __init__(self, initial_state=None, opponent_mode="random",
                 value_function=None, alpha=0.1):
        """Init Tic Tac Toe Game.

        # Parameters
            initial_state: numpy.ndarray
                empty state if it's None
                otherwise start from the state
                0 is empty
                1 is cross
                2 is circle
            opponent_mode: str
                random - play random move
                selfplay - play the move that hurts the best
            value_function: dict
                mapping from state to value
                initialize as empty dict if None
            alpha: float
                learning rate
        """
        if initial_state is None:
            self.state = np.zeros((3, 3))
            self.empty_state = self.get_empty_pos(self.state)
        else:
            self.state = initial_state
            self.empty_state = self.get_empty_pos(self.state)
        self.mode = opponent_mode
        if value_function is None:
            self.value_function = {}
            # set initial state value
            self.get_value(self.state)
        else:
            self.value_function = value_function
            self.get_value(self.state)
        self.alpha = alpha

    def get_empty_pos(self, state, verbose=False):
        """Get empty positions."""
        return (state == 0).nonzero()

    def reinitialize(self, clear_value=False, verbose=False):
        """Reinitialize the game."""
        self.state = np.zeros((3, 3))
        self.empty_state = self.get_empty_pos(self.state)

        if clear_value:
            self.value_function = {}
            self.get_value(self.state)

    def play_game(self, is_train=True, verbose=True):
        """Play one episode of the game.

        # Parameters
            verbose: bool
                print log if True
        """
        max_step = 9
        curr_step = 0
        win_flag = None

        while curr_step < max_step:
            # x play
            # select move
            best_move_value = 0
            best_state = None
            for id_x, id_y in zip(self.empty_state[0], self.empty_state[1]):
                temp_state = self.state.copy()
                temp_state[id_x, id_y] = 1
                value = self.get_value(temp_state)
                if value > best_move_value:
                    best_move_value = value
                    best_state = temp_state
            # update value accordingly
            curr_key = self.get_key(self.state)
            if best_state is not None:
                new_key = self.get_key(best_state)
                # temporal difference update
                if is_train:
                    self.value_function[curr_key] += \
                        self.alpha*(self.value_function[new_key] -
                                    self.value_function[curr_key])
                # update state
                self.state = best_state.copy()
                self.get_value(self.state)
                # update empty state list
                self.empty_state = self.get_empty_pos(self.state)
                # update step
                curr_step += 1
            # judge
            game_status = self.check_state(self.state)
            if verbose:
                self.print_game(self.state)
            if game_status == 1:
                win_flag = True
                break

            # o play
            # random play
            if self.mode == "random":
                # choose random state
                if self.empty_state[0].shape[0] != 0:
                    rand_idx = random.randint(
                        0, self.empty_state[0].shape[0]-1)
                    id_x, id_y = self.empty_state[0][rand_idx], \
                        self.empty_state[1][rand_idx]
                    # update state
                    self.state[id_x, id_y] = 2
                    self.get_value(self.state)
                    self.empty_state = self.get_empty_pos(self.state)
                    curr_step += 1
            elif self.mode == "selfplay":
                if self.empty_state[0].shape[0] != 0:
                    best_move_value = 0
                    best_state = None
                    for id_x, id_y in zip(
                            self.empty_state[0], self.empty_state[1]):
                        temp_state = self.state.copy()
                        temp_state[id_x, id_y] = 1
                        value = self.get_value(temp_state)
                        temp_state[id_x, id_y] = 2
                        if value > best_move_value:
                            best_move_value = value
                            best_state = temp_state
                    if best_state is not None:
                        # update state
                        self.state = best_state.copy()
                        self.get_value(self.state)
                        # update empty state list
                        self.empty_state = self.get_empty_pos(self.state)
                        # update step
                        curr_step += 1

            # judge again
            game_status = self.check_state(self.state)
            if verbose:
                self.print_game(self.state)
            if game_status == 0:
                win_flag = False
                break
        return win_flag

    def get_value(self, state, verbose=False):
        """Given a state, get a value.

        # Parameters
            state: numpy.ndarray
                a valid state as 3x3 matrix
            verbose: bool
                print log if true

        # Returns
            value: float
                the value given key
        """
        key = self.get_key(state)
        if key in self.value_function:
            if verbose:
                print ("The key %d=%.2f is found!",
                       (key, self.value_function[key]))
            return self.value_function[key]
        else:
            # add key for initial value
            value = self.check_state(state)
            self.value_function[key] = value
            return 0.5

    def check_substate(self, substate, verbose=False):
        """Check substate.

        # Parameters
            state: numpy.ndarray
                a vector of three elements
            verbose: bool
                print log if true

        # Returns
            value: float
                0: all circles
                1: all crosses
                0.5: no win
        """
        if substate[0] == 1 and substate[1] == 1 and substate[2] == 1:
            return 1
        elif substate[0] == 2 and substate[1] == 2 and substate[2] == 2:
            return 0
        else:
            return 0.5

    def check_state(self, state, verbose=False):
        """Check the state.

        # Parameters
            state: numpy.ndarray
                a valid state as 3x3 matrix
            verbose: bool
                print log if true

        # Returns
            value: float
                0: lose
                1: win
                0.5: in progress
        """
        # check if win, main player is cross
        if self.check_substate(state[0]) == 1 or \
           self.check_substate(state[1]) == 1 or \
           self.check_substate(state[2]) == 1 or \
           self.check_substate(state[:, 0]) == 1 or \
           self.check_substate(state[:, 1]) == 1 or \
           self.check_substate(state[:, 2]) == 1 or \
           self.check_substate([state[0, 0], state[1, 1],
                                state[2, 2]]) == 1 or \
           self.check_substate([state[0, 2], state[1, 1],
                                state[2, 0]]) == 1:
            return 1

        # check if lose
        if self.check_substate(state[0]) == 0 or \
           self.check_substate(state[1]) == 0 or \
           self.check_substate(state[2]) == 0 or \
           self.check_substate(state[:, 0]) == 0 or \
           self.check_substate(state[:, 1]) == 0 or \
           self.check_substate(state[:, 2]) == 0 or \
           self.check_substate([state[0, 0], state[1, 1],
                                state[2, 2]]) == 0 or \
           self.check_substate([state[0, 2], state[1, 1],
                                state[2, 0]]) == 0:
            return 0

        return 0.5

    def get_state(self, key, verbose=False):
        """Get state from a key.

        # Parameters
            key: int
                the key of the state
            verbose: bool
                print log if true
        # Returns
            state: numpy.ndarray
                the state
        """
        state = np.zeros((3, 3))
        state[2, 2] = key // 3**8
        key -= state[2, 2]*3**8
        state[2, 1] = key // 3**7
        key -= state[2, 1]*3**7
        state[2, 0] = key // 3**6
        key -= state[2, 0]*3**6
        state[1, 2] = key // 3**5
        key -= state[1, 2]*3**5
        state[1, 1] = key // 3**4
        key -= state[1, 1]*3**4
        state[1, 0] = key // 3**3
        key -= state[1, 0]*3**3
        state[0, 2] = key // 3**2
        key -= state[0, 2]*3**2
        state[0, 1] = key // 3**1
        key -= state[0, 1]*3**1
        state[0, 1] = key

        return state

    def get_key(self, state, verbose=False):
        """Get key of a state.

        # Parameters
            state: numpy.ndarray
                a valid state as 3x3 matrix
            verbose: bool
                print log if true

        # Returns
            key: int
                the key value given state
        """
        # convert state to key
        key = state[0, 0]+state[0, 1]*3+state[0, 2]*3**2 + \
            state[1, 0]*3**3+state[1, 1]*3**4+state[1, 2]*3**5 + \
            state[2, 0]*3**6+state[2, 1]*3**7+state[2, 2]*3**8
        return int(key)

    def print_game(self, state):
        """Print game given state."""
        print_dict = {1: "x ", 2: "o ", 0: "# "}
        print ("-"*30)
        for id_x in range(3):
            for id_y in range(3):
                print (print_dict[state[id_x, id_y]], end="")
            print("\n", end="")
        print ("-"*30)
