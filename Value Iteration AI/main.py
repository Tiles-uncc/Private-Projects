import copy
import numpy as np


def getIndexOfState(S, x, y):
    """ Get the index of the state that contains the (x,y) position in S

    Parameters:
        S (list): list of State objects
        x (integer): x coordinate of a cell
        y (integer): y coordinate of a cell

    Returns:
        int: in range [0,len(S)-1] if the position can be found. -1 otherwise
    """
    for i, s in enumerate(S):
        if s.x == x and s.y == y:
            return i
    return -1


def getPolicyForGrid(S, U, A, P, i_terminal_states):
    """ Computes the policy as a list of characters indicating which direction to move in at the state

    Parameters:
        S (list): States
        U (numpy array): Utilities
        A (list): Actions
        P (numpy array): Transition model matrix
        i_terminal_states (list): Indices of the terminal states

    Returns:
        list: 1d list of characters that indicate the action to take at each state
    """
    policy = []

    for i_s, s in enumerate(S):
        i_states = []

        # If it's a terminal state, then make the action be 'T'
        if i_s in i_terminal_states:
            action = 'T'

        # Otherwise, find the action that gives the best utility
        else:
            i_states = []

            # Get the index of each neighbor for a state
            i_up = getIndexOfState(S, s.x, s.y + 1)
            i_right = getIndexOfState(S, s.x + 1, s.y)
            i_down = getIndexOfState(S, s.x, s.y - 1)
            i_left = getIndexOfState(S, s.x - 1, s.y)

            # Check to make sure each one is not an obstacle
            if i_up != -1:
                i_states.append(i_up)

            if i_right != -1:
                i_states.append(i_right)

            if i_down != -1:
                i_states.append(i_down)

            if i_left != -1:
                i_states.append(i_left)

            # Append the state itself to consider the agent bouncing off the boundary
            i_states.append(i_s)

            # Calculate the expected utilities for each action in the state
            i_a_max_eu = 0
            max_eu = -100000  # don't wait to loop for i_a=0...
            for i_a, a in enumerate(A):

                # Get the expected utility for an action
                eu = 0
                for i_neighbor in i_states:
                    u_s_prime = U[i_neighbor]
                    prob_s_prime = P[i_a, i_s, i_neighbor]
                    eu += (prob_s_prime * u_s_prime)

                # Check if max expected utility
                if eu > max_eu:
                    max_eu = eu
                    i_a_max_eu = i_a

            # Set the action character
            action = A[i_a_max_eu]

        # Add the action to the policy
        policy.append(action)

    return policy


def printPolicyForGrid(policy, w, h, i_obs):
    """ Print out a policy in the form:
        ['r', 'r', 'r', 'T']
        ['u', '0', 'u', 'T']
        ['u', 'l', 'l', 'l']
        where the characters indicate the action to take at each state.
        '0' elements are obstacles in the grid.

    Parameters:
        policy (list): 1d list of characters indicating which action to take for each state
        w (int): width of the grid
        h (int): height of the grid
        i_obs(list): list of indices where obstacles are located

    Returns:
        None
    """

    # Insert 0's for obstacle tiles
    for i_ob in i_obs:
        policy.insert(i_ob, '0')

        # Blank line to isolate the policy
    print('\n')

    # Start at top of the grid, and print each row
    for y in range(h - 1, -1, -1):
        row = [policy[((w * y) + i)] for i in range(0, w)]
        print(row)

P = [[[0.1, 0.1, 0., 0., 0.8, 0., 0., 0., 0., 0., 0.],
      [0.1, 0.8, 0.1, 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0.1, 0., 0.1, 0., 0.8, 0., 0., 0., 0., 0.],
      [0., 0., 0.1, 0.1, 0., 0., 0.8, 0., 0., 0., 0.],
      [0., 0., 0., 0., 0.2, 0., 0., 0.8, 0., 0., 0.],
      [0., 0., 0., 0., 0., 0.1, 0.1, 0., 0., 0.8, 0.],
      [0., 0., 0., 0., 0., 0.1, 0.1, 0., 0., 0., 0.8],
      [0., 0., 0., 0., 0., 0., 0., 0.9, 0.1, 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0.1, 0.8, 0.1, 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0.1, 0.8, 0.1],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.1, 0.9]],

     [[0.1, 0.8, 0., 0., 0.1, 0., 0., 0., 0., 0., 0.],
      [0., 0.2, 0.8, 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0.1, 0.8, 0., 0.1, 0., 0., 0., 0., 0.],
      [0., 0., 0., 0.9, 0., 0., 0.1, 0., 0., 0., 0.],
      [0.1, 0., 0., 0., 0.8, 0., 0., 0.1, 0., 0., 0.],
      [0., 0., 0.1, 0., 0., 0., 0.8, 0., 0., 0.1, 0.],
      [0., 0., 0., 0.1, 0., 0., 0.8, 0., 0., 0., 0.1],
      [0., 0., 0., 0., 0.1, 0., 0., 0.1, 0.8, 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0.2, 0.8, 0.],
      [0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0.1, 0.8],
      [0., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0.9]],

     [[0.9, 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0.1, 0.8, 0.1, 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0.1, 0.8, 0.1, 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0.1, 0.9, 0., 0., 0., 0., 0., 0., 0.],
      [0.8, 0., 0., 0., 0.2, 0., 0., 0., 0., 0., 0.],
      [0., 0., 0.8, 0., 0., 0.1, 0.1, 0., 0., 0., 0.],
      [0., 0., 0., 0.8, 0., 0.1, 0.1, 0., 0., 0., 0.],
      [0., 0., 0., 0., 0.8, 0., 0., 0.1, 0.1, 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0.1, 0.8, 0.1, 0.],
      [0., 0., 0., 0., 0., 0.8, 0., 0., 0.1, 0., 0.1],
      [0., 0., 0., 0., 0., 0., 0.8, 0., 0., 0.1, 0.1]],

     [[0.9, 0., 0., 0., 0.1, 0., 0., 0., 0., 0., 0.],
      [0.8, 0.2, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0.8, 0.1, 0., 0., 0.1, 0., 0., 0., 0., 0.],
      [0., 0., 0.8, 0.1, 0., 0., 0.1, 0., 0., 0., 0.],
      [0.1, 0., 0., 0., 0.8, 0., 0., 0.1, 0., 0., 0.],
      [0., 0., 0.1, 0., 0., 0.8, 0., 0., 0., 0.1, 0.],
      [0., 0., 0., 0.1, 0., 0.8, 0., 0., 0., 0., 0.1],
      [0., 0., 0., 0., 0.1, 0., 0., 0.9, 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0.8, 0.2, 0., 0.],
      [0., 0., 0., 0., 0., 0.1, 0., 0., 0.8, 0.1, 0.],
      [0., 0., 0., 0., 0., 0., 0.1, 0., 0., 0.8, 0.1]]]
P = np.array(P)


class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def valueIterations(S, A, TM, R_states, discount, terminal_index_reward_pairs):
    U_prime = np.full(len(S), 0.0)
    U_prime[6] = -1.
    U_prime[10] = 1.
    threshold = 0.00000001

    delta = .1

    while delta > threshold:
        U = copy.deepcopy(U_prime)
        delta = 0.0

        for i in range(len(S)):
            if i != terminal_index_reward_pairs[0] and i != terminal_index_reward_pairs[1]:
                U_prime[i] = R_states[i] + (discount * getExpectedUtility(A, i, TM, U, S))
                if abs(U_prime[i] - U[i]) > delta:
                    delta = abs(U_prime[i] - U[i])
    return U


def getExpectedUtility(action, state, TM, U_value, set_of_states):
    x = []

    for i in range(len(action)):
        total = 0.0
        for j in range(len(set_of_states)):
            total += TM[i][state][j] * U_value[j]
        x.append(total)

    return max(x)

def main():
    states = [State(1, 1), State(1, 2), State(1, 3), State(1, 4),
              State(2, 1), State(2, 3), State(2, 4),
              State(3, 1), State(3, 2), State(3, 3), State(3, 4)]

    actions = ['u', 'r', 'd', 'l']

    rewards = np.array([-.04, -.04, -.04, -.04,
                        -.04, -.04, -1,
                        -.04, -.04, -.04, 1])

    discount = 1.0

    terminal = [6, 10]

    values = np.array(valueIterations(states, actions, P, rewards, discount, terminal))
    print(values)
    print()
    policy = getPolicyForGrid(states, values, actions, P, terminal)
    print("Policy: %s" % policy)
    printPolicyForGrid(policy, 4, 3, [5])


if __name__ == '__main__':
    main()
