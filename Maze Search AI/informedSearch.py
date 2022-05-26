import heapq
import math
import queue
import random
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

def readGrid(filename):
    grid = []
    with open(filename) as f:
        for l in f.readlines():
            grid.append([int(x) for x in l.split()])

    f.close()
    return grid


def outputGrid(grid, start, goal, path):
    filenameStr = 'path.txt'

    # open file
    f = open(filenameStr, 'w')

    # mark start/goal points
    grid[start[0]][start[1]] = 'S'
    grid[goal[0]][goal[1]] = 'G'

    # mark intermediate points
    for i, p in enumerate(path):
        if i > 0 and i < len(path) - 1:
            grid[p[0]][p[1]] = '+'

    # write grid to file
    for r, row in enumerate(grid):
        for c, col in enumerate(row):

            if c < len(row) - 1:
                f.write(str(col) + ' ')
            else:
                f.write(str(col))

        if r < len(grid) - 1:
            f.write("\n")

    # close file
    f.close()


# generates a random grid
def genGrid():
    print('In genGrid')

    num_rows = 10
    num_cols = 10

    grid = [[0] * num_cols for i in range(0, num_rows)]

    max_cost = 5
    ob_cost = 0

    for i_r in range(0, num_rows):
        for i_c in range(0, num_cols):

            # Default to obstacle cost
            cost = ob_cost

            # Chance to be an obstacle
            chance = random.random()
            if chance > 0.2:
                # Generate a random cost for the location
                cost = random.randint(1, max_cost)

            grid[i_r][i_c] = cost

    return grid


def printGrid(grid):
    for i in range(len(grid)):
        print(grid[i])


def printNodeList(l):
    for node in l:
        print(node.value)

def setPath(current, path):
    while current.parent != '':
        path.insert(0, current.parent.value)
        current = current.parent


class Node:
    def __init__(self, value, parent, g, h):
        self.value = value
        self.parent = parent
        self.g = g  # path cost
        self.h = h  # heuristic cost
        self.f = self.g + self.h  # g + h

    def __lt__(self, other):
        return self.f < other.f


def heuristic(location1, location2):
    return math.dist(location1, location2)


# Determines if list is in inList
def inList(node, theList):
    for n in theList:
        if n.value == node.value:
            return True
    return False


# Defines possible directions for node to go
def getNeighbors(location, grid):
    result = []

    up = location[:]  # [:]copys list and assigns to up
    up[0] -= 1
    if up[0] > -1 and grid[up[0]][up[1]] != 0:
        result.append(up)

    right = location[:]
    right[1] += 1
    if right[1] < len(grid[right[0]]) and grid[right[0]][right[1]] != 0:
        result.append(right)

    down = location[:]
    down[0] += 1
    if down[0] < len(grid) and grid[down[0]][down[1]] != 0:
        result.append(down)

    left = location[:]
    left[1] -= 1
    if left[1] > -1 and grid[left[0]][left[1]] != 0:
        result.append(left)

    return result


# Expands the node by putting unchecked nodes into openlist
def expandNode(node, openList, closedList, grid, goal, start):
    neighbors = getNeighbors(node.value, grid)  # neighbors is children
    for c in neighbors:  # for a location point in neighbors
        child = Node(c, node, node.g + grid[c[0]][c[1]], heuristic(c, goal))  # Node(value, parent, g, h, f)
        if not inList(child, closedList) and not inList(child,
                                                        openList):  # if openList.contains(c) == false and closedList.contains(c) == false
            heapq.heappush(openList, child)  # adds c object to openListCopy

    return openList


# Displays the grid
def printGrid(grid):
    for i in range(0, len(grid)):
        print(grid[i])


# Sets the path by visiting previous parent nodes
def setPath(current, path):
    while current.parent != '':
        path.insert(0, current.parent.value)
        current = current.parent


# Gets the path cost by iterating through path
def getPathCost(p, grid):
    cost = 0
    for i in range(len(p)):
        cost += grid[p[i][0]][p[i][1]]
    return cost


# Implements uninformed search
def aStarSearch(grid, start, goal):
    current = Node(start, '', 0, heuristic(start, goal))
    openList = []
    heapq.heapify(openList)
    heapq.heappush(openList, current)

    closedList = []
    path = []
    numExpanded = 0

    # while not openList.empty():
    while len(openList) > 0:
        # current = openList.get() # removes and returns element
        current = heapq.heappop(openList)
        closedList.append(current)
        # print(closedList[0].value)
        if current.value == goal:
            break
        else:
            openList = expandNode(current, openList, closedList, grid, goal, start)
            numExpanded += 1
    # if not openList.empty() or current == goal:
    if len(openList) > 0 or current == goal:
        setPath(current, path)
        path.append(goal)

    return [path, numExpanded]
def genStartGoal(grid):
    sRow = random.randint(0, len(grid) - 1)
    sCol = random.randint(0, len(grid[0]) - 1)

    while grid[sRow][sCol] == 0:
        sRow = random.randint(0, len(grid) - 1)
        sCol = random.randint(0, len(grid[0]) - 1)

    gRow = random.randint(0, len(grid) - 1)
    gCol = random.randint(0, len(grid[0]) - 1)

    while grid[gRow][gCol] == 0:
        gRow = random.randint(0, len(grid) - 1)
        gCol = random.randint(0, len(grid[0]) - 1)

    return [sRow, sCol], [gRow, gCol]


def genGrid(size, max_cost=9):
    """ Generates a grid with random values in range [0,max_cost] where 0s represent obstacle cells and 1-max_cost represent the step cost to move onto the cell from any neighbor.

    Parameters:
        size (int): The number of rows and columns to useh
        max_cost (int): The max step cost to move onto a celll in the grid

    Returns:
        2D list: The randomly generated grid
    """

    num_rows = size
    num_cols = size

    grid = [[0] * num_cols for i in range(0, num_rows)]

    ob_cost = 0
    ob_prob = 0.2

    for i_r in range(0, num_rows):
        for i_c in range(0, num_cols):

            # Default to obstacle cost
            cost = ob_cost

            # Chance to be an obstacle
            chance = random.random()
            if chance > ob_prob:
                # Generate a random cost for the location
                cost = random.randint(1, max_cost)

            grid[i_r][i_c] = cost

    return grid


def labelTile(grid, r, c, ax, text):
    """ Puts a character onto a grid cell, and changes text color based on cell color

    Parameters:
        grid (2D list): The grid to visualize
        r (int): The row of the cell to label
        c (int): The column of the cell to label
        text (string): The text to put onto the grid cell

    Returns:
        None
    """
    if grid[r][c] <= 3:
        ax.text(c, r, text, color="white", ha='center', va='center')
    else:
        ax.text(c, r, text, color="black", ha='center', va='center')


def visualizeGrid(grid, path=False, block=False, max_cost=9):
    """ Displays the grid as a grayscale image where each cell is shaded based on the step cost to move onto it.

    Parameters:
            grid (2D list): The grid to visualize
            path (2D list): The path to visualize.
            block (bool): True if pyplot.show should block program flow until the window is closed
            max_cost(int): Maximum step cost to move onto any cell in the grid

    Returns:
            None
    """
    tempGrid = []

    # Flip the values so that darker means larger cost
    for r in grid:
        row = []
        for col in r:
            if col != 0:
                col = (max_cost + 1) - col
            row.append(col)
        tempGrid.append(row)

    # Create colors
    cmap = matplotlib.cm.gray
    norm = colors.Normalize(vmin=0, vmax=max_cost)

    # Call imshow
    fig, ax = plt.subplots()
    ax.imshow(tempGrid, interpolation="none", cmap=cmap, norm=norm)

    # Put a 'p' character for each path position
    for i, loc in enumerate(path):
        if i == 0:
            labelTile(tempGrid, loc[1], loc[0], ax, "S")
        elif i == len(path) - 1:
            labelTile(tempGrid, loc[1], loc[0], ax, "G")
        else:
            labelTile(tempGrid, loc[1], loc[0], ax, "p")

    if len(grid) <= 20:
        # Set ticks
        tickInc = 1
    else:
        tickInc = int(len(grid) / 10)

    ax.set_xticks(np.arange(0, len(grid) + 1, tickInc))
    ax.set_yticks(np.arange(0, len(grid[0]) + 1, tickInc))
    ax.set_xticklabels(np.arange(0, len(grid) + 1, tickInc))
    ax.set_yticklabels(np.arange(0, len(grid[0]) + 1, tickInc))

    plt.show(block=False)


def runTests(displayGrids=True):
    """ Runs a series of planning queries on randomly generated maps, map sizes, and start and goal pairs

        Parameters:
                displayGrid (bool): True will use matplotlib to visualize the grids

        Returns:
                None
    """
    numExpanded = []
    totalGridSize = 20
    gridSizes = [i for i in range(10, totalGridSize, 5)]

    numTests = 2

    # For each grid size
    for gs in gridSizes:
        numEx = []
        # Do X tests where X=numTests
        for i in range(0, numTests):

            # Get random grid, start, and goal
            grid = genGrid(gs)
            start, goal = genStartGoal(grid)

            # Call algorithm
            [p, numExp] = aStarSearch(grid, start, goal)

            # Display grids if desired
            if i < 2 and gs <= 50 and displayGrids:
                visualizeGrid(grid, p)

            # Store data for single run
            numEx.append(numExp)

        # Store data for grid size
        numExpanded.append(numEx)

    # Get average of expanded nodes for each grid size
    neAvg = []
    for i, n in enumerate(numExpanded):
        print("Grid size: %s" % gridSizes[i])
        avg = 0
        for e in n:
            avg += e
        avg = avg / len(n)
        neAvg.append(avg)
        print("Average number of expanded nodes: %s" % avg)

    # Display bar graph for expanded node data
    plt.clf()
    plt.bar(gridSizes, neAvg)
    plt.show()


def main():
    runTests()


if __name__ == '__main__':
    main()
    print('\nExiting normally')
