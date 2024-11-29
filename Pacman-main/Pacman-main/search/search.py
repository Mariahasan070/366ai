# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import*
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    import time

    # Start the timer
    start_time = time.time()

    # Create a stack for storing states to visit
    stack = util.Stack()
    visited = set()
    stack.push((problem.getStartState(), []))
    
    # Counter for the total number of nodes visited
    total_nodes_visited = 0

    while not stack.isEmpty():
        current_state, path = stack.pop()

        if problem.isGoalState(current_state):
            # Calculate execution time in milliseconds
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            print(f"Execution Time: {execution_time:.4f} milliseconds")
            print(f"Total Nodes Visited: {total_nodes_visited}")  # Print total nodes visited
            return path

        if current_state not in visited:
            visited.add(current_state)
            total_nodes_visited += 1  # Increment the counter when a node is visited
            for successor, action, cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    stack.push((successor, path + [action]))

    # If the goal state is not found
    execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"Execution Time: {execution_time:.4f} milliseconds")
    print("No path found to the goal.")
    print(f"Total Nodes Visited: {total_nodes_visited}")  # Print total nodes visited
    return []



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    import time

    # Start the timer
    start_time = time.time()

    currPath = []  # The path that is popped from the frontier in each loop
    currState = problem.getStartState()  # The state(position) that is popped for the frontier in each loop
    print(f"Start State: {currState}")

    # Checking if the start state is also a goal state
    if problem.isGoalState(currState):
        # Calculate and print execution time
        execution_time = (time.time() - start_time) * 1000
        print(f"Execution Time: {execution_time:.4f} milliseconds")
        print("Total Nodes Visited: 1")  # Only the start state was visited
        return currPath

    frontier = util.Queue()
    frontier.push((currState, currPath))  # Insert just the start state, in order to pop it first
    explored = set()

    # Counter for total nodes visited
    total_nodes_visited = 0

    while not frontier.isEmpty():
        currState, currPath = frontier.pop()  # Popping a state and the corresponding path

        if currState not in explored:
            explored.add(currState)
            total_nodes_visited += 1  # Increment the node counter

            # Check if the current state is the goal
            if problem.isGoalState(currState):
                # Calculate and print execution time
                execution_time = (time.time() - start_time) * 1000
                print(f"Execution Time: {execution_time:.4f} milliseconds")
                print(f"Total Nodes Visited: {total_nodes_visited}")  # Print total nodes visited
                return currPath

            # Get successors and add to the frontier if not already explored
            frontierStates = [t[0] for t in frontier.list]
            for s in problem.getSuccessors(currState):
                if s[0] not in explored and s[0] not in frontierStates:
                    frontier.push((s[0], currPath + [s[1]]))  # Add successor state and updated path to frontier

    # If no solution is found, calculate and print execution time
    execution_time = (time.time() - start_time) * 1000
    print(f"Execution Time: {execution_time:.4f} milliseconds")
    print(f"Total Nodes Visited: {total_nodes_visited}")  # Print total nodes visited
    return []  # If this point is reached, no solution was found




def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    import time

    # Start the timer
    start_time = time.time()

    # Priority Queue to manage nodes to explore, prioritized by cost
    frontier = util.PriorityQueue()

    # Set to track visited states
    visited = set()

    # Counter for total nodes visited
    total_nodes_visited = 0

    # Get the starting state of the problem
    start_state = problem.getStartState()

    # Push the start state into the priority queue with an initial cost of 0
    # Each element in the queue is a tuple: (state, path, total_cost)
    frontier.push((start_state, [], 0), 0)  # The priority is the cost (0 initially)

    # While there are still nodes in the priority queue
    while not frontier.isEmpty():
        # Pop the node with the lowest cost
        cur_state, path, cost = frontier.pop()

        # If the current state is the goal, return the path to the goal
        if problem.isGoalState(cur_state):
            # Calculate and print the execution time in milliseconds
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            print(f"Execution Time: {execution_time:.4f} milliseconds")
            print(f"Total Nodes Visited: {total_nodes_visited}")
            return path

        # If the current state has not been visited
        if cur_state not in visited:
            # Mark the state as visited
            visited.add(cur_state)
            total_nodes_visited += 1  # Increment the node counter

            # Explore the successors (neighbors) of the current state
            for successor, action, step_cost in problem.getSuccessors(cur_state):
                # Calculate the total cost to reach the successor
                total_cost = cost + step_cost

                # Push the successor into the priority queue with the updated cost
                frontier.push((successor, path + [action], total_cost), total_cost)

    # If no solution is found, return an empty path and print the execution time
    execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"Execution Time: {execution_time:.4f} milliseconds")
    print(f"Total Nodes Visited: {total_nodes_visited}")
    return []









    


    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
