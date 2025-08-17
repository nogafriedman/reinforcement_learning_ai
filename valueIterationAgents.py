# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util
import time

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        "*** YOUR CODE HERE ***"
        self.best_action = dict()
        for state in self.mdp.getStates():
            self.best_action[state] = None
        self.current_discount = discount

        for state in self.mdp.getStates():
            if mdp.isTerminal(state):
                continue
            state_max_reward = -1*float('inf')
            for action in self.mdp.getPossibleActions(state):
                state_action_reward = 0
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    state_action_reward += prob * self.mdp.getReward(state, action, next_state)

                if state_action_reward >= state_max_reward:
                    self.best_action[state] = action
                    state_max_reward = state_action_reward
            self.values[state] = state_max_reward
        
        for it in range(iterations-1):
            kv = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                state_max_reward = -1*float('inf')
                has_actions = False
                for action in self.mdp.getPossibleActions(state):
                    state_action_reward = 0
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if self.mdp.isTerminal(next_state):
                            continue
                        next_state_reward = kv[next_state]*self.current_discount
                        next_state_reward += self.mdp.getReward(state, action, next_state)
                        next_state_reward *= prob
                        state_action_reward += next_state_reward
                        has_actions = True
                    if state_action_reward > state_max_reward:
                        self.best_action[state] = action
                        state_max_reward = state_action_reward
                if has_actions:     # don't change the values of terminal states
                    self.values[state] = state_max_reward

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getQValue(self, state, action):
        """
          The q-value of the state action pair
          (after the indicated number of value iteration
          passes).  Note that value iteration does not
          necessarily create this quantity and you may have
          to derive it on the fly.
        """
        "*** YOUR CODE HERE ***"
        transition_states_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        state_action_reward = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            state_action_reward += self.current_discount * prob * self.values[next_state]
        return state_action_reward
    
    def getPolicy(self, state):
        """
          The policy is the best action in the given state
          according to the values computed by value iteration.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        return self.best_action[state]

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)
  
