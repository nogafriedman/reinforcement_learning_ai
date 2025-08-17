This project is part of the Introduction to Artificial Intelligence course (Exercise 6).
The focus of this assignment is on Reinforcement Learning, specifically implementing and experimenting with Q-learning and SARSA algorithms in a grid-world environment.

Project Structure:
gridworld.py - Implements the GridWorld environment
qlearning_agent.py - Contains the Q-learning agent implementation
sarsa_agent.py - Contains the SARSA agent implementation
util.py - Helper functions and data structures
analysis.py - Experiment results and comparisons

How to Run:

Run experiments with Q-learning:
python3 gridworld.py --agent=QLearningAgent --episodes=500 --epsilon=0.1 --alpha=0.5 --gamma=0.9

Run experiments with SARSA:
python3 gridworld.py --agent=SARSAAgent --episodes=500 --epsilon=0.1 --alpha=0.5 --gamma=0.9

Visualize results:
python3 gridworld.py --agent=QLearningAgent --episodes=100 --display=GUI
python3 gridworld.py --agent=SARSAAgent --episodes=100 --display=Summary

