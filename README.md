# Deep Successor Reinforcement Learning (DSR)

DSR is a hybrid model-free and model-based deep RL algorithm to learn robust value functions. It decomposes the value function into two components -- a reward predictor and a successor map. The successor map represents the expected future state occupancy from any given state and the reward predictor maps states to scalar rewards. The value function of a state can be computed as the inner product between the successor map and the reward weights. 

DSR has several appealing properties including: increased sensitivity to distal reward changes due to factorization of reward and world dynamics, and the ability to extract bottleneck states (subgoals) given successor maps trained under a random policy. 

### Illustration on Doom (VizDoom)
In this environment, the agent's objective is to gather ammo. 
#### Environment (video walkthrough) 
[![doom play 3roomsbest](http://img.youtube.com/vi/QcIwm-ucGgo/0.jpg)](https://youtu.be/QcIwm-ucGgo "dsr")

#### Policy after learning 
![DSR after convergence](http://i.imgur.com/25Pd85W.gif)

### Illustration of MazeBase
#### Random Policy  
![random](https://drive.google.com/drive/folders/0B3yyTdZ1crn4SG84Wk04Y0dWdTg)

## Instructions
* To start training see:
```
./runner.sh
```
* Insructions to run subgoal scripts coming soon
 
# Acknowledgements
* [Deepmind's DQN codebase](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner)
* [Facebook's MazeBase Environment](https://github.com/facebook/MazeBase).
* [VizDoom](vizdoom.cs.put.edu.pl)
