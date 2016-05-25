# Deep Successor Reinforcement Learning (DSR)

DSR is a hybrid model-free and model-based deep RL algorithm to learn robust value functions. It decomposes the value function into two components -- a reward predictor and a successor map. The successor map represents the expected future state occupancy from any given state and the reward predictor maps states to scalar rewards. The value function of a state can be computed as the inner product between the successor map and the reward weights. 

DSR has several appealing properties including: increased sensitivity to distal reward changes due to factorization of reward and world dynamics, and the ability to extract bottleneck states (subgoals) given successor maps trained under a random policy. 

# Code coming shortly ... 
