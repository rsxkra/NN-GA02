# Snake Reinforcement Learning

Adapted from DragonWarrior15 Keras/TensorFlow codebase to PyTorch. Not everything is implemented, done for a class task.
Code for training a Deep Reinforcement Learning agent to play the game of Snake.
The agent takes 2 frames of the game as input (image) and predicts the action values for
the next action to take.
***
Sample games from the best performing agent at iteration 185500<br>
<img width="400" height="400" src="https://github.com/rsxkra/NN-GA02/blob/main/images/game_visual_v17.1_185500_14_ob_4_1.gif" alt="model v17.1 agent" ><img width="400" height="400" src="https://github.com/rsxkra/NN-GA02/blob/main/images/game_visual_v17.1_185500_14_ob_4_2.gif" alt="model v17.1 agent" >

***

## Running Graded assignment 02.
[game_environment.py](../game_environment.py) contains the necessary code to create and interact with the snake environment (class Snake and SnakeNumpy). The interface is similar to openai gym interface.

[agent.py](../agent.py) contains the agent for playing the game. It implements and trains a convolutional neural network for the action values, using a DeepQLearningAgent. Which is a Deep Q Learning Algorithm with CNN Network.

[training.py](../training.py) contains the complete code to train an agent.

[game_visualization.py](../game_visualization.py) contains the code to convert the game to mp4 format is desired.

### Regular steps
Run training.py, then select the best run to use in game_visualization.py, ex. 185 500 (in this case) and run the visualization.

## Additional dependencies.
matplotlib FFMPEG
* conda install -c conda-forge ffmpeg (for conda enviorments)

I used PyTorch verson 2.8. Some older keywords for functions are deprecated here, so I need to use the new keywords, which then may not work on older versions. However, I have added the deprecated varients to as fallback, which I think should work, but I have not tested it.
