# Autonomous Car FoodFinder

<h2><b>1. Introduction and Description</b></h2>

This document is a comprehensive guide to the FoodFinder Autonomous Car project. It is a simple reinforcement learning project entirely written in Python and its associated frameworks and libraries. Within 30-45 minutes, a user will be able to train a car to drive to every fool pellet while staying within the road infrastructure.

<p align="center">
<img src= https://user-images.githubusercontent.com/70033778/232990007-97b392cf-362f-4424-9281-86cea2ee2c7e.png>
</p>
<p align="center">
<img src= https://user-images.githubusercontent.com/70033778/232990029-bbc48238-a787-46ea-88df-8e191eaf4ea7.png>
</p>

<h2><b>2. Required Packages and Software</b></h2>

This project can run on any platform that is able to support the following packages:

<ul>
  <li>Python version >= 3.6</li>
  <li>NumPy version >= 1.19.5</li>
  <li>PyTorch version >= 1.12.1</li>
  <li>Matplotlib version >= 3.3.6</li>
</ul>

A display must be available if you want to enable the GUI as shown in Figure 1. The GUI can be disabled by commenting out the following line in the play_step() function of game.py:

```self.update_ui()```

along with commenting out all Pygame components. They only reside in game.py.

<br>
<h2><b>3. Running the Code </b></h2>

After all the necessary packages have been installed, the Zip file containing the code can be extracted. This should contain 6 Python files: agent.py, calc.py, car.py, game.py, model.py, and train.py, related as such:

<p align="center">
<img src= https://user-images.githubusercontent.com/70033778/232991297-4bbd56d7-bbfe-469a-883a-9b87cc2b699c.png>
</p>
<p align="center">
<img src= https://user-images.githubusercontent.com/70033778/232991449-054f0b72-e878-4ea8-8b4f-067742c74e02.png>
</p>

Run the following command in a Linux (or Mac) terminal to start training the car:
```python3 train.py```

Running the above command will produce a GUI as shown in Figure 1 accompanied with a text output in the terminal that contains all of the statistics for each training iteration:

<p align="center">
<img src= https://user-images.githubusercontent.com/70033778/232991850-de0a5577-f266-4138-a061-90b5a2d1da4e.png>
</p>
<p align="center">
<img src= https://user-images.githubusercontent.com/70033778/232991769-31e63cfe-f18d-473d-ae90-d1c99bd4dc47.png>
</p>

This contains information on the training iteration, reward accrued after each episode, the highest reward seen across all training, the average of the last 25 rewards, and the highest average reward seen across all sliding windows of 25 consecutive episodes. Usually, after 30 – 45 minutes of training, the car is able to consistently maneuver around and keep collecting each food pellet without colliding with the frame or going off road.

<br>
<h2><b>4. Adjustable Code </b></h2>

You can adjust the hyperparameters of the model in agent.py, such as the <b>learning rate</b>, <b>discount rate</b>, <b>memory size</b>, and <b>number of neurons per layer</b>.

```
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.85 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.outputs = 6
        self.model = Linear_QNet(85, 256, 256, self.outputs)
        self.trainer = QTrainer(self.model, LR, self.gamma)
```

You can also adjust the reward heuristics in game.py, for information such as <b>collisions</b>, whether the car <b>ate the food pellet</b>, and whether the car goes <b>towards or away from the food pellet</b>.

```
if self.is_collision() or self.frame_iteration > 1500:
            game_over = True
            reward = -15
            return reward, game_over

     if self.ate_food():
            reward = 15
            self.place_food()
     else: 
            dst = distance(self.car.front, self.food)
            if dst < self.min_dist:
                self.min_dist = dst 
                reward = 2
            else:
                reward = -5 
```

Additionally, the architecture of the neural network can be adjusted in the model.py file. It is made using PyTorch and by default is set to 2 fully-connected layers with a ReLU non-linearity modifying each layer.

<br>
<h2></b>5. Credits and Acknowledgements</b></h2>

This work would not have been possible without the decades of AI research and open-source projects that are available for students to learn. 

[1] This project was inspired by the Snake Game Reinforcement Learning tutorial produced by Patrick Loeber:

&emsp; &emsp; https://www.youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV

&emsp;&emsp;  https://github.com/patrickloeber/snake-ai-pytorch

[2] Additionally, the Q-learning video form the YouTube channel “The Computer Scientist” was extremely formative in my understanding of Q-learning and Reinforcement Learning frameworks such as OpenAI gym:

&emsp; &emsp; https://www.youtube.com/watch?v=wN3rxIKmMgE

[3] Finally, a general overview of Markov decision processes and search heuristics was derived from the Fall 2018 iteration of CS 188 at UC Berkeley lecture videos and projects:

&emsp; &emsp; https://inst.eecs.berkeley.edu/~cs188/fa18/

This project is also open-sourced for educational purposes.






