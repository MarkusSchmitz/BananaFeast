[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

[banana]: https://media.giphy.com/media/1uPiL9Amv5zkk/giphy.gif
"Banana"

# Project 1: Navigation

## Project Details

This Project implements a Reinforcment Learning Agent that learns to collect Bananas in a game environment.
![Banana][banana]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  The Agent is thereby incentivised to collect as many yellow bananas as possible, while avoiding blue bananas (Because they are evil).

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The environment thereby has a discrete sction space and is episodic, as every episode has a fixed length of 100 steps. The Agent is considered to have solved the task if it is able to optain an average score of over 13 in an episode.

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. Place the extracted folder in the BananaFeast GitHub repository, in the base directory.

3. Install the dependancies as stated in the [DRLND repository](https://github.com/udacity/deep-reinforcement-learning#dependencies).

4. in the console activate your environment created in step 2 and install the seaborn package with:
```
pip install seaborn 
```
5. Open a juypter notebook inside the BananFeast directory and start the banana_agent notebook.

## Instructions

### Watching a smart Agent

Set the "train" parameter to false and run the whole notebook.


### Training the agent

Go to the cell where the parameter grid is created.
Add or remove parameter values until you are satisfied with the coverage.

Cange the "train" Parameter on top of the Notebook to "True" and run the whole notebook.
To train an optimized Agent use the default values.

### Evaluating the Agent

The best Values reached by the Agent were ~16 points on average with epsilon 0.01 at 1500 episodes.
In Watch mode the Agent scores between 8 and 21 points depending on the randomness of the environment.

### Known Issues

The Agent has been seen to have issues if two bananas are in a further distance at each side of the screen. This makes the Agent jerk to both sides and effectively paralyzes it.
