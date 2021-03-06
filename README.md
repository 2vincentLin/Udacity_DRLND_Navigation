# Udacity_DRLND_Navigation

### Environment detail

The project uses unity environment for the banana collector game, the state space is a vector whose length is 37, whereas action space is a discrete vector whose length is 4, which contains 4 actions: forward, backward, left and right. To successfully solves the problem is for the agent to receive the average reward over 100 episodes.


### dependencies

For this project, you'll need to install the following packages whether using pip or conda.

- pytorch(>=0.4)
- numpy(>=1.11)
- matplotlib 

You also need to download Udacity environment for banana collector game, the links are in below.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

download the environment you need and put banana.exe in the same folder as the notebook.

### How to use the notebook

#### for training your own agent and watch it play

Open the train_and_replay.ipynb, follow the instruction.

#### just want to know the performance between different implementation and algorithm

Open Report.ipynb

**Everytime you run the UnityEnvironment, you'll see it appear on your screen, you should run `env.close()` before shuting down the notebook, otherwise you might have to restart the os to restart the UnityEnvironment.**
