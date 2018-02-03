"""
  TensorFlow translation of the torch example found here (written by SeanNaren).
  https://github.com/SeanNaren/TorchQLearningExample

  Original keras example found here (written by Eder Santana).
  https://gist.github.com/EderSantana/c7222daa328f0e885093#file-qlearn-py-L164

  The agent plays a game of catch. Fruits drop from the sky and the agent can choose the actions
  left/stay/right to catch the fruit before it reaches the ground.
"""

import tensorflow as tf
import numpy as np
import random
import math
import os

# Parameters
epsilon = 1  # The probability of choosing a random action (in training). This decays as iterations increase. (0 to 1)
epsilonMinimumValue = 0.001 # The minimum value we want epsilon to reach in training. (0 to 1)
nbActions = 3 # The number of actions. Since we only have left/stay/right that means 3 actions.
epoch = 1001 # The number of games we want the system to run for.
hiddenSize = 100 # Number of neurons in the hidden layers.
maxMemory = 500 # How large should the memory be (where it stores its past experiences).
batchSize = 50 # The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.
gridSize = 10 # The size of the grid that the agent is going to play the game on.
nbStates = gridSize * gridSize # We eventually flatten to a 1d tensor to feed the network.
discount = 0.9 # The discount is used to force the network to choose states that lead to the reward quicker (0 to 1)  
learningRate = 0.2 # Learning Rate for Stochastic Gradient Descent (our optimizer).

# Create the base model.
X = tf.placeholder(tf.float32, [None, nbStates])
W1 = tf.Variable(tf.truncated_normal([nbStates, hiddenSize], stddev=1.0 / math.sqrt(float(nbStates))))
b1 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))  
input_layer = tf.nn.relu(tf.matmul(X, W1) + b1)
W2 = tf.Variable(tf.truncated_normal([hiddenSize, hiddenSize],stddev=1.0 / math.sqrt(float(hiddenSize))))
b2 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))
hidden_layer = tf.nn.relu(tf.matmul(input_layer, W2) + b2)
W3 = tf.Variable(tf.truncated_normal([hiddenSize, nbActions],stddev=1.0 / math.sqrt(float(hiddenSize))))
b3 = tf.Variable(tf.truncated_normal([nbActions], stddev=0.01))
output_layer = tf.matmul(hidden_layer, W3) + b3

# True labels
Y = tf.placeholder(tf.float32, [None, nbActions])

# Mean squared error cost function
cost = tf.reduce_sum(tf.square(Y-output_layer)) / (2*batchSize)

# Stochastic Gradient Decent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)


# Helper function: Chooses a random value between the two boundaries.
def randf(s, e):
  return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s;


# The environment: Handles interactions and contains the state of the environment
class CatchEnvironment():
  def __init__(self, gridSize):
    self.gridSize = gridSize
    self.nbStates = self.gridSize * self.gridSize
    self.state = np.empty(3, dtype = np.uint8) 

  # Returns the state of the environment.
  def observe(self):
    canvas = self.drawState()
    canvas = np.reshape(canvas, (-1,self.nbStates))
    return canvas

  def drawState(self):
    canvas = np.zeros((self.gridSize, self.gridSize))
    canvas[self.state[0]-1, self.state[1]-1] = 1  # Draw the fruit.
    # Draw the basket. The basket takes the adjacent two places to the position of basket.
    canvas[self.gridSize-1, self.state[2] -1 - 1] = 1
    canvas[self.gridSize-1, self.state[2] -1] = 1
    canvas[self.gridSize-1, self.state[2] -1 + 1] = 1    
    return canvas        

  # Resets the environment. Randomly initialise the fruit position (always at the top to begin with) and bucket.
  def reset(self): 
    initialFruitColumn = random.randrange(1, self.gridSize + 1)
    initialBucketPosition = random.randrange(2, self.gridSize + 1 - 1)
    self.state = np.array([1, initialFruitColumn, initialBucketPosition]) 
    return self.getState()

  def getState(self):
    stateInfo = self.state
    fruit_row = stateInfo[0]
    fruit_col = stateInfo[1]
    basket = stateInfo[2]
    return fruit_row, fruit_col, basket

  # Returns the award that the agent has gained for being in the current environment state.
  def getReward(self):
    fruitRow, fruitColumn, basket = self.getState()
    if (fruitRow == self.gridSize - 1):  # If the fruit has reached the bottom.
      if (abs(fruitColumn - basket) <= 1): # Check if the basket caught the fruit.
        return 1
      else:
        return -1
    else:
      return 0

  def isGameOver(self):
    if (self.state[0] == self.gridSize - 1): 
      return True 
    else: 
      return False 

  def updateState(self, action):
    if (action == 1):
      action = -1
    elif (action == 2):
      action = 0
    else:
      action = 1
    fruitRow, fruitColumn, basket = self.getState()
    newBasket = min(max(2, basket + action), self.gridSize - 1) # The min/max prevents the basket from moving out of the grid.
    fruitRow = fruitRow + 1  # The fruit is falling by 1 every action.
    self.state = np.array([fruitRow, fruitColumn, newBasket])

  #Action can be 1 (move left) or 2 (move right)
  def act(self, action):
    self.updateState(action)
    reward = self.getReward()
    gameOver = self.isGameOver()
    return self.observe(), reward, gameOver, self.getState()   # For purpose of the visual, I also return the state.


# The memory: Handles the internal memory that we add experiences that occur based on agent's actions,
# and creates batches of experiences based on the mini-batch size for training.
class ReplayMemory:
  def __init__(self, gridSize, maxMemory, discount):
    self.maxMemory = maxMemory
    self.gridSize = gridSize
    self.nbStates = self.gridSize * self.gridSize
    self.discount = discount
    canvas = np.zeros((self.gridSize, self.gridSize))
    canvas = np.reshape(canvas, (-1,self.nbStates))
    self.inputState = np.empty((self.maxMemory, 100), dtype = np.float32)
    self.actions = np.zeros(self.maxMemory, dtype = np.uint8)
    self.nextState = np.empty((self.maxMemory, 100), dtype = np.float32)
    self.gameOver = np.empty(self.maxMemory, dtype = np.bool)
    self.rewards = np.empty(self.maxMemory, dtype = np.int8) 
    self.count = 0
    self.current = 0

  # Appends the experience to the memory.
  def remember(self, currentState, action, reward, nextState, gameOver):
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.inputState[self.current, ...] = currentState
    self.nextState[self.current, ...] = nextState
    self.gameOver[self.current] = gameOver
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.maxMemory

  def getBatch(self, model, batchSize, nbActions, nbStates, sess, X):
    
    # We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
    # batch we can (at the beginning of training we will not have enough experience to fill a batch).
    memoryLength = self.count
    chosenBatchSize = min(batchSize, memoryLength)

    inputs = np.zeros((chosenBatchSize, nbStates))
    targets = np.zeros((chosenBatchSize, nbActions))

    # Fill the inputs and targets up.
    for i in xrange(chosenBatchSize):
      if memoryLength == 1:
        memoryLength = 2
      # Choose a random memory experience to add to the batch.
      randomIndex = random.randrange(1, memoryLength)
      current_inputState = np.reshape(self.inputState[randomIndex], (1, 100))

      target = sess.run(model, feed_dict={X: current_inputState})
      
      current_nextState =  np.reshape(self.nextState[randomIndex], (1, 100))
      current_outputs = sess.run(model, feed_dict={X: current_nextState})      
      
      # Gives us Q_sa, the max q for the next state.
      nextStateMaxQ = np.amax(current_outputs)
      if (self.gameOver[randomIndex] == True):
        target[0, [self.actions[randomIndex]-1]] = self.rewards[randomIndex]
      else:
        # reward + discount(gamma) * max_a' Q(s',a')
        # We are setting the Q-value for the action to  r + gamma*max a' Q(s', a'). The rest stay the same
        # to give an error of 0 for those outputs.
        target[0, [self.actions[randomIndex]-1]] = self.rewards[randomIndex] + self.discount * nextStateMaxQ

      # Update the inputs and targets.
      inputs[i] = current_inputState
      targets[i] = target

    return inputs, targets

    
def main(_):
  print("Training new model")

  # Define Environment
  env = CatchEnvironment(gridSize)

  # Define Replay Memory
  memory = ReplayMemory(gridSize, maxMemory, discount)

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()
  
  winCount = 0
  with tf.Session() as sess:   
    tf.initialize_all_variables().run() 

    for i in xrange(epoch):
      # Initialize the environment.
      err = 0
      env.reset()
     
      isGameOver = False

      # The initial state of the environment.
      currentState = env.observe()
            
      while (isGameOver != True):
        action = -9999  # action initilization
        # Decides if we should choose a random action, or an action from the policy network.
        global epsilon
        if (randf(0, 1) <= epsilon):
          action = random.randrange(1, nbActions+1)
        else:          
          # Forward the current state through the network.
          q = sess.run(output_layer, feed_dict={X: currentState})          
          # Find the max index (the chosen action).
          index = q.argmax()
          action = index + 1     

        # Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
        if (epsilon > epsilonMinimumValue):
          epsilon = epsilon * 0.999
        
        nextState, reward, gameOver, stateInfo = env.act(action)
            
        if (reward == 1):
          winCount = winCount + 1

        memory.remember(currentState, action, reward, nextState, gameOver)
        
        # Update the current state and if the game is over.
        currentState = nextState
        isGameOver = gameOver
                
        # We get a batch of training data to train the model.
        inputs, targets = memory.getBatch(output_layer, batchSize, nbActions, nbStates, sess, X)
        
        # Train the network which returns the error.
        _, loss = sess.run([optimizer, cost], feed_dict={X: inputs, Y: targets})  
        err = err + loss

      print("Epoch " + str(i) + ": err = " + str(err) + ": Win count = " + str(winCount) + " Win ratio = " + str(float(winCount)/float(i+1)*100))
    # Save the variables to disk.
    save_path = saver.save(sess, os.getcwd()+"/model.ckpt")
    print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
  tf.app.run()

