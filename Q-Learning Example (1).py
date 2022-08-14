#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:


pip install cmake 'gym[atari]' scipy


# In[14]:


import gym

env = gym.make("Taxi-v3").env

env.reset()


# In[16]:


import numpy as np
q_table = np.zeros([env.observation_space.n,env.action_space.n])


# In[19]:


get_ipython().run_cell_magic('time', '', 'import random\nfrom IPython.display import clear_output\nalpha = 0.1\ngamma = 0.6\nepsilon = 0.1\n\nall_epochs = []\nall_penalties = []\n\nfor i in range(1, 100001):\n    state = env.reset()\n    \n    epochs, penalties, reward, = 0,0,0\n    done = False\n    \n    while not done:\n        if random.uniform(0,1)<epsilon:\n            action = env.action_space.sample() #explore\n        else:\n            action = np.argmax(q_table[state])\n        \n        next_state, reward, done, info = env.step(action)\n        \n        old_value = q_table[state,action]\n        next_max = np.max(q_table[next_state])\n        \n        new_value = (1-alpha)*old_value+alpha*(reward + gamma*next_max)\n        q_table[state,action] = new_value\n        \n        if reward == -10:\n            penalties += 1\n            \n        state = next_state\n        epochs += 1\n        \n    if i%100 == 0:\n        clear_output(wait=True)\n        print(f"Epsiode: {i}")\nprint("Training finished.\\n")\n')


# In[20]:


q_table[328]


# In[23]:


total_epochs, total_penalties = 0,0
episodes = 100

for i in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0,0,0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        
        if reward == -10:
            penalties += 1
        
        epochs += 1
        
    total_penalties += penalties
    total_epochs += epochs
    
print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs/episodes}")
print(f"Average penalties per episode: {total_penalties/episodes}")


# In[ ]:




