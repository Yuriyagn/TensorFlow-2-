# coding:utf-8
# @Version  : 1.0
# @Author   : yuri
# @File     : 神经网络.py
# @Time     : 2024/7/12 11:16
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import tensorflow_probability as tfp



binary_policy = tfp.distributions.Bernoulli(probs=0.5)
for i in range(5):
    action = binary_policy.sample(1)
    print("Action: ", action)

sample_actions = binary_policy.sample(500)
sns.displot(sample_actions)
plt.show()

action_dim = 4  # Dimension of the discrete action space
action_probability = [0.25, 0.25, 0.25, 0.25]
discrete_policy = tfp.distributions.Multinomial(
    probs=action_probability, total_count=1)
for i in range(5):
    action = discrete_policy.sample(1)
    print(action)
sns.displot(discrete_policy.sample(2).numpy())
plt.show()