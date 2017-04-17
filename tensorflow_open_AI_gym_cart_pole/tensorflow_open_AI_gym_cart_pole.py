import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #hide CUDA logging
import gym
#from gym import envs
#from gym.envs.registration import EnvSpec
#import random
import numpy as np
import tensorflow as tf #running the GPU version
from statistics import median, mean
from collections import Counter

score_requirement = 50
initial_games = 1000
hm_epochs = 5
batch_size = 500

#save model *think it tries to save to a CUDA path with out the full path
model_save_path = 'E:/Neural Network Projects/Python/tensorflow_open_AI_gym_cart_pole/tensorflow_open_AI_gym_cart_pole/temp/my_cart_poleV0_model.ckpt'

env = gym.make('CartPole-v0')
env.reset()

#print(envs.registry.all())
#print(EnvSpec('CartPole-v0'))

#def random_game():
#    for episode in range(5):
#        env.reset()
#        for t in range(goal_step):
#            env.render()
#            action = env.action_space.sample()
#            observation, reward, done, info = env.step(action)
#            if done:
#                break

#random_game()
def initial_training_data():
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        while True:
            action = env.action_space.sample() #random.randrange(0, 2)
            observation, reward, done, info = env.step(action)
            #observation = [position of cart, velocity of cart, angle of pole, rotation rate of pole]

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward

            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('training.npy', training_data_save)

    print('Average accepted score: ', mean(accepted_scores))
    print('Median accepted score: ', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


n_nodes_hidden_layer1 = 16
n_nodes_hidden_layer2 = 32

input_size = 4
output_size = 2

#input
x = tf.placeholder(tf.float32)
#output
y = tf.placeholder(tf.float32)

 #define weights and biases dictionary
hidden_1_layer = {'weights':tf.Variable(tf.random_normal([input_size, n_nodes_hidden_layer1])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hidden_layer1]))}

hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hidden_layer1,n_nodes_hidden_layer2])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hidden_layer2]))}

output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hidden_layer2,output_size])),
                'biases': tf.Variable(tf.random_normal([output_size]))}
#multilayer_perceptron
def neural_network_model(data):

    #(input_data * weights) + biases
    layer1 = tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases']
    layer1 = tf.nn.relu(layer1) #activation function nn.sigmoid nn.tanh

    layer2 = tf.matmul(layer1, hidden_2_layer['weights']) + hidden_2_layer['biases']
    layer2 = tf.nn.relu(layer2)

    output = tf.matmul(layer2, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(training_data):

    train_x = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]))
    #x1 = training_data[:,0]
    train_y = [i[1] for i in training_data]

    #print('x shape:', np.shape(train_x))
    #print('y shape:', np.shape(train_y))
    #x shape = [n, 4]
    #y shape = [n, 2]

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                batch_start = i
                batch_end = i + batch_size

                batch_x = np.array(train_x[batch_start:batch_end])#slice input data
                batch_y = np.array(train_y[batch_start:batch_end])#slice labels (output)

                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        
        saver.save(sess, model_save_path)#global_step=hm_epochs
        
def use_neural_network():

    scores = []
    choices = []

    prediction = neural_network_model(x)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_save_path)

        for each_game in range(10):
            score = 0
            game_memory = []
            prev_obs = []
            env.reset()
            
            while True:
                #env.render()

                if len(prev_obs)==0:
                    action = env.action_space.sample()#random.randrange(0,2)
                else:
                    predictedAction = sess.run(prediction, feed_dict={x:prev_obs.reshape(-1, len(prev_obs))})
                    action = np.argmax(predictedAction)#tf.arg_max is really slow
                    #predictedAction = sess.run(tf.arg_max(predictedAction),1)
                    #action = sess.run(tf.argmax(prediction.eval(feed_dict={x:prev_obs.reshape(-1, len(prev_obs))}),1)[0])
                    #print('prediction action:', predictedAction)

                choices.append(action)

                new_observation, reward, done, info = env.step(action)

                prev_obs = new_observation
                game_memory.append([new_observation, action])
                score+=reward
                if done: break

            scores.append(score)
            print('game:', each_game + 1, 'score:', score)

    print('Average Score:',sum(scores)/len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
    #print(score_requirement)


#initial_training_data()
#my_saved_training_data = np.load('training.npy')
#train_neural_network(my_saved_training_data)

use_neural_network()