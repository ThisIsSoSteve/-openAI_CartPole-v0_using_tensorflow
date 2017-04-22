import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #hide CUDA logging
import gym
#import random
import numpy as np
import tensorflow as tf #running the GPU version
from statistics import median, mean
from collections import Counter

score_requirement = 50
initial_games = 100000
hm_epochs = 10
batch_size = 100

#save model *think it tries to save to a CUDA path with out the full path
model_save_path = 'E:/Neural Network Projects/Python/tensorflow_open_AI_gym_cart_pole/tensorflow_open_AI_gym_cart_pole/temp/my_cart_poleV0_model.ckpt'

env = gym.make('CartPole-v0')
env.reset()

#env.seed(0)

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

input_size = env.observation_space.shape[0] #4
output_size = env.action_space.n #2

print("Observation shape:", input_size)
print("Action shape:", output_size)
#print("Highest Observation:", env.observation_space.high)
#print("Lowest Observation:", env.observation_space.low)

#input
x = tf.placeholder(tf.float32)
#output
y = tf.placeholder(tf.float32)

#network map [number of nodes, activation]
network_map = np.array([[32, 1],[4, 1],[output_size, -1]])
#network_map = tf.constant([[16, 0], [32, 0],[2, -1]])
network_dictionary = {}

def neural_network_model_generator(layer_feed):

    #layer_feed = data
    layer_size = input_size
    
    #rows = network_map.get_shape()[0].value
    rows = np.shape(network_map)[0]
    
    for i in range(rows):

        layer_number = i + 1

        network_dictionary['layer{}_weights'.format(layer_number)]  = tf.Variable(tf.random_normal([layer_size, network_map[i][0]]))
        network_dictionary['layer{}_biases'.format(layer_number)]  = tf.Variable(tf.random_normal([network_map[i][0]]))

        layer_feed = tf.matmul(layer_feed, network_dictionary['layer{}_weights'.format(layer_number)]) + network_dictionary['layer{}_biases'.format(layer_number)]
        layer_size = network_map[i][0]

        if(network_map[i][1] != -1):
            if(network_map[i][1] == 0):
                network_dictionary['layer{}_activation'.format(layer_number)] = tf.nn.relu(layer_feed)
            if(network_map[i][1] == 1):
                network_dictionary['layer{}_activation'.format(layer_number)] = tf.nn.sigmoid(layer_feed)

            layer_feed = network_dictionary['layer{}_activation'.format(layer_number)]
        
    return layer_feed

def train_neural_network(training_data):

    train_x = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]))
    #x1 = training_data[:,0]
    train_y = [i[1] for i in training_data]

    #print('x shape:', np.shape(train_x))
    #print('y shape:', np.shape(train_y))
    #x shape = [n, 4]
    #y shape = [n, 2]

    #prediction = neural_network_model(x)
    prediction = neural_network_model_generator(x)
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
    #prediction = neural_network_model(x)
    prediction = neural_network_model_generator(x)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_save_path)

        #print(sess.run(network_dictionary['layer1_weights']))
        #print(abs(sess.run(network_dictionary['layer1_weights']))>0.5 )
        #print(abs(sess.run(network_dictionary['layer2_weights']))>0.5 )

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
                    #action = np.argmax(action)
                    #print(sess.run(network_dictionary['layer1_activation'], feed_dict={x:prev_obs.reshape(-1, len(prev_obs))}))

                    #predictedAction = sess.run(tf.arg_max(predictedAction),1)
                    #action = sess.run(tf.argmax(prediction.eval(feed_dict={x:prev_obs.reshape(-1, len(prev_obs))}),1)[0])
                    #print('prediction action:', action)

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
#train_neural_network(np.load('training.npy'))
use_neural_network()