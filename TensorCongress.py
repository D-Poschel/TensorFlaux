# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:14:21 2017

@author: Daniel
"""

import tensorflow as tf
import numpy as np
import random

def main():
    
    data = []
    labels = []

    with open("house-votes-84.data","r") as input_file:
        for line in input_file:
            if(len(line.strip()) == 0):
                continue
            badVariableName1 = line.strip().split(",")
            badVaribleName2 = badVariableName1[1:17]
            badVaribleName4 = badVariableName1[0]
            
            for  x, s in enumerate(badVaribleName2):
                if(s == 'y'):
                    badVaribleName2[x] = 1
                if(s == 'n'):
                    badVaribleName2[x] = -1
                if(s == '?'):
                    badVaribleName2[x] = 0
                    
                if(badVaribleName4 == 'democrat'):
                    badVaribleName4 = 1
                if(badVaribleName4 == 'republican'):
                    badVaribleName4 = 0
            
            
            data.append(badVaribleName2)
            labels.append(badVaribleName4)
            
    print(data)
    print(labels)
    
    dataset = list(zip(data, labels))
    random.shuffle(dataset)
    test_length = int(len(dataset) * 0.67)

    print("test_length", test_length)
    train_dataset = dataset[:test_length]
    test_dataset = dataset[test_length:]
    
    #16 inputs and 2 possible outputs
    input_size = 16
    out_size = 2
    num_nodes = 168
    
    inputs = tf.placeholder("float", shape=[None, input_size])
    labels = tf.placeholder("int32", shape=[None])

    weights1 = tf.get_variable("weight1", shape=[input_size, num_nodes], initializer=tf.contrib.layers.xavier_initializer())
    bias1 = tf.get_variable("bias1", shape=[num_nodes], initializer=tf.constant_initializer(value=0.0))
    
    layer1 = tf.nn.relu(tf.matmul(inputs, weights1) + bias1)

    weights2 = tf.get_variable("weight2", shape=[num_nodes, num_nodes], initializer=tf.contrib.layers.xavier_initializer())
    bias2 = tf.get_variable("bias2", shape=[num_nodes], initializer=tf.constant_initializer(value=0.0))
    
    layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + bias2)

    weights3 = tf.get_variable("weight3", shape=[num_nodes, out_size], initializer=tf.contrib.layers.xavier_initializer())
    bias3 = tf.get_variable("bias3", shape=[out_size], initializer=tf.constant_initializer(value=0.0))
    
    outputs = tf.matmul(layer2, weights3) + bias3
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, out_size), logits=outputs))
    train = tf.train.AdamOptimizer().minimize(loss)

    predictions = tf.argmax(tf.nn.softmax(outputs), axis=1)
    
    with tf.Session() as sess:
        
        sess.run(tf.initialize_all_variables())

        for epoch in range(5000):
            batch = random.sample(train_dataset, 25)
            inputs_batch, labels_batch = zip(*batch)
            loss_output, prediction_output, _ = sess.run([loss, predictions, train], feed_dict={inputs: inputs_batch, labels: labels_batch})

            # print("Prediction output", prediction_output)
            # print("Labels batch", labels_batch)

            # accuracy = np.mean(labels_batch == prediction_output)

            # print("train", "loss", loss_output, "accuracy", accuracy)

        # test our trained model with test data
        batch = random.sample(test_dataset, 100)
        inputs_batch, labels_batch = zip(*batch)
        loss_output, prediction_output, _ = sess.run([loss, predictions, train], feed_dict={inputs: inputs_batch, labels: labels_batch})
        accuracy = np.mean(labels_batch == prediction_output)

        print("Prediction output: ", prediction_output)
        print("Labels batch: ", labels_batch)
        print("Loss: ", loss_output)
        print("Accuracy: ", accuracy)

            
            
if __name__ == "__main__":
    main()