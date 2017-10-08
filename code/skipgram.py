#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
Author: Satoshi Tsutsui <stsutsui@indiana.edu>
I consluted the public implementation of word2vec:https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/b95dcdf7bd3efa0f6ff3c28e01d3d42d2084b35c/examples/04_word2vec_no_frills.py
"""



import os
import numpy as np
import tensorflow as tf
import json

def build_model(BATCH_SIZE,VOCAB_SIZE,EMBED_SIZE,NUM_SAMPLED):
    '''
    Build the model (i.e. computational graph) and return the placeholders (input and output) and the loss 
    '''
    # define the placeholders for input and output
    with tf.name_scope('data'):
        center_node = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_node')
        context_node = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='context_node')
        negative_samples = (tf.placeholder(tf.int32, shape=[NUM_SAMPLED], name='negative_samples'),
            tf.placeholder(tf.float32, shape=[BATCH_SIZE,1], name='true_expected_count'),
            tf.placeholder(tf.float32, shape=[NUM_SAMPLED], name='sampled_expected_count'))

    #https://github.com/tensorflow/tensorflow/blob/624bcfe409601910951789325f0b97f520c0b1ee/tensorflow/python/ops/nn_impl.py#L943-L946
    # Sample the negative labels.
    #   sampled shape: [num_sampled] tensor
    #   true_expected_count shape = [batch_size, 1] tensor
    #   sampled_expected_count shape = [num_sampled] tensor

    # Assemble this part of the graph on the CPU. You can change it to GPU if you have GPU
    # define weights. In word2vec, it's actually the weights that we care about

    with tf.name_scope('embedding_matrix'):
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), 
                            name='embed_matrix')

    # define the inference
    with tf.name_scope('loss'):
        embed = tf.nn.embedding_lookup(embed_matrix, center_node, name='embed')

        #construct variables for NCE loss
        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
                                                    stddev=1.0 / (EMBED_SIZE ** 0.5)), 
                                                    name='nce_weight')
        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

        # define loss function to be NCE loss function
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                            biases=nce_bias, 
                                            labels=context_node, 
                                            inputs=embed,
                                            sampled_values = negative_samples, 
                                            num_sampled=NUM_SAMPLED, 
                                            num_classes=VOCAB_SIZE), name='loss')

        loss_summary = tf.summary.scalar("loss_summary", loss)

    return center_node,context_node,negative_samples,loss

def traning_op(loss,LEARNING_RATE):
    '''
    Return optimizer
    define one step for SGD
    '''
    #define optimizer
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    return optimizer

def train(center_node_placeholder,context_node_placeholder,negative_samples_placeholder,loss,dataset,optimizer,NUM_EPOCHS,BATCH_SIZE,NUM_SAMPLED,care_type,LOG_DIRECTORY,LOG_INTERVAL,MAX_KEEP_MODEL):
    '''
    tensorflow training loop
    define SGD trining
    *epoch index starts from 1! not 0.
    '''
    care_type = True if care_type==1 else False

    # For tensorboad  
    merged = tf.summary.merge_all()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(max_to_keep=MAX_KEEP_MODEL)#tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_loss = 0.0 # we use this to calculate late average loss in the last LOG_INTERVAL steps
        writer = tf.summary.FileWriter(LOG_DIRECTORY, sess.graph)
        global_iteration = 0
        iteration = 0
        while (dataset.epoch <= NUM_EPOCHS):
            current_epoch=dataset.epoch
            center_node_batch,context_node_batch  = dataset.get_batch(BATCH_SIZE)
            negative_samples  = dataset.get_negative_samples(pos_index=context_node_batch[0],num_negatives=NUM_SAMPLED,care_type=care_type)
            context_node_batch = context_node_batch.reshape((-1,1))
            loss_batch, _ ,summary_str = sess.run([loss, optimizer,merged], 
                                    feed_dict={
                                    center_node_placeholder:center_node_batch,
                                    context_node_placeholder:context_node_batch,
                                    negative_samples_placeholder: negative_samples
                                    })
            writer.add_summary(summary_str,global_iteration)
            total_loss += loss_batch

            # print(loss_batch)

            iteration+=1
            global_iteration+=1

            if LOG_INTERVAL > 0:
                if global_iteration % LOG_INTERVAL == 0:
                    print('Average loss: {:5.1f}'.format(total_loss / LOG_INTERVAL))
                    total_loss = 0.0
                    #save model
                    model_path=os.path.join(LOG_DIRECTORY,"/model_temp.ckpt")
                    save_path = saver.save(sess, model_path)
                    print("Model saved in file: %s" % save_path)

            if dataset.epoch - current_epoch > 0:
                print("Epoch %d end"% current_epoch)
                dataset.shffule()
                #save model
                model_path=os.path.join(LOG_DIRECTORY,"model_epoch%d.ckpt"%dataset.epoch)
                save_path = saver.save(sess, model_path)
                print("Model saved in file: %s" % save_path)
                print('Average loss in this epoch: {:5.1f}'.format(total_loss / iteration))
                total_loss = 0.0
                iteration=0

        model_path=os.path.join(LOG_DIRECTORY,"model_final.ckpt")
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
        writer.close()

        print("Save final embeddings as numpy array")
        np_node_embeddings = tf.get_default_graph().get_tensor_by_name("embedding_matrix/embed_matrix:0")
        np_node_embeddings = sess.run(np_node_embeddings)
        np.savez(os.path.join(LOG_DIRECTORY,"node_embeddings.npz"),np_node_embeddings)

        with open(os.path.join(LOG_DIRECTORY,"index2nodeid.json"), 'w') as f:  
            json.dump(dataset.index2nodeid, f, sort_keys=True, indent=4)  

if __name__ == '__main__':
    pass
    #test code  