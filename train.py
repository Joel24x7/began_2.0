import math
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_prep import load_data, prep_mnist_color
from model import Began


def train(model, epochs=100):

    #Setup file structure
    project_dir, logs_dir, samples_dir, models_dir = setup_dirs(project_num=2.3)
    data_dir = 'data'

    if not os.path.exists(data_dir):
        prep_mnist_color(data_dir)

    #Setup model
    x, z, lr, kt = model.initInputs()
    dis_loss, gen_loss, d_x_loss, d_z_loss = model.loss(x, z, kt)
    dis_opt, gen_opt = model.optimizer(dis_loss, gen_loss, lr)
    sample = model.get_sample()

    #Setup data
    data = load_data(data_dir)
    np.random.shuffle(data)
    start_time = time.time()

    #Setup inputs
    batch_size = model.batch_size
    num_batches_per_epoch = len(data) // batch_size

    #hyperparameters
    lrate = 0.0001
    lambda_kt = 0.001
    gamma = 0.4
    kt_var = 0.0
    epoch_drop = 3

    #Tensorboard
    m_global = d_x_loss + tf.abs(gamma * d_x_loss - d_z_loss)
    tf.summary.scalar('convergence', m_global)
    tf.summary.scalar('kt', kt)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    checkpoint_root = tf.train.latest_checkpoint(models_dir,latest_filename=None)

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./{}'.format(logs_dir), sess.graph)

        #Load previous training
        if checkpoint_root != None:
            saver.restore(sess, checkpoint_root)
        else:
            sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):

            learning_rate = lrate * math.pow(0.2, epoch+1 // epoch_drop)
            for batch_step in range(num_batches_per_epoch):

                #Prep batch
                start_data_batch = batch_step * batch_size
                end_data_batch = start_data_batch + batch_size
                batch_data = data[start_data_batch:end_data_batch, :, :, :]
                z_batch = np.random.uniform(-1,1,size=[batch_size, model.noise_dim])
                
                feed_dict={x: batch_data, z: z_batch, lr: learning_rate, kt: kt_var}
                _, real_loss = sess.run([dis_opt, d_x_loss], feed_dict=feed_dict)
                _, fake_loss = sess.run([gen_opt, d_z_loss], feed_dict=feed_dict)
                kt_var = kt_var + lambda_kt * (gamma * real_loss - fake_loss)
                convergence = real_loss + np.abs(gamma * real_loss - fake_loss)

                curr_step = epoch * num_batches_per_epoch + batch_step
                print('Time: {:.0} Epoch: {} {}/{} convergence: {:.4} kt: {:.4}'.format(time.time() - start_time, epoch, batch_step, num_batches_per_epoch, convergence, kt_var))

                if curr_step % 2500 == 0:
                    images = sess.run(sample)
                    images = (images + 1.0) / 2.0
                    for i in range(images.shape[0]):
                        tmpName = '{}/train_image{}.png'.format(samples_dir, i)
                        img = images[i, :, :, :]
                        plt.imshow(img)
                        plt.savefig(tmpName)
                
                if curr_step % 300 == 0:
                    summary = sess.run(merged, feed_dict)
                    train_writer.add_summary(summary, curr_step)
                    saver.save(sess, './{}/began'.format(models_dir), global_step = epoch)

def test(model):

    #Setup file structure
    project_dir, logs_dir, samples_dir, models_dir = setup_dirs(project_num=2.3)

    #Setup model
    _, z, _, _ = model.initInputs()
    sample = model.get_sample(reuse=False)
    saver = tf.train.Saver()
    checkpoint_root = tf.train.latest_checkpoint(models_dir,latest_filename=None)

    with tf.Session() as sess:

        if checkpoint_root != None:
            saver.restore(sess, checkpoint_root)
        else:
            sess.run(tf.global_variables_initializer())

        images = sess.run(sample)
        images = (images + 1.0) / 2.0

        for i in range(images.shape[0]):
            tmpName = '{}/test_image{}.png'.format(samples_dir, i)
            img = images[i, :, :, :]
            plt.imshow(img)
            plt.savefig(tmpName)


def setup_dirs(project_num):

    project_dir = 'mnist_began_{}'.format(project_num)
    logs_dir = '{}/logs_{}'.format(project_dir, project_num)
    samples_dir = '{}/results_{}'.format(project_dir, project_num)
    models_dir = '{}/models_{}'.format(project_dir, project_num)

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
        os.makedirs(logs_dir)
        os.makedirs(samples_dir)
        os.makedirs(models_dir)

    return project_dir, logs_dir, samples_dir, models_dir
