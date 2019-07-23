import math
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_prep import load_data
from model import Began


def train(model, epochs=100):

    print('\nStarting training\n')

    #Setup model
    x, z, lr, kt = model.initInputs()
    dis_loss, gen_loss, d_x_loss, d_z_loss = model.loss(x, z, kt)
    dis_opt, gen_opt = model.optimizer(dis_loss, gen_loss, lr)

    sample = model.get_sample()

    #Setup data
    data = load_data()
    np.random.shuffle(data)
    start_time = time.time()

    #Setup inputs
    batch_size = model.batch_size
    num_batches_per_epoch = len(data) // batch_size

    #hyperparameters
    lrate = 0.0001
    lambda_kt = 0.001
    gamma = 0.5
    kt_var = 0.0
    epoch_drop = 3

    #Tensorboard
    m_global = d_x_loss + tf.abs(gamma * d_x_loss - d_z_loss)
    tf.summary.scalar('convergence', m_global)
    tf.summary.scalar('kt', kt_var)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    print('\nTraining Setup Complete\n')

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./logs', sess.graph)
        sess.run(tf.global_variables_initializer())
        
        print('\nBeginning epoch iterations\n')
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
                _, gen_loss = sess.run([gen_opt, d_z_loss], feed_dict=feed_dict)
                kt_var = kt_var + lambda_kt * (gamma * real_loss - gen_loss)

                if batch_step % 300 == 0:
                    summary = sess.run(merged, feed_dict)
                    tmp_step = epoch * num_batches_per_epoch + batch_step
                    train_writer.add_summary(summary, tmp_step)
                    convergence = real_loss + np.abs(gamma * real_loss - gen_loss)

                    print('Epoch:', '%04d' % epoch, '%05d/%05d' % (batch_step, num_batches_per_epoch), 'convergence: {:.4}'.format(convergence))
                    
                    images = sess.run(sample)
                    for img in range(images.shape[0]):
                        tmpName = 'results/train_image{}.png'.format(img)
                        print(img.shape)
                        plt.imshow(img)
                        plt.savefig(tmpName)

                
            saver.save(sess, './models/began', global_step = epoch)

def test(model):

    print('\nStarting testing\n')

    #Setup model
    _, z, _, _ = model.initInputs()
    sample = model.get_sample()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        checkpoint_root = tf.train.latest_checkpoint('models',latest_filename=None)
        saver.restore(sess, checkpoint_root)

        images = sess.run(sample)
        for img in range(images):
            tmpName = 'results/train_image{}.png'.format(img)
            print(img.shape)
            plt.imshow(img)
            plt.savefig(tmpName)
