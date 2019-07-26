import tensorflow as tf
from layers import *

class Began(object):
    def __init__(self):
        self.batch_size = 16
        self.noise_dim = 128
        self.image_size = 64
        self.image_depth = 3
        self.num_filters = 128
    
    def initInputs(self):
        x = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.image_depth], name='input_data')
        z = tf.placeholder(tf.float32, [None, self.noise_dim], name='input_noise')
        lr = tf.placeholder(tf.float32, [], name='learning_rate')
        kt = tf.placeholder(tf.float32, [], name='equilibrium_term')
        return x, z, lr, kt
    
    def decoder(self, input, scope_name, reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()

            h0 = dense_layer(input_layer=input, units = 8 * 8 * self.num_filters, scope='dec_h0')
            h0 = tf.reshape(h0, [-1, 8, 8, self.num_filters])

            conv1 = conv_layer(input_layer=h0, layer_depth=self.num_filters, scope='dec1')
            conv1 = tf.nn.elu(conv1)
            conv2 = conv_layer(input_layer=conv1, layer_depth=self.num_filters, scope='dec2')
            conv2 = tf.nn.elu(conv2)

            upsample1 = upsample(conv=conv2, size=[16,16])
            conv3 = conv_layer(input_layer=upsample1, layer_depth=self.num_filters, scope='dec3')
            conv3 = tf.nn.elu(conv3)
            conv4 = conv_layer(input_layer=conv3, layer_depth=self.num_filters, scope='dec4')
            conv4 = tf.nn.elu(conv4)

            upsample2 = upsample(conv=conv4, size=[32,32])
            conv5 = conv_layer(input_layer=upsample2, layer_depth=self.num_filters, scope='dec5')
            conv5 = tf.nn.elu(conv5)
            conv6 = conv_layer(input_layer=conv5, layer_depth=self.num_filters, scope='dec6')
            conv6 = tf.nn.elu(conv6)

            upsample3 = upsample(conv=conv6, size=[64,64])
            conv7 = conv_layer(input_layer=upsample3, layer_depth=self.num_filters, scope='dec7')
            conv7 = tf.nn.elu(conv7)
            conv8 = conv_layer(input_layer=conv7, layer_depth=self.num_filters, scope='dec8')
            conv8 = tf.nn.elu(conv8)

            conv9 = conv_layer(input_layer=conv8, layer_depth=3, scope='decoder_image')
            decoder_output = tf.nn.tanh(conv9)
            return decoder_output

    def encoder(self, images, scope_name, reuse=False):

        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()

            organized_images = tf.reshape(images, [-1, self.image_size, self.image_size, 3])
                
            conv0 = conv_layer(input_layer=organized_images, layer_depth=self.num_filters, scope='enc0')
            conv0 = tf.nn.elu(conv0)
            conv1 = conv_layer(input_layer=conv0, layer_depth=self.num_filters, scope='enc1')
            conv1 = tf.nn.elu(conv1)
            conv2 = conv_layer(input_layer=conv1, layer_depth=self.num_filters, scope='enc2')
            conv2 = tf.nn.elu(conv2)

            # sub1 = subsample(conv=conv2)
            sub1 = strided_conv_subsample(conv2, self.num_filters, scope='sub1')
            conv3 = conv_layer(input_layer=sub1, layer_depth=self.num_filters*2, scope='enc3')
            conv3 = tf.nn.relu(conv3)
            conv4 = conv_layer(input_layer=conv3, layer_depth=self.num_filters*2, scope='enc4')
            conv4 = tf.nn.elu(conv4)

            # sub2 = subsample(conv=conv4)
            sub2 = strided_conv_subsample(conv4, self.num_filters*2, scope='sub2')
            conv5 = conv_layer(input_layer=sub2, layer_depth=self.num_filters*3, scope='enc5')
            tf.nn.elu(conv5)
            conv6 = conv_layer(input_layer=conv5, layer_depth=self.num_filters*3, scope='enc6')
            tf.nn.elu(conv6)

            # sub3 = subsample(conv=conv6)
            sub3 = strided_conv_subsample(conv6, self.num_filters*3, scope='sub3')
            conv7 = conv_layer(input_layer=sub3, layer_depth=self.num_filters*4, scope='enc7')
            tf.nn.elu(conv6)
            conv8 = conv_layer(input_layer=conv7, layer_depth=self.num_filters*4, scope='enc8')
            tf.nn.elu(conv8)

            encoder_output = dense_layer(input_layer=conv8, units=self.noise_dim, scope='encoder_output')
            # encoder_output = tf.nn.tanh(dense9)
            return encoder_output

    def generator(self, noise, reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()
            dec = self.decoder(noise, scope, reuse)
        return dec
    
    def discriminator(self, image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            enc = self.encoder(image, scope, reuse)
            dec = self.decoder(enc, scope, reuse)
        return dec

    def loss(self, x, z, kt):
        g_z = self.generator(z)
        d_x = self.discriminator(x)
        d_z = self.discriminator(g_z, reuse=True)

        d_x_loss = l1_loss(x, d_x)
        d_z_loss = l1_loss(g_z, d_z)
        dis_loss = d_x_loss - kt * d_z_loss
        gen_loss = d_z_loss
        return dis_loss, gen_loss, d_x_loss, d_z_loss

    def optimizer(self, dis_loss, gen_loss, learning_rate):
        dis_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

        adam = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5, beta2=0.999)
        dis_opt = adam.minimize(dis_loss, var_list=dis_vars)
        gen_opt = adam.minimize(gen_loss, var_list=gen_vars)
        return dis_opt, gen_opt

    def get_sample(self, num_samples=-1, reuse=True):
        if num_samples == -1:
            num_samples = self.batch_size
        noise = np.random.uniform(-1,1,size=[num_samples, self.noise_dim])
        noise = np.float32(noise)
        images = self.generator(noise, reuse)
        return images