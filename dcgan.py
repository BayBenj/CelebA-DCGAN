
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
from tqdm import trange, tqdm
from skimage import io, transform

def loadCelebAData(size=5):
	files_in = os.listdir('img_align_celeba')
	files = np.random.choice(files_in, size=size)
	images = []
	for f in tqdm(files):
			images.append(transform.resize(io.imread('img_align_celeba/' + f), (64,64,3), mode='constant'))#(178,218,3)
	result = np.asarray(images)
	return result

def lrelu(x, alpha=0.2):
	return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def dcgan_generator(z, name = "g_generator"):
	with tf.variable_scope(name) as scope:
		# if reuse==True:
		if scope.trainable_variables():
			scope.reuse_variables()
		z = tf.reshape(z, [1, 100], name="g_reshape0")
		reshape1 = tf.contrib.layers.fully_connected(z, 16384, activation_fn=None, biases_initializer=tf.contrib.layers.variance_scaling_initializer(), weights_initializer=tf.contrib.layers.variance_scaling_initializer())
		reshape2 = tf.reshape(reshape1, [1, 4, 4, 1024], name="g_reshape2")
		g0 = tf.layers.conv2d_transpose(reshape2, 512, 3, strides = [2,2], padding = "SAME", use_bias=True, name = "g_conv0", kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), activation=tf.nn.relu)
		print("g0: {}".format(g0.shape))
		g1 = tf.layers.conv2d_transpose(g0, 256, 3, strides = [2,2], padding = "SAME", use_bias=True, name = "g_conv1", kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), activation=tf.nn.relu)
		print("g1: {}".format(g1.shape))
		g2 = tf.layers.conv2d_transpose(g1, 128, 3, strides = [2,2], padding = "SAME", use_bias=True, name = "g_conv2", kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), activation=tf.nn.relu)
		print("g2: {}".format(g2.shape))
		g3 = tf.layers.conv2d_transpose(g2, 3, 3, strides = [2,2], padding = "SAME", use_bias=True, name = "g_conv3", kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), activation=tf.nn.sigmoid)
		print("g3: {}".format(g3.shape))
		return g3

def dcgan_discriminator(input_image, reuse=False, name = "d_discriminator"):
	with tf.variable_scope(name) as scope:
		# if reuse==True:
		if scope.trainable_variables():
			scope.reuse_variables()
		print("input_image: {}".format(input_image.shape))
		h0 = tf.layers.conv2d(input_image, 64, 3, name="d_conv0", reuse=reuse, kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), activation=None, padding="SAME")
		print("h0: {}".format(h0.shape))
		dr0 = lrelu(h0)
		print("dr0: {}".format(dr0.shape))
		h1 = tf.layers.conv2d(dr0, 32, 3, name="d_conv1", reuse=reuse, kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), activation=None, padding="SAME")
		dr1 = lrelu(h1)
		print("h1: {}".format(dr1.shape))
		h2 = tf.layers.conv2d(dr1, 8, 3, name="d_conv2", reuse=reuse, kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), activation=None, padding="SAME")
		dr2 = lrelu(h2)
		print("h2: {}".format(dr2.shape))
		h3 = tf.layers.conv2d(dr2, 2, 3, name="d_conv3", reuse=reuse, kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), activation=None, padding="SAME")
		dr3 = lrelu(h3)
		print("h3: {}".format(dr3.shape))
		dr3 = tf.reshape(dr3,[1,64*64*2])
		scalar = tf.contrib.layers.fully_connected(dr3, 1, reuse=reuse, activation_fn=None, biases_initializer=tf.contrib.layers.variance_scaling_initializer(), weights_initializer=tf.contrib.layers.variance_scaling_initializer(), scope=scope)
		return scalar

def get_z():
	return np.random.uniform(-1, 1, 100)

EPOCHS = 1000
C = 0.0001
BETA_1 = 0
BETA_2 = 0.9
CELEBA_IMAGE_DIMS = [1,64,64,3]
N_CRITIC = 5
LAMBDA = 10
DATA_SIZE = 20
true_images = loadCelebAData(DATA_SIZE)
print("Loaded {} true images!".format(len(true_images)))

true_img = tf.placeholder(tf.float32, CELEBA_IMAGE_DIMS)
tf_label = tf.placeholder(tf.int64)
z_node = tf.placeholder(tf.float32, [100])
epsilon = tf.placeholder(tf.float32, shape = [])

with tf.name_scope("d_discriminator_loss") as scope:
	img_attempt = dcgan_generator(z_node)
	print("img_attempt: {}".format(img_attempt.shape))
	x_hat = epsilon * true_img + (1 - epsilon) * img_attempt
	print("x_hat: {}".format(x_hat.shape))
	one = dcgan_discriminator(img_attempt)
	two = dcgan_discriminator(true_img, reuse=True)
	three = LAMBDA * ((tf.norm(tf.gradients(dcgan_discriminator(x_hat, reuse=True), x_hat)) - 1) ** 2)
	disc_loss = one - two + three

with tf.name_scope("g_generator_loss") as scope:
	gen_loss = -dcgan_discriminator(img_attempt, reuse=True)

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if "d_" in var.name]
g_vars = [var for var in t_vars if "g_" in var.name]

with tf.name_scope("d_discriminator_train") as scope:
	disc_train = tf.train.AdamOptimizer(C, BETA_1, BETA_2).minimize(disc_loss, var_list=d_vars)

with tf.name_scope("g_generator_train") as scope:
	gen_train = tf.train.AdamOptimizer(C, BETA_1, BETA_2).minimize(gen_loss, var_list=g_vars)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	writer = tf.summary.FileWriter("/Users/Benjamin/Desktop/byu/Semester 8/CS501R/lab7", sess.graph)
	sess.run(init)

	for epoch in range(EPOCHS):
		i = 0
		for true_image in true_images:
			true_image = np.reshape(true_image, CELEBA_IMAGE_DIMS)
			z_disc = get_z()
			eps = np.random.rand()
			w, d_loss = sess.run([disc_train, disc_loss], feed_dict = {z_node: z_disc, true_img: true_image, epsilon: eps}) #find disc loss, then update disc
			# print("Disc loss: {}".format(d_loss))
			if i % N_CRITIC == 0:
				z_gen = get_z()
				theta, g_loss = sess.run([gen_train, gen_loss], feed_dict = {z_node: z_gen}) #find gen loss, then update gen
				# print("Gen loss: {}".format(g_loss))
			i += 1
		if epoch % 1 == 0:
			z_test = get_z()
			out_image = sess.run(img_attempt, feed_dict = {z_node: z_test})
			out_image = np.reshape(out_image, [64, 64, 3])
			plt.imsave("final{}.png".format(epoch), out_image)

	z1 = get_z()
	z2 = get_z()
	eps = np.arange(11)/10.0
	print(eps)
	i = 0
	for a in eps:
		tmp = a * z1 + (1 - a) * z2
		interp_image = sess.run(img_attempt, feed_dict={z_node: tmp})
		interp_image = np.reshape(interp_image, [64, 64, 3])
		plt.imsave("interp{}.png".format(i), interp_image)
		i += 1

	writer.close
