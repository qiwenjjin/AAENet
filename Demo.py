from __future__ import print_function
try:
    raw_input
except:
    raw_input = input
import numpy as np
from keras.layers import concatenate
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense
from keras.utils import plot_model
from keras.datasets import mnist
from keras.optimizers import Adam
from losses import SAD
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from datetime import datetime
from sklearn.manifold import TSNE
from absl import flags
from absl import app
from scipy.io import loadmat, savemat
from keras.callbacks import TensorBoard
import tensorflow as tf
np.set_printoptions(threshold=np.inf)


FLAGS = flags.FLAGS

# General
flags.DEFINE_bool("adversarial", True, "Use Adversarial Autoencoder or regular Autoencoder")
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("reconstruct", False, "Reconstruct image")
flags.DEFINE_bool("generate", False, "Generate image from latent")
flags.DEFINE_bool("generate_grid", False, "Generate grid of images from latent space (only for 2D latent)")
flags.DEFINE_bool("plot", False, "Plot latent space")
flags.DEFINE_integer("latent_dim", 5, "Latent dimension")

# Train
flags.DEFINE_integer("epochs", 500, "Number of training epochs")
flags.DEFINE_integer("train_samples", 3600, "Number of training samples from MNIST")
flags.DEFINE_integer("batchsize", 40, "Training batchsize")

# Test
flags.DEFINE_integer("test_samples", 3600, "Number of test samples from MNIST")
flags.DEFINE_list("latent_vec", None, "Latent vector (use with --generate flag)")


callbacks = [TensorBoard(
        log_dir='./logs'
    )
]
mat_contents=loadmat('Synthetic_new.mat')
x=mat_contents['Y']
E=mat_contents['E_init']
E_T=tf.transpose(E)
gaussian_dsitribution=loadmat('dsitribution.mat')
mean=gaussian_dsitribution['mean']
cov=gaussian_dsitribution['cov']
l_vca=0.01
l_2=0
use_bias=False
rand_x = np.random.RandomState(42)
rand_y = np.random.RandomState(42)

def E_reg(weight_matrix):
    print('weight_matrix',weight_matrix.shape)
    return l_vca*SAD(weight_matrix,E)+l_2*tf.reduce_mean(tf.matmul(tf.transpose(weight_matrix,perm=[1,0]),weight_matrix))
    
def create_model(input_dim, latent_dim, verbose=True, save_graph=False):

    autoencoder_input = Input(shape=(input_dim,))
    generator_input = Input(shape=(input_dim,))
    encoder = Sequential()
    encoder.add(Dense(intermediate_dim1, input_shape=(input_dim,), activation='relu',name='Dense_1'))
    encoder.add(Dense(intermediate_dim2, activation='relu',name='Dense_2'))
    encoder.add(Dense(latent_dim, activation='sigmoid',name='Dense_3'))

    ####################################################################
    decoder = Sequential()
    decoder.add(Dense(input_dim, input_shape=(latent_dim,), activation='linear', name='endmembers', use_bias=use_bias,
                      kernel_constraint=non_neg(), kernel_regularizer=E_reg, kernel_initializer=initializer))
    #print('weight_Dense_1',weight_Dense_1)
    if FLAGS.adversarial:
        discriminator = Sequential()
        discriminator.add(Dense(intermediate_dim2, input_shape=(latent_dim,), activation='relu'))
        discriminator.add(Dense(intermediate_dim1, activation='relu'))
        discriminator.add(Dense(1, activation='sigmoid'))

    autoencoder= Model(autoencoder_input,decoder(encoder(autoencoder_input)))
    autoencoder.compile(optimizer=Adam(lr=1e-4), loss=SAD)
    

    if FLAGS.adversarial:
        discriminator.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")
        discriminator.trainable = False
        generator = Model(generator_input, discriminator(encoder(generator_input)))
        generator.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")

    if verbose:
        print("Autoencoder Architecture")
        print(autoencoder.summary())
        if FLAGS.adversarial:
            print("Discriminator Architecture")
            print(discriminator.summary())
            print("Generator Architecture")
            print(generator.summary())

    if save_graph:
        plot_model(autoencoder, to_file="autoencoder_graph.png")
        if FLAGS.adversarial:
            plot_model(discriminator, to_file="discriminator_graph.png")
            plot_model(generator, to_file="generator_graph.png")

    if FLAGS.adversarial:
        return autoencoder, discriminator, generator, encoder, decoder
    else:
        return autoencoder, None, None, encoder, decoder

def train(n_samples, batch_size, n_epochs):
    autoencoder, discriminator, generator, encoder, decoder = create_model(input_dim=200, latent_dim=FLAGS.latent_dim)
    past = datetime.now()
    for epoch in np.arange(1, n_epochs + 1):
        autoencoder_losses = []
        if FLAGS.adversarial:
            discriminator_losses = []
            generator_losses = []

        
        for batch in np.arange(len(x) / batch_size):
            start = int(batch * batch_size)
            end = int(start + batch_size)
            samples = x[start:end]

            autoencoder_history = autoencoder.fit(x=samples,y=samples,epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
            if FLAGS.adversarial:
                fake_latent = encoder.predict(samples)
                real_sample=np.zeros([batch_size,FLAGS.latent_dim])
                real_sample=np.random.multivariate_normal(mean=mean, cov=conv, size=batch_size)
                discriminator_input = np.concatenate((fake_latent, real_sample))
                discriminator_labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))
                discriminator_history = discriminator.fit(x=discriminator_input, y=discriminator_labels, epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
                generator_history = generator.fit(x=samples, y=np.ones((batch_size, 1)), epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
            autoencoder_losses.append(autoencoder_history.history["loss"])
            if FLAGS.adversarial:
                discriminator_losses.append(discriminator_history.history["loss"])
                generator_losses.append(generator_history.history["loss"])
        now = datetime.now()
        print("\nEpoch {}/{} - {:.1f}s".format(epoch, n_epochs, (now - past).total_seconds()))
        print("Autoencoder Loss: {}".format(np.mean(autoencoder_losses)))

        if FLAGS.adversarial:
            print("Discriminator Loss: {}".format(np.mean(discriminator_losses)))
            print("Generator Loss: {}".format(np.mean(generator_losses)))
        past = now

        if epoch % 50 == 0:
            print("\nSaving models...")
            # autoencoder.save('{}_autoencoder.h5'.format(desc))
            encoder.save('{}_encoder.h5'.format(desc))
            decoder.save('{}_decoder.h5'.format(desc))
            # if FLAGS.adversarial:
            #     discriminator.save('{}_discriminator.h5'.format(desc))
            #     generator.save('{}_generator.h5'.format(desc))

    # autoencoder.save('{}_autoencoder.h5'.format(desc))
    encoder.save('{}_encoder.h5'.format(desc))
    decoder.save('{}_decoder.h5'.format(desc))
    z_latent= encoder.predict(x, batch_size=1)
    return z_latent
    
    # if FLAGS.adversarial:
        # discriminator.save('{}_discriminator.h5'.format(desc))
        # generator.save('{}_generator.h5'.format(desc))

def reconstruct(n_samples):
    encoder = load_model('{}_encoder.h5'.format(desc))
    decoder = load_model('{}_decoder.h5'.format(desc))
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    choice = np.random.choice(np.arange(n_samples))
    original = x_test[choice].reshape(1, 200)
    normalize = colors.Normalize(0., 255.)
    original = normalize(original)
    latent = encoder.predict(original)
    reconstruction = decoder.predict(latent)
    draw([{"title": "Original", "image": original}, {"title": "Reconstruction", "image": reconstruction}])

def generate(latent=None):
    decoder = load_model('{}_decoder.h5'.format(desc))
    if latent is None:
        latent = np.random.randn(1, FLAGS.latent_dim)
    else:
        latent = np.array(latent)
    sample = decoder.predict(latent.reshape(1, FLAGS.latent_dim))
    draw([{"title": "Sample", "image": sample}])

def draw(samples):
    fig = plt.figure(figsize=(5 * len(samples), 5))
    gs = gridspec.GridSpec(1, len(samples))
    for i, sample in enumerate(samples):
        ax = plt.Subplot(fig, gs[i])
        ax.imshow((sample["image"] * 255.).reshape(60, 60), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title(sample["title"])
        fig.add_subplot(ax)
    plt.show(block=False)
    raw_input("Press Enter to Exit")

def generate_grid(latent=None):
    decoder = load_model('{}_decoder.h5'.format(desc))
    samples = []
    for i in np.arange(400):
        latent = np.array([(i % 20) * 1.5 - 15., 15. - (i / 20) * 1.5])
        samples.append({
            "image": decoder.predict(latent.reshape(1, FLAGS.latent_dim))
        })
    draw_grid(samples)

def draw_grid(samples):
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(20, 20, wspace=-.5, hspace=0)
    for i, sample in enumerate(samples):
        ax = plt.Subplot(fig, gs[i])
        ax.imshow((sample["image"] * 255.).reshape(28, 28), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        # ax.set_title(sample["title"])
        fig.add_subplot(ax)
    plt.show(block=False)
    raw_input("Press Enter to Exit")
    # fig.savefig("images/{}_grid.png".format(desc), bbox_inches="tight", dpi=300)

def plot(n_samples):
    encoder = load_model('{}_encoder.h5'.format(desc))
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_test[:n_samples].reshape(n_samples, 200)
    y = y_test[:n_samples]
    normalize = colors.Normalize(0., 255.)
    x = normalize(x)
    latent = encoder.predict(x)
    if FLAGS.latent_dim > 2:
        tsne = TSNE()
        print("\nFitting t-SNE, this will take awhile...")
        latent = tsne.fit_transform(latent)
    fig, ax = plt.subplots()
    for label in np.arange(10):
        ax.scatter(latent[(y_test == label), 0], latent[(y_test == label), 1], label=label, s=3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_aspect('equal')
    ax.set_title("Latent Space")
    plt.show(block=False)
    raw_input("Press Enter to Exit")
    # fig.savefig("images/{}_latent.png".format(desc), bbox_inches="tight", dpi=300)

def main(argv):
    global desc,intermediate_dim1,intermediate_dim2,intermediate_dim3
    original_dim=200
    intermediate_dim1 = int(np.ceil(original_dim*1.2) + 5)
    intermediate_dim2 = int(max(np.ceil(original_dim/4), FLAGS.latent_dim+2) + 3)
    intermediate_dim3 = int(max(np.ceil(original_dim/10), FLAGS.latent_dim+1))
    if FLAGS.adversarial:
        desc = "aae"
    else:
        desc = "regular"
    if FLAGS.train:
        z_latent=train(n_samples=FLAGS.train_samples, batch_size=FLAGS.batchsize, n_epochs=FLAGS.epochs)
        savemat('Abundance '+'.mat', {'z_latent':z_latent})
        savemat('Endmember' + '.mat', {'E':W})
        return z_latent
    elif FLAGS.reconstruct:
        reconstruct(n_samples=FLAGS.test_samples)
    elif FLAGS.generate:
        if FLAGS.latent_vec:
            assert len(FLAGS.latent_vec) == FLAGS.latent_dim, "Latent vector provided is of dim {}; required dim is {}".format(len(FLAGS.latent_vec), FLAGS.latent_dim)
            generate(FLAGS.latent_vec)
        else:
            generate()
    elif FLAGS.generate_grid:
        generate_grid()
    elif FLAGS.plot:
        plot(FLAGS.test_samples)


if __name__ == "__main__":
    app.run(main)
    
