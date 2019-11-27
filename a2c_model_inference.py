import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.manifold import TSNE

from baselines.a2c.utils import conv, fc, conv_to_fc
from baselines.common.distributions import CategoricalPdType
from baselines.common import tf_util
from baselines.common.tf_util import adjust_shape

from tensorflow.math import reduce_std
from tensorflow.nn   import softmax

class a2c_model:

    def __init__(self, batch_size=None):

        self.sess = tf_util.get_session()

        # self.obs_space = (1,84,84,4)
        self.obs_dtype = np.uint8
        self.num_action = 4
        if batch_size is not None:
            self.batch_size = batch_size
            self.obs_space = (self.batch_size, 84, 84, 4)
        else:
            self.obs_space = (1, 84, 84, 4)

        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):

            self.X = tf.placeholder(shape=self.obs_space, dtype=self.obs_dtype, name='Observations')
            self.Encoded_X = tf.cast(self.X, tf.float32)

            with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):

                self.Scaled_images = tf.cast(self.Encoded_X, tf.float32) / 255.

                activ   = tf.nn.relu
                self.h1 = activ(conv(x=self.Scaled_images, scope='c1',  nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
                self.h2 = activ(conv(x=self.h1,            scope='c2',  nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
                self.h3 = activ(conv(x=self.h2,            scope='c3',  nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
                self.h3 = conv_to_fc(x=self.h3)

                self.policy_latent = activ(fc(x=self.h3, scope='fc1', nh=512, init_scale=np.sqrt(2)))

            self.vf_latent     = self.policy_latent
            self.vf_latent = tf.layers.flatten(self.vf_latent)
            self.policy_latent = tf.layers.flatten(self.policy_latent)


            # Based on the action space, will select what probability distribution type
            self.pdtype = CategoricalPdType(self.num_action)
            self.pd, self.pi = self.pdtype.pdfromlatent(self.policy_latent, init_scale=0.01)
            # Take an action
            self.action = self.pd.sample()

            # Calculate the neg log of our probability
            self.neglogp = self.pd.neglogp(self.action)

            self.softmax_out = softmax(self.pi, axis=1)

            # Std of outputs
            self.std = reduce_std(self.softmax_out, axis=1)

            self.vf = fc(self.vf_latent, 'vf', 1)
            self.vf = self.vf[:, 0]

        tf.global_variables_initializer().run(session=self.sess)

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def value(self, observation, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)
        - observation     observation data (either single or a batch)
        - **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)
        return:
        - value estimate
        """
        return self._evaluate(self.vf, observation, *args, **kwargs)


    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)
        - observation     observation data (either single or a batch)
        - **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)
        return:
        - action
        - value estimate
        - negative log likelihood of the action under current policy parameters
        - current latent activations vector
        """

        a, v, neglogp, latent_out, std = self._evaluate([self.action, self.vf,
                                                    self.neglogp, self.policy_latent,
                                                    self.std], observation, **extra_feed)
        return a, v, neglogp, latent_out, std


    def load(self, load_path):

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        loaded_params = joblib.load(os.path.expanduser(load_path))

        restores = []

        if isinstance(loaded_params, list):
            assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
            for d, v in zip(loaded_params, variables):
                restores.append(v.assign(d))
        else:
            for v in variables:
                restores.append(v.assign(loaded_params[v.name]))

        self.sess.run(restores)


def generate(model=None, observations=None):

    if model is None:
        print("[Error] - model didn't specified")
    if observations is None:
        print("[Error] - observations didn't specified")

    a, v, neglogp, latent_out = model.step(observations)

    return a, v, neglogp, latent_out


def latent_analysis(latent=None, observations=None):

    actions_meaning = {
        0: 'NOOP',
        1: 'FIRE',
        2: 'RIGHT',
        3: 'LEFT'
    }

    main_fig = plt.figure()

    ax1 = main_fig.add_subplot(111)

    latent_embedded = TSNE(n_components=2, verbose=True).fit_transform(latent)
    sc = ax1.scatter(latent_embedded[:, 0],
                     latent_embedded[:, 1], s=10, picker=1)

    def onpick(event):
        ind = event.ind

        Obsfig, axs = plt.subplots(2, 2)
        Obsfig.suptitle('observation number {}\{}'.format(ind[0], observations.shape[0]))
        im0 = axs[0,0].matshow(observations[ind[0], :, :, 0])
        im1 = axs[0,1].matshow(observations[ind[0], :, :, 1])
        im2 = axs[1,0].matshow(observations[ind[0], :, :, 2])
        im3 = axs[1,1].matshow(observations[ind[0], :, :, 3])
        print('onpick observation: ', observations[ind[0], :, :, 0].shape)
        plt.show()

    main_fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()


def main(args):

    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('--load_model', help='Path to trained model to load from', default=None, type=str)
    arg_parser.add_argument('--load_observations', help='Path to observations (*.npy file)', default=None, type=str)
    arg_parser.add_argument('--save_exp', default=False, action='store_true')
    arg_parser.add_argument('--latent_analysis', default=False, action='store_true')
    args, unknown_args = arg_parser.parse_known_args(args)

    observations = np.load(args.load_observations)

    num_observations = observations.shape[0]

    model = a2c_model(batch_size=num_observations)
    model.load(args.load_model)

    a, v, _, latent_out = generate(model, observations)



    if args.save_exp:
        import datetime
        timestr = '_'.join(str(datetime.datetime.now()).split('.')[:-1]).replace(' ', '_').replace(':', '-')
        exp_dir = args.load_model + '_' + 'exp_' + timestr
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        latent_basename = "latent"
        latent_save_file = os.path.join(exp_dir, latent_basename)
        action_basename = "action"
        action_save_file = os.path.join(exp_dir, action_basename)
        value_basename  = "value"
        value_save_file = os.path.join(exp_dir, value_basename)

        np.save(latent_save_file, latent_out)
        np.save(action_save_file, a)
        np.save(value_save_file, v)

    if args.latent_analysis:
        latent_analysis(latent_out, observations)

if __name__ == '__main__':
    main(sys.argv)








