""" Modified from: https://github.com/MorvanZhou/ """

import tensorflow as tf
import numpy as np

cell_size = 30
size_dense_val = 20
size_dense_act = 10
entropy_beta = 0.01


class DeepStoppingActorCritic(object):
    def __init__(self, scope, dim_action, dim_state, session, opt_a, opt_c,
                 global_actor=None, global_net_scope='Global_Net'):

        self.session = session
        self.opt_a = opt_a
        self.opt_c = opt_c

        if scope == global_net_scope:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, dim_state], 'state')
                self.a_prob, self.v, self.a_params, self.c_params = \
                    self._build_net(scope, dim_action, dim_state)
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, dim_state], 'state')
                self.a_his = tf.placeholder(tf.int32, [None, 1], 'action')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'tar_val')

                self.a_prob, self.v, self.a_params, self.c_params = \
                    self._build_net(scope, dim_action, dim_state)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(
                        tf.log(self.a_prob + 1e-5) *
                        tf.one_hot(self.a_his, dim_action), axis=1,
                        keepdims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(
                        self.a_prob * tf.log(self.a_prob + 1e-5),
                        axis=1, keepdims=True)
                    self.exp_v = entropy_beta * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in
                                             zip(self.a_params,
                                                 global_actor.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in
                                             zip(self.c_params,
                                                 global_actor.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.opt_a.apply_gradients(
                        zip(self.a_grads, global_actor.a_params))
                    self.update_c_op = self.opt_c.apply_gradients(
                        zip(self.c_grads, global_actor.c_params))

    def _build_net(self, scope, dim_action, dim_state):
        w_init = tf.random_normal_initializer(0., .1)
        # only critic controls the rnn update
        with tf.variable_scope('critic'):
            s = tf.expand_dims(self.s, axis=1, name='timely_input')
            # [time_step, feature] => [time_step, batch, feature]
            rnn_cell = tf.contrib.rnn.BasicRNNCell(cell_size)
            self.init_state = rnn_cell.zero_state(batch_size=1,
                                                  dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell, inputs=s, initial_state=self.init_state,
                time_major=True)
            # flatten state representation
            cell_out = tf.reshape(outputs, [-1, cell_size],
                                  name='flatten_rnn_outputs')
            l_c = tf.layers.dense(cell_out, size_dense_val, tf.nn.relu6,
                                  kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init,
                                name='v')  # state value

        # state representation is based on critic
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(cell_out, size_dense_act, tf.nn.relu6,
                                  kernel_initializer=w_init, name='la')
            action = tf.layers.dense(l_a, dim_action, tf.nn.softmax,
                                     kernel_initializer=w_init, name='action')

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope + '/critic')
        return action, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        # local grads applies to global net
        self.session.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):  # run by a local
        self.session.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, cell_state, flag_train):  # run by a local
        s = s[np.newaxis, :]
        a_w, cell_state = self.session.run([self.a_prob, self.final_state],
                                           {self.s: s,
                                            self.init_state: cell_state})

        if flag_train:
            a = np.random.choice(range(a_w.shape[1]), p=a_w.ravel())
        else:
            a = np.argmax(a_w)

        return a, cell_state
