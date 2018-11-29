import tensorflow as tf
import scipy.signal
import numpy as np
import gym
from modules.stopping_actor import DeepStoppingActor
from modules.environment import RSVPCPEnvironment

# Size of mini batches to run training on
MINI_BATCH = 30
REWARD_FACTOR = 0.001


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Weighted random selection returns n_picks random indexes.
# the chance to pick the index i is give by the weight weights[i].
def weighted_pick(weights, n_picks):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return np.searchsorted(t, np.random.rand(n_picks) * s)


# Discounting function used to calculate discounted returns.
def discounting(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Normalization of inputs and outputs
def norm(x, upper, lower=0.):
    return (x - lower) / max((upper - lower), 1e-12)


class Worker(object):
    def __init__(self, name, s_size, a_size, trainer, model_path,
                 global_episodes, user_num, test):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))
        self.is_test = test
        self.a_size = a_size

        # Create the local copy of the network and the tensorflow
        self.local_A = DeepStoppingActor(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.env = RSVPCPEnvironment(user_num)

    def get_env(self):
        return self.env

    def train(self, rollout, sess, gamma, r):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        rewards_list = np.asarray(rewards.tolist() + [r]) * REWARD_FACTOR
        discounted_rewards = discounting(rewards_list, gamma)[:-1]

        # Advantage estimation
        # JS, P Moritz, S Levine, M Jordan, P Abbeel,
        # "High-dimensional continuous control using generalized advantage estimation."
        # arXiv preprint arXiv:1506.02438 (2015).
        values_list = np.asarray(values.tolist() + [r]) * REWARD_FACTOR
        advantages = rewards + gamma * values_list[1:] - values_list[:-1]
        discounted_advantages = discounting(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # sess.run(self.local_AC.reset_state_op)
        rnn_state = self.local_A.state_init
        feed_dict = {self.local_A.target_v: discounted_rewards,
                     self.local_A.inputs: np.vstack(states),
                     self.local_A.actions: np.vstack(actions),
                     self.local_A.advantages: discounted_advantages,
                     self.local_A.state_in[0]: rnn_state[0],
                     self.local_A.state_in[1]: rnn_state[1]}
        p_l, e_l, g_n, v_n, _ = sess.run([self.local_A.policy_loss,
                                          self.local_A.entropy,
                                          self.local_A.grad_norms,
                                          self.local_A.var_norms,
                                          self.local_A.apply_grads],
                                         feed_dict=feed_dict)
        return p_l / len(rollout), e_l / len(
            rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_mini_buffer = []
                episode_values = []
                episode_states = []
                episode_reward = 0
                episode_step_count = 0

                # Restart environment
                terminal = False
                s = self.env.reset()

                rnn_state = self.local_A.state_init

                # Run an episode
                while not terminal:
                    episode_states.append(s)
                    if self.is_test:
                        self.env.render()

                    # Get preferred action distribution
                    try:
                        a_dist, v, rnn_state = sess.run(
                            [self.local_A.policy,
                             self.local_A.state_out,
                             self.local_A.state_in],
                            feed_dict={self.local_A.inputs: [s],
                                       self.local_A.state_in[0]: rnn_state[0],
                                       self.local_A.state_in[1]: rnn_state[1]})
                    except:
                        asd = s
                        import pdb
                        pdb.set_trace()
                    v = v[0]

                    a0 = weighted_pick(a_dist[0],
                                       1)  # Use stochastic distribution sampling
                    if self.is_test:
                        a0 = np.argmax(a_dist[0])  # Use maximum when testing
                    a = np.zeros(self.a_size)
                    a[a0] = 1

                    s2, r, terminal = self.env.step(np.argmax(a))

                    episode_reward += r

                    episode_buffer.append([s, a, r, s2, terminal, v[0, 0]])
                    episode_mini_buffer.append([s, a, r, s2, terminal, v[0, 0]])

                    episode_values.append(v[0, 0])

                    # Train on mini batches from episode
                    if len(
                            episode_mini_buffer) == MINI_BATCH and not self.is_test:
                        v1 = sess.run([self.local_A.value],
                                      feed_dict={self.local_A.inputs: [s],
                                                 self.local_A.state_in[0]:
                                                     rnn_state[0],
                                                 self.local_A.state_in[1]:
                                                     rnn_state[1]})
                        v_l, p_l, e_l, g_n, v_n = self.train(
                            episode_mini_buffer, sess, gamma, v1[0][0])
                        episode_mini_buffer = []

                    # Set previous state for next step
                    s = s2
                    total_steps += 1
                    episode_step_count += 1

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if episode_count % 10 == 0 and not episode_count % 100 == 0 and not self.is_test:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward',
                                      simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length',
                                      simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value',
                                      simple_value=float(mean_value))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    if episode_count % 100 == 0 and not self.is_test:
                        saver.save(sess, self.model_path + '/model-' + str(
                            episode_count) + '.cptk')
                    if episode_count % 10 == 0 and not self.is_test:
                        mean_reward = np.mean(self.episode_rewards[-5:])
                        print("| Reward: " + str(mean_reward), " | Episode",
                              episode_count)

                    sess.run(self.increment)  # Next global episode

                episode_count += 1
