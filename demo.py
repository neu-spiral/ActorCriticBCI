"""
Asynchronous Advantage Actor Critic (A3C) Method with Recurrent Neural Network.
Network gates state input of probability distribution appended with a normalized
sequence number. Decides on stopping using RSVPKeyboard environment

Modified from: https://github.com/MorvanZhou/
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from modules.progress_bar import progress_bar
import matplotlib.pyplot as plt
from modules.stopping_actor import DeepStoppingActorCritic
from modules.environment import RSVPCPEnvironment
import datetime

num_workers = multiprocessing.cpu_count()
max_step_episode = 20
max_global_epoch = 60000
global_network_scope = 'Global_Net'
backprop_num = 50

flag_train = True  # if set to true trains the mode and saves the model to path
max_num_mc = 1000  # # of episodes in testing if flag is false
model_path = "model/model_discount_3/model.ckpt"

gamma = 0.90  # decay for reward in time
learning_rate_actor = 0.001  # learning rate for actor
learning_rate_critic = 0.005  # learning rate for critic
global_running_reward = []
list_decision, list_steps = [], []
global_epoch = 0
hop = 500

size_state = 28
size_action = 2

# difficulty is increased, agent is tasked to perform faster!
step_difficulty = max_global_epoch / 6  # steps to increase difficulty


class Worker(object):
    def __init__(self, name, global_a_c):
        """Args:
            name(str): environment name
            global_a_c(environment): environment to be used with the worker.
                environment should have following functions;
                    step - reset"""
        self.env = RSVPCPEnvironment(int(name[2:]))
        self.name = name
        self.actor_critic = DeepStoppingActorCritic(name, size_action,
                                                    size_state, sess,
                                                    optimizer_actor,
                                                    optimizer_critic,
                                                    global_actor=global_a_c)

    def work(self):
        """ Interact with the environment. Perform an episode, get (s,a,r)
            tuples and apply gradient descent on the actor and critic until
            maximum number of epochs is reached. """

        # global parameters to be shared among different workers.
        global global_running_reward, global_epoch, list_decision, list_steps

        # global step_counter
        total_step = 1

        # Have a buffer for (s,a,r) tuples. Network gradient descent is usually
        #   faster than running environment. It is beneficial to store an
        #   instance to use after in the training.
        buffer_s, buffer_a, buffer_r = [], [], []
        while not coord.should_stop() and global_epoch < max_global_epoch:
            s = self.env.reset()
            ep_r = 0
            # zero rnn state at beginning
            rnn_state = sess.run(self.actor_critic.init_state)
            keep_state = rnn_state.copy()  # keep rnn state for global
            done = 0
            # local machine episode count. Start with -1 to increment directly
            #   within the for loop
            local_count = -1
            while not done:
                # local machine episode count
                local_count += 1
                # get the action and next rnn state
                a, rnn_state_ = self.actor_critic.choose_action(s, rnn_state,
                                                                flag_train)

                dif_mul = int(global_epoch / step_difficulty)
                s_, r, done, info = self.env.step(a, dif_mul=dif_mul)

                # If number of trials in an episodes exceeds threshold, stop.
                if local_count == max_step_episode:
                    done = True

                # If episode is finished, update actual statistics;
                #   correct / incorrect selection and trials spent.
                if done:
                    if info[1] == True:
                        list_decision.append(1)
                        list_steps.append(info[0])
                    elif info[1] == False:
                        list_decision.append(0)
                        list_steps.append(info[0])

                # compute episode reward
                ep_r += r

                # update buffer (s,a,r)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                # update global and assign to local net
                if total_step % backprop_num == 0 or done:
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        # compute predicted value using critic
                        v_s_ = sess.run(self.actor_critic.v,
                                        {self.actor_critic.s: s_[np.newaxis, :],
                                         self.actor_critic.init_state:
                                             rnn_state_})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    # update buffer
                    buffer_s, buffer_a, buffer_v_target = np.vstack(
                        buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)

                    # create feed dictionary to be updated
                    feed_dict = {
                        self.actor_critic.s: buffer_s,
                        self.actor_critic.a_his: buffer_a,
                        self.actor_critic.v_target: buffer_v_target,
                        self.actor_critic.init_state: keep_state,
                    }

                    # update the global network using all intances generated
                    self.actor_critic.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.actor_critic.pull_global()
                    keep_state = rnn_state_.copy()

                s = s_  # renew current state
                rnn_state = rnn_state_  # renew rnn state
                total_step += 1

                # if an episode is finished, update the running reward list and
                #   report the current reward of the episode.
                if done:
                    if len(
                            global_running_reward) == 0:
                        global_running_reward.append(ep_r)
                    else:
                        global_running_reward.append(
                            0.9 * global_running_reward[-1] + 0.1 * ep_r)
                    print("{}, EP:{} | EP_r:{}".format(self.name, global_epoch,
                                                       global_running_reward[
                                                           -1]))
                    global_epoch += 1
                    break


if __name__ == "__main__":

    sess = tf.Session()
    now = datetime.datetime.now()

    with tf.device("/cpu:0"):
        optimizer_actor = tf.train.RMSPropOptimizer(learning_rate_actor,
                                                    name='RMSPropA')
        optimizer_critic = tf.train.RMSPropOptimizer(learning_rate_critic,
                                                     name='RMSPropC')
        global_actor_critic = \
            DeepStoppingActorCritic(global_network_scope, size_action,
                                    size_state, sess, optimizer_actor,
                                    optimizer_critic)

        workers = []
        # Create workers
        for i in range(num_workers):
            i_name = 'W_%i' % i  # worker name
            workers.append(Worker(i_name, global_actor_critic))
    saver = tf.train.Saver(var_list=tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=global_network_scope))

    if flag_train:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            job = lambda: worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        coord.join(worker_threads)

        save_path = saver.save(sess, model_path)
        print("Model saved in path: %s" % save_path)

        # Plot final training results
        fig = plt.figure()
        ax = fig.add_subplot(211)
        tmp = [np.sum(global_running_reward[a:a + hop]) / hop for a in
               range(1, len(list_decision) - hop)]
        ax.plot(np.arange(len(tmp)), tmp)
        ax.set_ylabel('Total moving reward')

        ax1 = fig.add_subplot(212)
        ax2 = ax1.twinx()
        tmp = [np.sum(list_decision[a:a + hop]) / hop for a in
               range(1, len(list_decision) - hop)]
        ax1.plot(np.arange(len(tmp)), tmp, label='acc')
        ax1.set_xlabel('step')
        ax1.set_ylabel('Accuracy')
        tmp2 = [np.sum(list_steps[a:a + hop]) / hop for a in
                range(1, len(list_steps) - hop)]
        ax2.plot(np.arange(len(tmp)), tmp2, color='k', label='seq')
        ax2.set_ylabel('Sequences')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc=4)
        plt.show()

        print("Accuracy:{}, Sequence:{}".format(np.max(tmp),
                                                tmp2[np.argmax(tmp)]))

    # Test the pretrained model on the path
    else:
        saver.restore(sess, model_path)

        env = RSVPCPEnvironment(int(12))
        actor_critic_test = global_actor_critic

        progress_bar(0, max_num_mc, prefix='Progress:', suffix='Complete',
                     length=50)
        for idx in range(max_num_mc):
            progress_bar(idx + 1, max_num_mc, prefix='Progress:',
                         suffix='Complete', length=50)
            s = env.reset()
            # zero rnn state at beginning
            rnn_state = sess.run(actor_critic_test.init_state)
            keep_state = rnn_state.copy()  # keep rnn state for global
            done = 0
            # local machine episode count. Start with -1 to increment directly
            #   within the for loop
            local_count = -1
            while not done:
                # local machine episode count
                local_count += 1
                # get the action and next rnn state
                a, rnn_state_ = actor_critic_test.choose_action(s, rnn_state,
                                                                flag_train)
                s_, r, done, info = env.step(a)

                s = s_  # renew current state
                rnn_state = rnn_state_  # renew rnn state

                if local_count == max_step_episode:
                    done = True

                if done:
                    if info[1] == True:
                        list_decision.append(1)
                        list_steps.append(info[0])
                    elif info[1] == False:
                        list_decision.append(0)
                        list_steps.append(info[0])
                    break

        tmp = [np.sum(list_decision[a:a + hop]) / hop for a in
               range(1, len(list_decision) - hop)]
        tmp2 = [np.sum(list_steps[a:a + hop]) / hop for a in
                range(1, len(list_steps) - hop)]
        print("Accuracy:{}, Sequence:{}".format(np.max(tmp),
                                                tmp2[np.argmax(tmp)]))
