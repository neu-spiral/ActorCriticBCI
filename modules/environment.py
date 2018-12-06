""" Environment to replicate an brain computer interface task with batch
    queries. """

import os
from modules.oracle import BinaryRSVPOracle
import scipy.io as sio
from modules.main_frame import DecisionMaker, EvidenceFusion
from scipy.stats import iqr
from sklearn.neighbors.kde import KernelDensity
from modules.main_frame import alphabet
from modules.query_methods import randomQuery, NBestQuery, MomentumQueryingLog
import numpy as np

max_num_seq = 20  # maximum number of allowed sequences
sc_threshold = 100  # threshold to remove outliers data
delta = 2  # manually separate reward distributions conditioned on (a,s) tuple
len_query = 5  # number of trials in a sequence (batch querying)
evidence_names = ['LM', 'Eps']  # types of evidences during simulation


class RSVPCPEnvironment(object):
    """ RSVP Copy Phrase environment;
        state: [probability distribution of the alphabet, normalized sequence #]
        action: [continue querying, stop querying and decide]
        reward: negative for each query
                very negative for incorrect decision
                positive for correct decision
            Make sure that rewards match the constraints of the actor critic
            approach. If expected reward of querying is higher than making a
            decision, system will continue querying endlessly.
        Fnc:
            reset(): resets the environment and re-initializes components
            step(a): given an action, computes the next state and reward
        Attr:
            alp(list[str]): symbol set in the typing system
            list_filename(list[str]): path of the files
            pre_phrase(str): current state of the system
            phrase(str): success state of the system
            oracle(rsvp_oracle): rsvp keyboard oracle (a synth. user)
            decision_maker(DecisionMaker): rsvp inference environment
            conjugator(Conjugator): rsvp evidence fuse environment
            step_counter(int): global step counter for the task (num sequences)
            dist(list[KDE]): kernel density estimates to calculate likelihoods
                each KDE is for each possible class respectively.
            sti(list[str]): list of stimuli presented to the user
            """

    def __init__(self, user_num):

        self.alp = alphabet()
        # path to data (data should be a .mat file)
        # TODO: ask for a path input
        path = "./data/"
        self.list_filename = os.listdir(path)[1:]

        # List of query methods. If required, can be populated
        query_method = MomentumQueryingLog(alp=self.alp, len_query=len_query,
                                           gam=1, lam=0.3, updateLam=False)
        # load filename and initialize the user from MATLAB file
        try:
            filename = self.list_filename[user_num]
            tmp = sio.loadmat(path + filename)
            x = tmp['scores']
            y = tmp['trialTargetness']
        except:
            import pdb
            pdb.set_trace()

        # modify data for outliers
        y = y[x > -sc_threshold]
        x = x[x > -sc_threshold]
        y = y[x < sc_threshold]
        x = x[x < sc_threshold]
        x[y == 1] += delta

        # create the oracle for the user

        self.pre_phrase = self.alp[np.random.randint(len(self.alp))]
        self.phrase = self.pre_phrase + self.alp[
            np.random.randint(len(self.alp))]
        self.oracle = BinaryRSVPOracle(x, y, phrase=self.phrase, alp=self.alp)
        self.oracle.update_state(self.pre_phrase)

        # create the decision maker.
        self.decision_maker = DecisionMaker(state=self.pre_phrase,
                                            len_query=len_query, alp=self.alp,
                                            query_method=query_method,
                                            artificial_stop=True)
        self.conjugator = EvidenceFusion(evidence_names, len_dist=len(self.alp))
        self.step_counter = 0

        self.dist = []  # distributions for positive and negative classes
        # create an EEG model that fits   the user explicitly
        bandwidth = 1.06 * min(np.std(x),
                               iqr(x) / 1.34) * np.power(x.shape[0], -0.2)
        classes = np.unique(y)
        cls_dep_x = [x[np.where(y == classes[i])] for i in range(len(classes))]
        for i in range(len(classes)):
            self.dist.append(KernelDensity(bandwidth=bandwidth))

            dat = np.expand_dims(cls_dep_x[i], axis=1)
            self.dist[i].fit(dat)

        self.sti = []

    def reset(self):
        """ Resets all environment variables to their initial position. Randomly
            initializes current state and win state. Resets oracle and decision
            maker and update their state respectively.
            Return:
                s(ndarray[float]): state of the environment
                    [p(alp), norm(# seq)] """

        # number of trials in a sequence (bathced querying)
        alp = alphabet()
        # create the oracle for the user
        self.pre_phrase = self.alp[np.random.randint(len(self.alp))]
        idx_phrase = np.random.randint(len(self.alp))
        self.phrase = self.pre_phrase + self.alp[idx_phrase]
        self.oracle.phrase = self.phrase
        self.oracle.update_state(self.pre_phrase)

        # create the decision maker.
        self.decision_maker.reset(state=self.pre_phrase)
        self.conjugator.reset_history()

        # TODO: make it optional to adjust the difficulty of the language model
        lm_prior = np.abs(np.random.randn(len(alp)))
        # lm_prior[self.alp.index(self.oracle.state)] = np.min(lm_prior)
        # lm_prior[self.alp.index(self.oracle.state)] /= 100
        lm_prior /= np.sum(lm_prior)
        prob = self.conjugator.update_and_fuse({evidence_names[0]: lm_prior})
        prob_new = np.array([i for i in prob])
        d, self.sti = self.decision_maker.decide(prob_new)
        s = np.array(
            self.decision_maker.list_epoch[-1]['list_distribution'])[-1]

        self.step_counter = 0
        s = np.append(s, self.step_counter / max_num_seq)

        return s

    def step(self, a):
        """ Takes the step given an action
            Args:
                a(bin): a binary decision to stop or continue
            Return:
                s(ndarray[float]): state of the environment
                r(float): reward for the (a,s) pair
                d(bin): 1 if episode is finished 0 if not
                info(list[..]): a list of information varialbes that can
                    are going to be used during training or to report perf. """
        d = a

        if (not d) and self.step_counter < 30:
            d = 0
            score = self.oracle.answer(self.sti)
            # get the likelihoods for the scores
            likelihood = []
            for i in score:
                dat = np.squeeze(i)
                dens_0 = self.dist[0].score_samples(
                    dat.reshape(1, -1))[0]
                dens_1 = self.dist[1].score_samples(
                    dat.reshape(1, -1))[0]
                likelihood.append(np.asarray([dens_0, dens_1]))
            likelihood = np.array(likelihood)
            # compute likelihood ratio for the query
            lr = np.exp(likelihood[:, 1] - likelihood[:, 0])

            # initialize evidence with all ones
            evidence = np.ones(len(self.alp))

            c = 0
            # update evidence of the queries that are asked
            for q in self.sti:
                idx = self.alp.index(q)
                evidence[idx] = lr[c]
                c += 1

            # update posterior and decide what to do
            prob = self.conjugator.update_and_fuse(
                {evidence_names[1]: evidence})
            prob_new = np.array([i for i in prob])
            _, self.sti = self.decision_maker.decide(prob_new)
            s = np.array(
                self.decision_maker.list_epoch[-1]['list_distribution'])[-1]
            s = np.append(s, self.step_counter / max_num_seq)
            if self.step_counter < 5:
                r = - 1
            else:
                r = - 5

            is_correct = None
            self.step_counter += 1

        else:
            s = np.array(self.decision_maker.list_epoch[-1][
                             'list_distribution'])[-1]
            s = np.append(s, self.step_counter / max_num_seq)

            if np.argmax(s[:-1]) == self.alp.index(self.oracle.state):

                r = 250
                is_correct = True
            else:
                r = -5
                is_correct = False
            d = 1

        return s, r, d == 1, [self.step_counter, is_correct]
