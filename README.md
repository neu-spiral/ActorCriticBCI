# AC-BCI (Actor Critic for Brain Computer Interfaces)
------------------------
To see results in a nutshell run `demo.py`. Trains an actor using multiple CPUs with provided data.

Asynchronous Advantage Actor Critic (A3C) with discrete action space for our BCI system RSVPKeyboard. The environment includes a synth. oracle (user, `oracle.py`) a main-frame (BCI environment, `main_frame.py`) and within the main frame an actor (`stopping_actor.py`) which decides to stop. All these modules are brought together within the environment (`environment.py`) to simulate episodes for training.

__________________________
Within the implementation we use a recurrent neural network to learn the transition between query sequences within the episode. Actor and critic networks are trained separately and both connected to RNN. We denote state, action and reward tuple with (s,a,r);
1. *s*: Probability distribution over alphabet appended with number of sequences.
2. *a*: Commitment to a symbol. 
3. *r*: Positive for each correct decision. Negative for incorrect decision. 
        We also penalize each stimuli presentation to avoid infinite loops.
        
If you want to use data please cite:
`@article{kocanaogullari2018optimal,
  title={Optimal Query Selection Using Multi-Armed Bandits},
  author={Kocanaogullari, Aziz and Marghi, Yeganeh M and Akcakaya, Murat and Erdogmus, Deniz},
  journal={IEEE Signal Processing Letters},
  year={2018},
  publisher={IEEE}
}`
