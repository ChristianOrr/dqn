# Deep Q-Learning with Acme
This repository contains an implementation of DQN using Acme. This implementation is based off the Acme implementation, but contains the following additional features:
- Pretrained weights for the model.
- Convenient method for loading pretrained wieghts.
- Logging training data to Tensorboard.
- A demo of the model playing the atari game after it has been trained.

DQN is capabale of playing many Atari games, although not with equal competency, see the Vanilla DQN paper in the references for performance evaluation. The weights are unfortunately not transferable to different games, since the weights save the learner object, which is dependant on the environment.  


## Installation Requirements
The latest version of Acme is required. Don't install the default Acme from pip, because it's too outdated, it needs to be installed using: 
```
pip install git+https://github.com/deepmind/acme.git#egg=dm-acme[jax,tf,envs]
```
If you want jax for GPU, run the following command first:
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
``` 
Note that the version of tensorflow is important for Reverb, so its better to install tensorflow with the acme package as shown above. Open AI Gym must not be greater then version 25, since there are breaking changes introduced in version 26.


## References 
- [Acme Deep Q-Networks (DQN)](https://github.com/deepmind/acme/tree/master/acme/agents/jax/dqn) 
- [Acme Run DQN Example](https://github.com/deepmind/acme/blob/master/examples/baselines/rl_discrete/run_dqn.py)
- [DQN Pretrained Models](https://huggingface.co/ChristianOrr/dqn/tree/main)
- [Vanilla Deep Q-learning by Mnih et al., 2013](https://arxiv.org/abs/1312.5602)
