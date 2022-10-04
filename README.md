# Deep Q-Learning with Acme
This repository contains an implementation of DQN using Acme. This implementation is based off the Acme implementation, but contains the following additional features:
- Pretrained weights for the model.
- Convenient method for loading pretrained wieghts.
- Logging training data to Tensorboard.
- A demo of the model playing the atari game after it has been trained.




## Installation Requirements
The latest version of Acme is required. Don't install the default Acme from pip, because it's too outdated, it needs to be installed using: 
```
pip install git+https://github.com/deepmind/acme.git#egg=dm-acme[jax,tf,envs]
```
If you want jax for GPU, run the following command first:
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
``` 



## References 
- [Acme Deep Q-Networks (DQN)](https://github.com/deepmind/acme/tree/master/acme/agents/jax/dqn) 
- [Acme Run DQN Example](https://github.com/deepmind/acme/blob/master/examples/baselines/rl_discrete/run_dqn.py)
- [DQN Pretrained Models](https://huggingface.co/ChristianOrr/dqn/tree/main)
- [Vanilla Deep Q-learning by Mnih et al., 2013](https://arxiv.org/abs/1312.5602)
