from typing import Sequence, Tuple, Optional
import functools
import sys
import time
import argparse
import dm_env
import tensorflow as tf
from acme.jax.experiments import config
from acme.utils import counting
from acme.tf import savers
import os
import reverb
import jax
import acme
from acme import core
from acme import specs
from acme import wrappers
from acme import types
from acme.agents.jax import dqn
from acme.jax import experiments
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import loggers
from acme.agents.jax.dqn import losses
import gym
from acme.utils.loggers import base, TerminalLogger, Dispatcher
from acme.utils.loggers.tf_summary import TFSummaryLogger
from absl import logging
from keras.utils import data_utils
import flax.linen as nn
import jax.numpy as jnp

pretrained_weights = {"pong"}

parser = argparse.ArgumentParser(description='Training for DQN Flax')
parser.add_argument("--env_type", default="Pong", help='Type of environment, options include: Pong, Breakout...', required=False)
parser.add_argument("--actor_steps", default=500_000, help='Number of steps to train', required=False)
parser.add_argument("--logs_dir", default="./logs", help='Path to save Tensorboard logs', required=False)
parser.add_argument("--saves_dir", default="./saves", help='Path to save and load checkpoints', required=False)
parser.add_argument("--weights_path",
                    help='One of the following pretrained weights (will download automatically): '
                         '"pong"'
                         'or a path to pretrained checkpoint file (for fine turning)',
                    default=None, required=False)
parser.add_argument("--lr", help="Initial value for learning rate.", default=0.0001, type=float, required=False)
parser.add_argument("--batch_size", help='batch size to use during training', type=int, default=32)
parser.add_argument("--eval_steps", default=1_000, help='Perform evaluation at every eval_steps frequency', required=False)
parser.add_argument("--save_freq", help='Time in minutes to wait before saving a new checkpoint', type=int, default=10)
args = parser.parse_args()




# Type of environment, options include:
# Pong, Breakout
ENV_TYPE = args.env_type
ACTOR_STEPS = args.actor_steps
LOGS_DIR = f"{args.logs_dir}/flax/{ENV_TYPE}/"
CHECKPOINT_DIR = os.path.abspath(f"{args.saves_dir}/flax/{ENV_TYPE}/")
WEIGHTS = args.weights_path
LR = args.lr
BATCH_SIZE = args.batch_size
EVAL_STEPS = args.eval_steps
SAVE_FREQ = args.save_freq


# Check if the weights are valid
if not (WEIGHTS is None or
        WEIGHTS in pretrained_weights or
        tf.io.gfile.exists(WEIGHTS) or
        tf.io.gfile.exists(WEIGHTS + ".index")):
    raise ValueError('The `WEIGHTS` argument should be either '
                        '`None` (random initialization), '
                        f'one of the following pretrained weights: {pretrained_weights}, '
                        'or the path to the weights file to be loaded. \n'
                        f'Received `weights={WEIGHTS}`')




def make_atari_environment(
    level: str = 'Pong',
    sticky_actions: bool = True,
    zero_discount_on_life_loss: bool = False,
    oar_wrapper: bool = False,
    num_stacked_frames: int = 4,
    flatten_frame_stack: bool = False,
    grayscaling: bool = True,
    render_mode: str = None
) -> dm_env.Environment:
    """Loads the Atari environment."""
    # Internal logic.
    version = 'v0' if sticky_actions else 'v4'
    level_name = f'{level}NoFrameskip-{version}'
    env = gym.make(
        level_name, 
        full_action_space=True, 
        render_mode=render_mode
        )

    wrapper_list = [
        wrappers.GymAtariAdapter,
        functools.partial(
            wrappers.AtariWrapper,
            to_float=True,
            max_episode_len=108_000,
            num_stacked_frames=num_stacked_frames,
            flatten_frame_stack=flatten_frame_stack,
            grayscaling=grayscaling,
            zero_discount_on_life_loss=zero_discount_on_life_loss,
        ),
        wrappers.SinglePrecisionWrapper,
  ]

    if oar_wrapper:
        # E.g. IMPALA and R2D2 use this particular variant.
        wrapper_list.append(wrappers.ObservationActionRewardWrapper)

    return wrappers.wrap_all(env, wrapper_list)


def make_environment(seed: int) -> dm_env.Environment:
    del seed
    return make_atari_environment(
        level=ENV_TYPE,
        sticky_actions=False,
        zero_discount_on_life_loss=True
    )



Images = jnp.ndarray

class PolicyNetwork(nn.Module):


    @nn.compact
    def __call__(self, inputs: Images) -> jnp.ndarray:
        num_actions = environment_spec.actions.num_values

        inputs_rank = jnp.ndim(inputs)
        batched_inputs = inputs_rank == 4
        if inputs_rank < 3 or inputs_rank > 4:
            raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)


        outputs = nn.Sequential(
            [nn.Conv(features=16, kernel_size=(8, 8), strides=4, padding="SAME", use_bias=True), nn.relu,
            nn.Conv(features=32, kernel_size=(4, 4), strides=2, padding="SAME", use_bias=True), nn.relu]
        )(inputs)
        # Flatten
        outputs = outputs.reshape(outputs.shape[0], -1)

        outputs = nn.Sequential(
            [nn.Dense(features=256),
            nn.Dense(num_actions)]
        )(outputs)


        if batched_inputs:
            return jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
        return jnp.reshape(outputs, [-1])  # [D]






def make_dqn_atari_network(
    environment_spec: specs.EnvironmentSpec) -> dqn.DQNNetworks:
    """Creates networks for training DQN on Atari."""
    network_flax = PolicyNetwork()


    obs = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
    network = networks_lib.FeedForwardNetwork(
        init=lambda rng: network_flax.init(rng, obs), apply=network_flax.apply)
    typed_network = networks_lib.non_stochastic_network_to_typed(network)
    return dqn.DQNNetworks(policy_network=typed_network)














dqn_config = dqn.DQNConfig(
            discount=0.99,
            eval_epsilon=0.0,
            learning_rate=LR,
            n_step=1,
            epsilon=0.1,
            target_update_period=1000,
            min_replay_size=20_000,
            max_replay_size=100_000,
            samples_per_insert=8,
            batch_size=BATCH_SIZE
            )
loss_fn = losses.QLearning(
        discount=dqn_config.discount, 
        max_abs_reward=1.
        )
dqn_builder = dqn.DQNBuilder(
        dqn_config, 
        loss_fn=loss_fn
        )



def make_logger(
        name: str,
        steps_key: Optional[str] = None,
        task_id: Optional[int] = None,
) -> loggers.Logger:
    terminal_logger = TerminalLogger(label=name, print_fn=logging.info)
    tb_logger = TFSummaryLogger(LOGS_DIR, label=name, steps_key=steps_key)
    serialize_fn = base.to_numpy
    logger = Dispatcher([terminal_logger, tb_logger], serialize_fn)
    return logger




checkpoint_config = experiments.CheckpointingConfig(
    max_to_keep=3,
    directory=CHECKPOINT_DIR,
    add_uid=False,
    time_delta_minutes=SAVE_FREQ
)


experiment_config = experiments.ExperimentConfig(
    builder=dqn_builder,
    environment_factory=make_environment,
    network_factory=make_dqn_atari_network,
    logger_factory=make_logger,
    seed=0,
    max_num_actor_steps=ACTOR_STEPS,
    checkpointing=checkpoint_config
    )  


class _LearningActor(core.Actor):
    """Actor which learns (updates its parameters) when `update` is called.
    This combines a base actor and a learner. Whenever `update` is called
    on the wrapping actor the learner will take a step (e.g. one step of gradient
    descent) as long as there is data available for training
    (provided iterator and replay_tables are used to check for that).
    Selecting actions and making observations are handled by the base actor.
    Intended to be used by the `run_experiment` only.
    """
    def __init__(self, actor: core.Actor, learner: core.Learner,
                iterator: core.PrefetchingIterator,
                replay_tables: Sequence[reverb.Table],
                sample_sizes: Sequence[int],
                checkpointer: Optional[savers.Checkpointer]):
        """Initializes _LearningActor.
        Args:
        actor: Actor to be wrapped.
        learner: Learner on which step() is to be called when there is data.
        iterator: Iterator used by the Learner to fetch training data.
        replay_tables: Collection of tables from which Learner fetches data
            through the iterator.
        sample_sizes: For each table from `replay_tables`, how many elements the
            table should have available for sampling to wait for the `iterator` to
            prefetch a batch of data. Otherwise more experience needs to be
            collected by the actor.
        checkpointer: Checkpointer to save the state on update.
        """
        self._actor = actor
        self._learner = learner
        self._iterator = iterator
        self._replay_tables = replay_tables
        self._sample_sizes = sample_sizes
        self._learner_steps = 0
        self._checkpointer = checkpointer

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        return self._actor.select_action(observation)

    def observe_first(self, timestep: dm_env.TimeStep):
        self._actor.observe_first(timestep)

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        self._actor.observe(action, next_timestep)

    def _maybe_train(self):
        trained = False
        while True:
            if self._iterator.ready():
                self._learner.step()
                batches = self._iterator.retrieved_elements() - self._learner_steps
                self._learner_steps += 1
                assert batches == 1, (
                    'Learner step must retrieve exactly one element from the iterator'
                    f' (retrieved {batches}). Otherwise agent can deadlock. Example '
                    'cause is that your chosen agent'
                    's Builder has a `make_learner` '
                    'factory that prefetches the data but it shouldn'
                    't.')
                trained = True
            else:
                # Wait for the iterator to fetch more data from the table(s) only
                # if there plenty of data to sample from each table.
                for table, sample_size in zip(self._replay_tables, self._sample_sizes):
                    if not table.can_sample(sample_size):
                        return trained
                    # Let iterator's prefetching thread get data from the table(s).
                time.sleep(0.001)

    def update(self):
        if self._maybe_train():
            # Update the actor weights only when learner was updated.
            self._actor.update()
        if self._checkpointer:
            self._checkpointer.save()


def _disable_insert_blocking(
    tables: Sequence[reverb.Table]
) -> Tuple[Sequence[reverb.Table], Sequence[int]]:
    """Disables blocking of insert operations for a given collection of tables."""
    modified_tables = []
    sample_sizes = []
    for table in tables:
        rate_limiter_info = table.info.rate_limiter_info
        rate_limiter = reverb.rate_limiters.RateLimiter(
            samples_per_insert=rate_limiter_info.samples_per_insert,
            min_size_to_sample=rate_limiter_info.min_size_to_sample,
            min_diff=rate_limiter_info.min_diff,
            max_diff=sys.float_info.max)
        modified_tables.append(table.replace(rate_limiter=rate_limiter))
        # Target the middle of the rate limiter's insert-sample balance window.
        sample_sizes.append(
            max(1, int(
                (rate_limiter_info.max_diff - rate_limiter_info.min_diff) / 2)))
    return modified_tables, sample_sizes

"""Runs a simple, single-threaded training loop using the default evaluators.
It targets simplicity of the code and so only the basic features of the
ExperimentConfig are supported.
Arguments:
experiment: Definition and configuration of the agent to run.
eval_every: After how many actor steps to perform evaluation.
num_eval_episodes: How many evaluation episodes to execute at each
    evaluation step.
"""
experiment = experiment_config
eval_every=EVAL_STEPS
num_eval_episodes = 1


key = jax.random.PRNGKey(experiment.seed)

# Create the environment and get its spec.
environment = experiment.environment_factory(experiment.seed)
environment_spec = experiment.environment_spec or specs.make_environment_spec(
    environment)

# Create the networks and policy.
networks = experiment.network_factory(environment_spec)
policy = config.make_policy(
    experiment=experiment,
    networks=networks,
    environment_spec=environment_spec,
    evaluation=False)

# Create the replay server and grab its address.
replay_tables = experiment.builder.make_replay_tables(environment_spec,
                                                    policy)

# Disable blocking of inserts by tables' rate limiters, as this function
# executes learning (sampling from the table) and data generation
# (inserting into the table) sequentially from the same thread
# which could result in blocked insert making the algorithm hang.
replay_tables, rate_limiters_max_diff = _disable_insert_blocking(
    replay_tables)

replay_server = reverb.Server(replay_tables, port=None)
replay_client = reverb.Client(f'localhost:{replay_server.port}')

# Parent counter allows to share step counts between train and eval loops and
# the learner, so that it is possible to plot for example evaluator's return
# value as a function of the number of training episodes.
parent_counter = counting.Counter(time_delta=0.)

dataset = experiment.builder.make_dataset_iterator(replay_client)
# We always use prefetch as it provides an iterator with an additional
# 'ready' method.
dataset = utils.prefetch(dataset, buffer_size=1)

# Create actor, adder, and learner for generating, storing, and consuming
# data respectively.
# NOTE: These are created in reverse order as the actor needs to be given the
# adder and the learner (as a source of variables).
learner_key, key = jax.random.split(key)
learner = experiment.builder.make_learner(
    random_key=learner_key,
    networks=networks,
    dataset=dataset,
    logger_fn=experiment.logger_factory,
    environment_spec=environment_spec,
    replay_client=replay_client,
    counter=counting.Counter(parent_counter, prefix='learner', time_delta=0.))

adder = experiment.builder.make_adder(replay_client, environment_spec, policy)

actor_key, key = jax.random.split(key)
actor = experiment.builder.make_actor(
    actor_key, policy, environment_spec, variable_source=learner, adder=adder)

# Create the environment loop used for training.
train_counter = counting.Counter(
    parent_counter, prefix='actor', time_delta=0.)
train_logger = experiment.logger_factory('actor',
                                        train_counter.get_steps_key(), 0)



checkpointer = savers.Checkpointer(
    objects_to_save={
        'learner': learner,
        'counter': parent_counter
    },
    time_delta_minutes=experiment.checkpointing.time_delta_minutes,
    directory=experiment.checkpointing.directory,
    add_uid=experiment.checkpointing.add_uid,
    max_to_keep=experiment.checkpointing.max_to_keep)


ckpt = tf.train.Checkpoint(
    learner=acme.tf.savers.SaveableAdapter(learner),
    counter=acme.tf.savers.SaveableAdapter(parent_counter)
    )
mgr = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, 1)




if WEIGHTS in pretrained_weights:
    pretrained_models_url = f"https://huggingface.co/ChristianOrr/dqn/resolve/main/{WEIGHTS}_haiku/"

    data_utils.get_file(f"{CHECKPOINT_DIR}/{WEIGHTS}_haiku/checkpoint", f"{pretrained_models_url}checkpoint")
    data_utils.get_file(f"{CHECKPOINT_DIR}/{WEIGHTS}_haiku/ckpt-1.data-00000-of-00002", f"{pretrained_models_url}ckpt-1.data-00000-of-00002")
    data_utils.get_file(f"{CHECKPOINT_DIR}/{WEIGHTS}_haiku/ckpt-1.data-00001-of-00002", f"{pretrained_models_url}ckpt-1.data-00001-of-00002")
    data_utils.get_file(f"{CHECKPOINT_DIR}/{WEIGHTS}_haiku/ckpt-1.index", f"{pretrained_models_url}ckpt-1.index")

    # latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR + "{WEIGHTS}_haiku/")
    latest_ckpt = f"{CHECKPOINT_DIR}/{WEIGHTS}_haiku/ckpt-1"
    ckpt.restore(latest_ckpt).assert_consumed()
elif WEIGHTS is not None:
    latest_ckpt = tf.train.latest_checkpoint(WEIGHTS)
    ckpt.restore(latest_ckpt).assert_consumed()


# Replace the actor with a LearningActor. This makes sure that every time
# that `update` is called on the actor it checks to see whether there is
# any new data to learn from and if so it runs a learner step. The rate
# at which new data is released is controlled by the replay table's
# rate_limiter which is created by the builder.make_replay_tables call above.
actor = _LearningActor(actor, learner, dataset, replay_tables,
                        rate_limiters_max_diff, None #checkpointer
                        )

train_loop = acme.EnvironmentLoop(
    environment,
    actor,
    counter=train_counter,
    logger=train_logger,
    observers=experiment.observers)

max_num_actor_steps = (
    experiment.max_num_actor_steps -
    parent_counter.get_counts().get(train_counter.get_steps_key(), 0))


# Create the evaluation actor and loop.
eval_counter = counting.Counter(
    parent_counter, prefix='evaluator', time_delta=0.)
eval_logger = experiment.logger_factory('evaluator',
                                        eval_counter.get_steps_key(), 0)
eval_policy = config.make_policy(
    experiment=experiment,
    networks=networks,
    environment_spec=environment_spec,
    evaluation=True)
eval_actor = experiment.builder.make_actor(
    random_key=jax.random.PRNGKey(experiment.seed),
    policy=eval_policy,
    environment_spec=environment_spec,
    variable_source=learner)
eval_loop = acme.EnvironmentLoop(
    environment,
    eval_actor,
    counter=eval_counter,
    logger=eval_logger,
    observers=experiment.observers)

print("\n--------------------------")
print("Starting Training")
steps = 0
while steps < max_num_actor_steps:
    eval_loop.run(num_episodes=num_eval_episodes)
    steps += train_loop.run(num_steps=eval_every)
eval_loop.run(num_episodes=num_eval_episodes)
print("Training Complete!")
print("--------------------------\n")
# Save final checkpoint
ckpt.save(f"{CHECKPOINT_DIR}ckpt")


# Test the trained model
env = make_atari_environment(
        level=ENV_TYPE,
        sticky_actions=False,
        zero_discount_on_life_loss=True,
        render_mode="human"
    )

test_loop = acme.EnvironmentLoop(
    env,
    eval_actor)

print("\n--------------------------")
print("Starting Testing")
test_loop.run(num_episodes=1)
print("Testing Complete!")
print("--------------------------\n")

