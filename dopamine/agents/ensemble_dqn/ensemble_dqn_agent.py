"""Compact implementation of an ensemble dqn agent with prioritized replay.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib
from dopamine.replay_memory.circular_replay_buffer import ReplayElement
from dopamine.replay_memory import prioritized_replay_buffer
import gin.tf
import numpy as np
import os
import tensorflow as tf


@gin.configurable
class EnsembleDQNAgent(dqn_agent.DQNAgent):
  """A compact implementation of a simplified Ensemble DQN agent."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               representation_network=atari_lib.NatureDQNRepresentationNetwork,
               head_network=atari_lib.NatureDQNHeadNetwork,
               num_ensemble=1,
               rew_noise_scale=0.0,
               sample_ensemble_particles=False,
               add_prior_net=False,
               add_prior_values=False,
               reweight_loss=True,
               min_val=-100.,
               max_val=100.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               priority_type='loss',
               tf_device='/cpu:*',
               use_staging=False,
               optimizer=tf.compat.v1.train.AdamOptimizer(
                   learning_rate=0.00025, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500):
    assert priority_type in ['loss', 'td_error', 'variance_reduction']

    self.representation_network = representation_network
    self.head_network = head_network

    self._replay_scheme = replay_scheme
    self._priority_type = priority_type
    self._num_ensemble = num_ensemble
    self._rew_noise_scale = rew_noise_scale
    self._sample_ensemble_particles = sample_ensemble_particles
    self._add_prior_net = add_prior_net
    self._add_prior_values = add_prior_values
    self._reweight_loss = reweight_loss
    self._min_val = min_val
    self._max_val = max_val
    # TODO(b/110897128): Make agent optimizer attribute private.
    self.optimizer = optimizer

    super().__init__(
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def _create_representation_net(self, name):
    return self.representation_network(name=name)

  def _create_head(self, name):
    return self.head_network(self.num_actions, name=name)

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A `WrappedPrioritizedReplayBuffer` object.

    Raises:
      ValueError: if given an invalid replay scheme.
    """
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    # Both replay schemes use the same data structure, but the 'uniform' scheme
    # sets all priorities to the same value (which yields uniform sampling).

    extra_storage_types = [ReplayElement('reward_noise', (self._num_ensemble,), np.float32)]

    return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype,
        extra_storage_types=extra_storage_types)

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """
    # _network_template instantiates the model and returns the network object.
    # The network object can be used to generate different outputs in the graph.
    # At each call to the network, the parameters will be reused.

    # Construct the online, target, and prior state representations.
    self.online_representation_net = self._create_representation_net(name='Online_Rep')
    self.target_representation_net = self._create_representation_net(name='Target_Rep')
    self.prior_representation_net = self._create_representation_net(name='Prior_Rep')

    # Cosntruct the online, target, and prior heads.
    self.online_heads = [self._create_head(name='Online_Head_{}'.format(i)) for i in range(self._num_ensemble)]
    self.target_heads = [self._create_head(name='Target_Head_{}'.format(i)) for i in range(self._num_ensemble)]
    self.prior_heads = [self._create_head(name='Prior_Head_{}'.format(i)) for i in range(self._num_ensemble)]

    # Construct the Q-values used for selecting actions - average over ensemble outputs.
    representation = self.online_representation_net(self.state_ph).representation
    prior_representation = self.prior_representation_net(self.state_ph).representation

    self._all_net_q_values = tf.concat(
        [self.online_heads[i](representation).q_values + self.prior_heads[i](prior_representation).q_values for i in range(
            self._num_ensemble)], axis=0)
    self._net_q_values = tf.reduce_mean(self._all_net_q_values, axis=0)[None, :]
    # TODO(bellemare): Ties should be broken. They are unlikely to happen when
    # using a deep network, but may affect performance with a linear
    # approximation scheme.
    self._q_argmax = tf.argmax(self._net_q_values, axis=1)[0]

    # Construct the replay state and next state q-values for each ensemble member.
    replay_representation = self.online_representation_net(
        self._replay.states).representation
    self._replay_net_q_values = [self.online_heads[i](
        replay_representation).q_values for i in range(self._num_ensemble)]
    
    replay_prior_representation = self.prior_representation_net(
        self._replay.states).representation
    self._replay_prior_net_q_values = [self.prior_heads[i](
        replay_prior_representation).q_values for i in range(self._num_ensemble)]

    replay_next_target_representation = self.target_representation_net(
        self._replay.next_states).representation
    self._replay_next_target_net_q_values = [self.target_heads[i](
        replay_next_target_representation).q_values for i in range(self._num_ensemble)]

    replay_next_prior_representation = self.prior_representation_net(
        self._replay.next_states).representation
    self._replay_next_prior_net_q_values = [self.prior_heads[i](
        replay_next_prior_representation).q_values for i in range(self._num_ensemble)]
    return

  def _build_target_q_op(self):
      # Get the maximum Q-value across the actions dimension.
      if self._add_prior_net:
        replay_next_qt_max = [tf.reduce_max(
            self._replay_next_target_net_q_values[i] + self._replay_next_prior_net_q_values[i], 1) for i in range(self._num_ensemble)]
      else:
        replay_next_qt_max = [tf.reduce_max(
            self._replay_next_target_net_q_values[i], 1) for i in range(self._num_ensemble)]

      if self._add_prior_values:
        delta = ((self._max_val - self._min_val) / self._num_ensemble)
        replay_next_qt_max = [replay_next_qt_max[i] + tf.stop_gradient( 
            (self._min_val + delta * i) * tf.ones_like(replay_next_qt_max[i])) for i in range(self._num_ensemble)]
      
      # Calculate the Bellman target value.
      #   Q_t = R_t + \gamma^N * Q'_t+1
      # where,
      #   Q'_t+1 = \argmax_a Q(S_t+1, a)
      #          (or) 0 if S_t is a terminal state,
      # and
      #   N is the update horizon (by default, N=1).
      return [self._replay.rewards + tf.squeeze(self._replay.rew_noise[:, i]) + self.cumulative_gamma * replay_next_qt_max[i] * (
          1. - tf.cast(self._replay.terminals, tf.float32)) for i in range(self._num_ensemble)]

  def _build_train_op(self):
      replay_action_one_hot = tf.one_hot(
          self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    
      if self._add_prior_net:
        replay_chosen_q = [tf.reduce_sum(
            (self._replay_net_q_values[i] + tf.stop_gradient(self._replay_prior_net_q_values[i])) * replay_action_one_hot,
            axis=1,
            name='replay_chosen_q') for i in range(self._num_ensemble)]
      else:
        replay_chosen_q = [tf.reduce_sum(
            self._replay_net_q_values[i] * replay_action_one_hot,
            axis=1,
            name='replay_chosen_q') for i in range(self._num_ensemble)]

      target_qs = self._build_target_q_op()
      target = tf.stop_gradient([target_qs[i] for i in range(self._num_ensemble)])

      if self._add_prior_values:
        delta = (self._max_val - self._min_val) / self._num_ensemble
        replay_chosen_q = [replay_chosen_q[i] + tf.stop_gradient(
            (self._min_val + delta * i) * tf.ones_like(replay_chosen_q[i])) for i in range(self._num_ensemble)]

      concat_replay_chosen_q = tf.stack(replay_chosen_q)
      concat_target = tf.concat(target, axis=0)

      if self._sample_ensemble_particles:
        ensemble_indices = tf.random.uniform(
            shape=(self._num_ensemble,),
            minval=0,
            maxval=self._num_ensemble,
            dtype=tf.dtypes.int32)
        
        concat_replay_chosen_q = tf.gather(concat_replay_chosen_q, ensemble_indices, axis=0)
        concat_target = tf.gather(concat_target, ensemble_indices, axis=0)

      loss = tf.compat.v1.losses.huber_loss(concat_target, concat_replay_chosen_q, reduction=tf.losses.Reduction.NONE)

      # Axis 0 is the ensemble axis.
      loss_for_priority = tf.reduce_mean(loss, axis=0)
      loss = tf.reduce_sum(loss, axis=0)

      if self._replay_scheme == 'prioritized':
          # The original prioritized experience replay uses a linear exponent
          # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
          # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
          # a fixed exponent actually performs better, except on Pong.
          probs = self._replay.transition['sampling_probabilities']
          loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
          loss_weights /= tf.reduce_max(loss_weights)

          # Rainbow and prioritized replay are parametrized by an exponent alpha,
          # but in both cases it is set to 0.5 - for simplicity's sake we leave it
          # as is here, using the more direct tf.sqrt(). Taking the square root
          # "makes sense", as we are dealing with a squared loss.
          # Add a small nonzero value to the loss to avoid 0 priority items. While
          # technically this may be okay, setting all items to 0 priority will cause
          # troubles, and also result in 1.0 / 0.0 = NaN correction terms.
          if self._priority_type == 'loss':
              priorities = tf.sqrt(loss_for_priority + 1e-10)
          
          # TODO(saurabh): Implement standard TD error based prioritization.
          elif self._priority_type == 'td_error':
              target_avg = tf.reduce_mean(tf.concat(target, axis=0), axis=0)
              pred_avg = tf.reduce_mean(tf.stack(replay_chosen_q), axis=0)
              priorities = tf.compat.v1.losses.huber_loss(pred_avg, target_avg, 
                  reduction=tf.losses.Reduction.NONE)

          # TODO(saurabh): Implement variance reduction based prioritization.
          # Specifically, compute the variance reduction expression using the optimal alpha.
          elif self._priority_type == 'variance_reduction':
              target_variances = tf.math.reduce_variance(tf.concat(target, axis=0), axis=0)
              curr_variances = tf.math.reduce_variance(tf.stack(replay_chosen_q), axis=0)

              # Compute covariances
              mean_chosen_q = tf.reduce_mean(concat_replay_chosen_q, axis=0, keepdims=True)
              mean_target = tf.reduce_mean(concat_target, axis=0, keepdims=True)

              shifted_chosen_q = concat_replay_chosen_q - mean_chosen_q
              shifted_target = concat_target - mean_target

              # -1 for bias correction
              covariances = tf.reduce_sum(shifted_chosen_q * shifted_target, axis=0) / (self._num_ensemble - 1)
              
              # Find optimum alpha
              numerator = curr_variances - covariances
              denominator = curr_variances + target_variances - 2 * covariances

              optimized_alpha = numerator / denominator
              optimized_alpha *= 1 - tf.cast(tf.math.is_nan(optimized_alpha), tf.float32)
              optimized_alpha = tf.clip_by_value(optimized_alpha, 0, 1)
              
              # Below are old checks used. Should be equivalent to above.
              # optimized_alpha[denominator == 0] = 1.
              # optimized_alpha[numerator == 0] = 0.
              # optimized_alpha[optimized_alpha > 1] = 1.
              # optimized_alpha[optimized_alpha < 0] = 0.

              priorities = (1. - tf.square(1. - optimized_alpha)) * curr_variances - tf.square(
                  optimized_alpha) * target_variances - 2. * optimized_alpha * (1. - optimized_alpha) * covariances  
              priorities = tf.math.maximum(tf.math.maximum(priorities, 1e-100 * tf.ones_like(priorities)), curr_variances - target_variances)

          update_priorities_op = self._replay.tf_set_priority(
              self._replay.indices, priorities)

          # Weight the loss by the inverse priorities.
          if self._reweight_loss:
            loss = loss_weights * loss
      else:
          update_priorities_op = tf.no_op()

      if self.summary_writer is not None:
          with tf.compat.v1.variable_scope('Losses'):
              tf.compat.v1.summary.scalar('HuberLoss', tf.reduce_mean(loss))

      # Optimization ops.
      opt_ops = []

      # Get the representation network vars.
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      trainables_online_rep = tf.compat.v1.get_collection(
          tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
          scope=os.path.join(scope, 'Online_Rep'))
      
      if len(trainables_online_rep) > 0:
        # Compute the gradients for a list of variables.
        grads_and_vars = self.optimizer.compute_gradients(loss, trainables_online_rep)

        # grads_and_vars is a list of tuples (gradient, variable).  Divide gradents by ensemble size.
        normalized_grads_and_vars = [(gv[0] / float(self._num_ensemble), gv[1]) for gv in grads_and_vars]

        # Ask the optimizer to apply the normalized gradients.
        opt_ops.append(self.optimizer.apply_gradients(normalized_grads_and_vars))

      # Do the same for each network head, but without scaling.
      for i in range(self._num_ensemble):
          trainables_online_head = tf.compat.v1.get_collection(
              tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
              scope=os.path.join(scope, 'Online_Head_{}'.format(i)))

          # Compute the gradients with respect to the head network variables.
          grads_and_vars = self.optimizer.compute_gradients(loss, trainables_online_head)

          # Ask the optimizer to apply the normalized gradients.
          opt_ops.append(self.optimizer.apply_gradients(grads_and_vars))

      return opt_ops

  def _build_sync_op(self):
      # First do the representation networks.

      # Get trainable variables from online and target DQNs
      sync_qt_ops = []

      rep_ops = []
      scope = tf.compat.v1.get_default_graph().get_name_scope()
      trainables_online = tf.compat.v1.get_collection(
          tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
          scope=os.path.join(scope, 'Online_Rep'))
      trainables_target = tf.compat.v1.get_collection(
          tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
          scope=os.path.join(scope, 'Target_Rep'))

      for (w_online, w_target) in zip(trainables_online, trainables_target):
          # Assign weights from online to target network.
          sync_qt_ops.append(w_target.assign(w_online, use_locking=True))
      
      # Now do the head networks.

      for i in range(self._num_ensemble):
          trainables_online = tf.compat.v1.get_collection(
              tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
              scope=os.path.join(scope, 'Online_Head_{}'.format(i)))
          trainables_target = tf.compat.v1.get_collection(
              tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
              scope=os.path.join(scope, 'Target_Head_{}'.format(i)))

          for (w_online, w_target) in zip(trainables_online, trainables_target):
              # Assign weights from online to target network.
              sync_qt_ops.append(w_target.assign(w_online, use_locking=True))

      return sync_qt_ops

  def step(self, reward, observation):
      self._last_observation = self._observation
      self._record_observation(observation)

      if not self.eval_mode:

          rew_noise = np.random.normal(loc=0, scale=self._rew_noise_scale, size=self._num_ensemble)

          self._store_transition(self._last_observation, self.action, reward, False, rew_noise=rew_noise)
          self._train_step()

      self.action = self._select_action()
      return self.action

  def end_episode(self, reward):
      """Signals the end of the episode to the agent.
      We store the observation of the current time step, which is the last
      observation of the episode.
      Args:
      reward: float, the last reward from the environment.
      """

      if not self.eval_mode:
          rew_noise = np.random.normal(loc=0, scale=self._rew_noise_scale, size=self._num_ensemble)
          self._store_transition(self._observation, self.action, reward, True, rew_noise=rew_noise)

  def _store_transition(self,
                      last_observation,
                      action,
                      reward,
                      is_terminal,
                      rew_noise=None,
                      priority=None):
      
      if priority is None:
          if self._replay_scheme == 'uniform':
              priority = 1.
          else:
              priority = self._replay.memory.sum_tree.max_recorded_priority

      if not self.eval_mode:
          self._replay.add(last_observation, action, reward, is_terminal, rew_noise, priority)
