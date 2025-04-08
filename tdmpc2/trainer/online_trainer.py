from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self.best_train_reward = -float('inf')
		self.add_noise = False
		if self.cfg.checkpoint != '':
			self.agent.load(self.cfg.checkpoint)
			print(f'Loaded checkpoint: {self.cfg.checkpoint}')

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		# 検証して報酬と成功率を計算
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.use_cpg:
				if obs[3]  > 0:
					# 左足前なら0で初期化
					cpg_states = torch.full((self.cfg.action_dim,), np.pi/2)
				else:
					# 右足前ならpiで初期化
					cpg_states = torch.full((self.cfg.action_dim,), -np.pi/2)
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				if self.cfg.use_cpg:
					action, cpg_states = self.agent.act(obs, cpg_states, t0=t==0, eval_mode=True)
				else:
					action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)

	def to_td(self, obs, cpg_states=None, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if self.cfg.use_cpg:
			td = TensorDict(dict(
				obs=obs,
				cpg_states=cpg_states.unsqueeze(0),
				action=action.unsqueeze(0),
				reward=reward.unsqueeze(0),
			), batch_size=(1,))
		else:
			td = TensorDict(dict(
				obs=obs,
				action=action.unsqueeze(0),
				reward=reward.unsqueeze(0),
			), batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, True
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True
			
			if self._step == self.cfg.add_noise_step:
				self.add_noise = True
				self.logger.save_agent(self.agent, 'pretrain')
				print('add noise start!!')
				
			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					# if eval_metrics['episode_reward'] > self.best_eval_reward and self._step > self.cfg.add_noise_step + 10000:
					# 	self.best_eval_reward = eval_metrics['episode_reward']
					# 	self.logger.save_agent(self.agent, 'best_eval')
					# 	print(f'best model saved: {self.best_eval_reward:.2f}')
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
					)
					if train_metrics['episode_reward'] > self.best_train_reward and self._step > self.cfg.add_noise_step + 30000:
						self.best_train_reward = train_metrics['episode_reward']
						self.logger.save_agent(self.agent, 'best')
						print(f'best model saved: {self.best_train_reward:.2f}')
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				# obsにノイズを加える
				if self.add_noise:
					obs += torch.randn_like(obs) * self.cfg.obs_noise_scale
				if self.cfg.use_cpg:
					if obs[3]  > 0:
						cpg_states = torch.full((self.cfg.action_dim,), np.pi/2)
					else:
						cpg_states = torch.full((self.cfg.action_dim,), -np.pi/2)
					self._tds = [self.to_td(obs, cpg_states)]
				else:
					self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps or self.cfg.checkpoint != '':
				if self.cfg.use_cpg:
					action, cpg_states = self.agent.act(obs, cpg_states, t0=len(self._tds)==1)
				else:
					action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			# obsにノイズを加える
			if self.add_noise:
				obs += torch.randn_like(obs) * self.cfg.obs_noise_scale
			if self.cfg.use_cpg:
				self._tds.append(self.to_td(obs, cpg_states.squeeze(0), action, reward))
			else:
				self._tds.append(self.to_td(obs, action=action, reward=reward))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps and self.cfg.checkpoint == '':
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for i in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1
	    
		# model save
		self.logger.finish(self.agent)

