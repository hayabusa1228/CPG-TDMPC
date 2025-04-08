import numpy as np
import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.world_model_with_CPG import WorldModelWithCPG



class TDMPC2:
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self.use_cpg = cfg.use_cpg
		self.device = torch.device('cuda')
		self.model =  WorldModelWithCPG(cfg).to(self.device)  if self.use_cpg else WorldModel(cfg).to(self.device) 
		# 世界モデルのパラメータを最適化
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			# Q関数のアンサンブル　
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
		], lr=self.cfg.lr)
		# 方策のパラメータを最適化
		if self.use_cpg:
			self.pi_optim = torch.optim.Adam([
				{'params': self.model._cpg._ap.parameters()},
				{'params': self.model._cpg._apw.parameters()},
				{'params': self.model._cpg._ar.parameters()},
			], lr=self.cfg.lr)
		else:
			self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.
		
		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.
		
		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(state_dict["model"])

	@torch.no_grad()
	def act(self, obs, cpg_states=None, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.
		
		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).
		
		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		z = self.model.encode(obs, task)
		if self.cfg.mpc:
			# MPC planning
			if self.use_cpg:
				a, cpg_states = self.plan(z, cpg_states, t0=t0, eval_mode=eval_mode, task=task)
				return a.cpu(), cpg_states.cpu()
			else:
				a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
				return a.cpu()
		else:
			# planningなしで直接行動を選択
			# 1番目と6番目の返り値を取得
			if self.use_cpg:
				cpg_states = cpg_states.unsqueeze(0) if cpg_states.dim() == 1 else cpg_states
				result = self.model.pi(z, cpg_states, task)
				a, cpg_states = result[int(not eval_mode)][0], result[6][0]
				return a.cpu(), cpg_states.cpu()
			else:
				a = self.model.pi(z, task)[int(not eval_mode)][0]
				return a.cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		if self.use_cpg:
			for t in range(self.cfg.horizon-1):
				reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
				z = self.model.next(z, actions[t], task)
				G += discount * reward
				discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			# 探索を1step減らして、確実な情報だけ使うようにした
			return G + discount * self.model.Q(z, actions[self.cfg.horizon-1], task, return_type='avg')
		else:
			for t in range(self.cfg.horizon):
				reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
				z = self.model.next(z, actions[t], task)
				G += discount * reward
				discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

	@torch.no_grad()
	def plan(self, z, cpg_states =None,t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.
		
		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""		
		# Sample policy trajectories
		# 行動plamning
		# 方策から得た行動と正規分布からサンプリングした行動を併用して学習する
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			if self.use_cpg:
				_cpg_states = cpg_states.repeat(self.cfg.num_pi_trajs, 1)
				for t in range(self.cfg.horizon-1):
					_, pi_actions[t] ,_, _, _, _, _cpg_states = self.model.pi(_z, _cpg_states, task, eval_mode)
					if t == 0:
						new_cpg_states = _cpg_states
					_z = self.model.next(_z, pi_actions[t], task)
				_, pi_actions[-1],_, _, _, _, _cpg_states = self.model.pi(_z, _cpg_states, task, eval_mode)
			else:
				for t in range(self.cfg.horizon):
					pi_actions[t] = self.model.pi(_z, task)[1]
					_z = self.model.next(_z, pi_actions[t], task)
				pi_actions[-1] = self.model.pi(_z, task)[1]

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			# num_pi_trajs個の方策から得た行動を使う　それ以降は正規
			actions[:, :self.cfg.num_pi_trajs] = pi_actions
	
		# Iterate MPPI　meanとstdを学習して正規分布からサンプリング
		for _ in range(self.cfg.iterations):

			# Sample actions
			# repameterization trick
			# num_pi_trajs以降は正規分布からサンプリング
			actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
				.clamp(-1, 1)
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			if self.use_cpg:
				# 正規分布からのサンプリングにもcpg_states_after_horizonを使っているのは怪しい
				value = self._estimate_value(z, actions, task).nan_to_num_(0)
			else:
				value = self._estimate_value(z, actions, task).nan_to_num_(0)
			# 最も良い行動とそのインデックスを取得
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score /= score.sum(0)
			mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
				.clamp_(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		score = score.squeeze(1).cpu().numpy()
		# 高いスコアの方が選ばれやすいサンプリングで行動選択(p=score) 
		rand_idx = np.random.choice(np.arange(score.shape[0]), p=score)
		actions = elite_actions[:, rand_idx]
		action_idx = elite_idxs[rand_idx]

		# 選ばれた行動が方策から得た行動であれば、その時のcpg_statesを使う
		if self.use_cpg:
			if action_idx < self.cfg.num_pi_trajs:
				new_cpg_states = new_cpg_states[action_idx]
			else:
				# ランダム行動の場合はnew_cpg_statesの平均を使う　若干怪しい
				new_cpg_states = new_cpg_states.mean(0)

		self._prev_mean = mean
		# 結局ホライズンステップ分で評価をするが、最もよい最初の行動だけ返す
		a, std = actions[0], std[0]
		if not eval_mode:
			# Add noise to action during training
			a += std * torch.randn(self.cfg.action_dim, device=std.device)
		# cpgの状態遷移は行動に依存しないので、ここで改めて更新する
		# ランダム行動でもcpg_statesは更新されるのが怪しい
		if self.use_cpg:
			return a.clamp_(-1, 1), new_cpg_states
		else:
			return a.clamp_(-1, 1)
		
	def update_pi(self, zs,task, cpg_states=None):
		"""
		Update policy using a sequence of latent states.
		
		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""

		# CPGを使う場合の場合分けがいりそう ここだけやればどうにかなりそう
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		# 敵対的ノイズを加える
		zs += torch.randn_like(zs) * self.cfg.latent_noise_scale

		if self.use_cpg:
			self.model._cpg.train()
			# CPGを使う場合はCPGのap apwのエントロピーも考慮する

			# バッファーにあるzとCPG状態をスタートから一ステップだけ行動を決定
			pis = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.action_dim, device=self.device)
			log_pis = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.action_dim, device=self.device)
			ap_log_pis = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.action_dim, device=self.device)
			apw_log_pis = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.action_dim, device=self.device)

			# cpg_stateにも敵対的ノイズを加える
			# cpg_states += cpg_states + torch.randn_like(cpg_states) * self.cfg.cpg_noise_scale
			for t in range(self.cfg.horizon+1):
				_, pis[t], log_pis[t], _, ap_log_pis[t], apw_log_pis[t], _ = self.model.pi(zs[t], cpg_states[t],task)
			
			qs = self.model.Q(zs, pis, task, return_type='avg')
			self.scale.update(qs[0])

			# Loss is a weighted sum of Q-values
			rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
			pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
			ap_loss = (self.cfg.cpg_entropy_coef * ap_log_pis.mean(dim=(1,2)) * rho).mean()
			apw_loss = (self.cfg.cpg_entropy_coef * apw_log_pis.mean(dim=(1,2)) * rho).mean()
			pi_loss += ap_loss + apw_loss
			pi_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model._cpg.parameters(), self.cfg.grad_clip_norm)

			self.pi_optim.step()
			self.model.track_q_grad(True)
			self.model._cpg.eval()

		else: 
			_, pis, log_pis, _ = self.model.pi(zs, task)
			qs = self.model.Q(zs, pis, task, return_type='avg')
			self.scale.update(qs[0])
			qs = self.scale(qs)

			# Loss is a weighted sum of Q-values
			rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
			pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
			pi_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
			self.pi_optim.step()
			self.model.track_q_grad(True)

		return pi_loss.item()

	@torch.no_grad()
	def _td_target(self, next_z, reward, task ,cpg_states=None):
		"""
		Compute the TD-target from a reward and the observation at the following time step.
		
		Args:
			next_z (torch.Tensor): Latent state at the following time step. (horizon, batch_size, latent_dim)
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).
			cpg_states (torch.Tensor): CPG states at the following time step. (horizon, batch_size, cpg_dim)
		
		Returns:
			torch.Tensor: TD-target.
		"""

		if self.use_cpg:
			# next_zそれぞれに対してcpg状態を考慮して行動を選択して報酬を計算
			pi = torch.empty(self.cfg.horizon, self.cfg.batch_size, self.cfg.action_dim, device=self.device)
			for t in range(self.cfg.horizon):
				pi[t] = self.model.pi(next_z[t], cpg_states[t], task)[1]
		else:
			pi = self.model.pi(next_z, task)[1]
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.
		
		Args:
			buffer (common.buffer.Buffer): Replay buffer.
		
		Returns:
			dict: Dictionary of training statistics.

		"""

		if self.use_cpg:
			obs, cpg_states, action, reward, task = buffer.sample()
		else:
			obs, action, reward, task = buffer.sample()

		# obsにノイズを加える
		# obs += torch.randn_like(obs) * self.cfg.obs_noise_scale

		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			if self.use_cpg:
				td_targets = self._td_target(next_z,reward, task, cpg_states[1:])
			else:
				td_targets = self._td_target(next_z, reward, task)

		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t in range(self.cfg.horizon):
			z = self.model.next(z, action[t], task)
			# ダイナミクスの一貫性を保つための損失
			consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		
		# Compute 　ソフトクロスエントロピー損失
		reward_loss, value_loss = 0, 0
		for t in range(self.cfg.horizon):
			reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
			for q in range(self.cfg.num_q):
				# 方策エントロピー
				value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
		consistency_loss *= (1/self.cfg.horizon)
		reward_loss *= (1/self.cfg.horizon)
		value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()

		# Update policy
		if self.use_cpg:
			pi_loss = self.update_pi(zs.detach(), task, cpg_states)
		else:
			pi_loss = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value),
		}

