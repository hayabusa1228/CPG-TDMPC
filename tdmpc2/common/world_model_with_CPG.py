import torch

from common import math, init
from common.world_model import WorldModel
from common.CPG import CPG


class WorldModelWithCPG(WorldModel):
	def __init__(self, cfg):
		super().__init__(cfg)
		self._cpg = CPG(cfg)
		# CPGの重みを初期化
		self._cpg.apply(init.weight_init)
	
	def pi(self, z, cpg_states,task, eval_mode=False):
		"""
		Samples an action from the policy prio,ion with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		# CPGに行動決定させる
		cpg_states, log_pi_p, log_pi_pw = self._cpg.update_states(z, cpg_states, task, eval_mode)
		mu, log_std = self._cpg.action(z,cpg_states, task)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mu)

		if self.cfg.multitask: # Mask out unused action dimensions
			mu = mu * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
		pi = mu + eps * log_std.exp()
		mu, pi, log_pi = math.squash(mu, pi, log_pi)

		if eval_mode:
			pi = mu

		return mu, pi, log_pi, log_std, log_pi_p, log_pi_pw, cpg_states
