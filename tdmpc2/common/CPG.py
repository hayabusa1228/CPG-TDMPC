from common import layers, math


import torch
import torch.nn as nn

# 世界モデルの潜在表現をCPGの内部状態に変換する

class CPG(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.omega_bias = torch.tensor(cfg.omega_bias).to(torch.device('cuda'))
		self.time_step = torch.tensor(cfg.time_step).to(torch.device('cuda'))
		self.K = torch.tensor(cfg.K).to(torch.device('cuda'))
		self.omega_scale = torch.tensor(cfg.omega_scale).to(torch.device('cuda'))
		self._ap = layers.mlp(cfg.latent_dim + cfg.action_dim*2 , 2*[cfg.cpg_mlp_dim], 2*cfg.action_dim)
		self._apw = layers.mlp(cfg.latent_dim + cfg.action_dim*2, 2*[cfg.cpg_mlp_dim], 2*cfg.action_dim)
		self._ar = layers.mlp(cfg.latent_dim+2*cfg.action_dim, 2*[cfg.cpg_mlp_dim], 2*cfg.action_dim)
		
	# CPGの内部状態を更新する
	#  x: 世界モデルの潜在表現 size: [batch_size, latent_dim] horizonのforは関数の呼び出し元で行う
	def update_states(self, x, prev_states, task=None, eval_mode=False):
		"""
		Forward pass through the model.
		"""
		log_std_dif = torch.tensor(self.cfg.log_std_max) - self.cfg.log_std_min
		log_std_min = torch.tensor(self.cfg.log_std_min)

		# CPG更新信号 ap apw
		# [batch_size, action_dim]
		# xとprev_statesを結合してCPGの内部状態を更新 
		z = torch.cat([x.to(torch.device('cuda')), torch.cos(prev_states.to(torch.device('cuda'))), torch.sin(prev_states.to(torch.device('cuda')))], dim=-1).to(torch.device('cuda'))
		mu_p, log_std_p = self._ap(z).chunk(2, dim=-1)
		mu_pw, log_std_pw  = self._apw(z).chunk(2, dim=-1)

		if not eval_mode:
			log_std_p = math.log_std(log_std_p, log_std_min, log_std_dif)
			eps_p = torch.randn_like(mu_p)
			log_pi_p = math.gaussian_logprob(eps_p, log_std_p, size=self.cfg.action_dim)
			ap = mu_p + eps_p * log_std_p.exp()
			mu_p, ap, log_pi_p = math.squash(mu_p, ap, log_pi_p)
			ap = ap.to(torch.device('cuda')) 
			
			log_std_pw = math.log_std(log_std_pw, log_std_min, log_std_dif)
			eps_pw = torch.randn_like(mu_pw)
			log_pi_pw = math.gaussian_logprob(eps_pw, log_std_pw, size=self.cfg.action_dim)
			apw = mu_pw + eps_pw * log_std_pw.exp()
			mu_pw, apw, log_pi_pw = math.squash(mu_pw, apw, log_pi_pw)
			apw = apw.to(torch.device('cuda'))
		else:
			ap = mu_p.to(torch.device('cuda'))
			apw = mu_pw.to(torch.device('cuda'))
			log_pi_p = torch.zeros_like(mu_p)
			log_pi_pw = torch.zeros_like(mu_pw)

		# CPGの内部状態を更新
		states = self.update(ap, apw, prev_states)
		return states, log_pi_p, log_pi_pw

	# CPGの内部状態から行動を決定する
	def action(self, x, cpg_states, task=None):

		z = torch.cat([x.to(torch.device('cuda')), torch.cos(cpg_states.to(torch.device('cuda'))), torch.sin(cpg_states.to(torch.device('cuda')))], dim=-1).to(torch.device('cuda'))
		# 行動信号　ar
		mu, log_std = self._ar(z).chunk(2, dim=-1)
		return mu, log_std
	
	# prev_states: [batch_size, action_dim]
	def update(self, ap, apw, prev_states):
		"""
		CPGの内部状態を更新
		"""

		prev_states = prev_states.to(torch.device('cuda'))

		states = torch.zeros(prev_states.shape).to(torch.device('cuda'))
		
		sum_sin = torch.zeros(prev_states.shape).to(torch.device('cuda'))
		sum_sin = torch.sum(torch.sin(prev_states.unsqueeze(1) - prev_states.unsqueeze(2)), dim=2)

		# for i in range(self.cfg.action_dim):
		# 	states[:, i] += self.time_step * (self.omega_bias + self.omega_scale * apw[:, i] + self.K * sum_sin[:, i] +  ap[:, i])
		# 	# [0, 2pi]に収める ここも変更した
		# 	states[:, i] = torch.fmod(states[:, i], 2 * torch.pi)
		# print("ds",  self.time_step * (self.omega_bias + self.omega_scale * apw + self.K * sum_sin +  ap)[0])
		states = prev_states + self.time_step * (self.omega_bias + self.omega_scale * apw + self.K * sum_sin +  ap)
		# states = torch.fmod(states, 2 * torch.pi)

		# [-pi, pi]に収める ex) 2pi -> 0  pi + 1 -> -pi + 1  -pi - 1 -> pi - 1
		states = torch.where(states > torch.pi, -torch.pi + torch.fmod(states, torch.pi), states)
		states = torch.where(states < -torch.pi, torch.pi - torch.fmod(-states, torch.pi), states)


	    # statesが[-pi, pi]に収まらなければエラーを出す
		# if torch.any(states+torch.pi < 0):
		# 	print("error -piを下回る")
		

		
		return states

