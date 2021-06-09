import torch

from ..models import InverseModel

class InversePredictor():
	"""
	uses an inverse model to predict which of the actions are similar 
	(in the sense that they lead to similar states after action execution)
	"""
	def __init__(self, n_actions, params, discrete=True):
		self.inv_model = InverseModel(n_actions, params, discrete)
		self.params = params

		optim_params = list(self.inv_model.parameters())
		self.optimizer = torch.optim.Adam(optim_params, lr=self.params['learning_rate'])
	
	def save(self, name, model_dir, is_best):
		self.inv_model.save(name, model_dir, is_best)

	def predict(self, s, sp, encoder):
		# inverse model takes in embeddings, not states
		z = encoder(s)
		zp = encoder(sp)
		return self.inv_model.forward(z, zp)
	
	def update(self, batch, encoder):
		self.inv_model.train()
		self.optimizer.zero_grad()

		s = torch.stack(batch.state)
		a = torch.stack(batch.action)
		sp = torch.stack(batch.next_state)

		# detach the gradients of the encodings so that the encoder weights aren't affected
		z = encoder(s).detach()
		zp = encoder(sp).detach()
		loss = self.inv_model.compute_loss(z, a, zp)

		loss.backward()
		self.optimizer.step()

		return loss.detach().item()
