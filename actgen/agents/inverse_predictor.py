import torch

from ..models import InverseModel

class InversePredictor():
	"""
	uses an inverse model to predict which of the actions are similar 
	(in the sense that they lead to similar states after action execution)
	"""
	def __init__(self, n_actions, params, discrete=True):
		self.inv_model = InverseModel(n_actions, params, discrete).to(params['device'])
		self.params = params

		optim_params = list(self.inv_model.parameters())
		self.optimizer = torch.optim.Adam(optim_params, lr=self.params['learning_rate'])
	
	def save(self, name, model_dir, is_best):
		self.inv_model.save(name, model_dir, is_best)

	def predict(self, batch, encoder):
		s = torch.stack(batch.state).float().to(self.params['device'])
		sp = torch.stack(batch.next_state).float().to(self.params['device'])
		# inverse model takes in embeddings, not states
		z = encoder(s)
		zp = encoder(sp)
		logits = self.inv_model.forward(z, zp)
		probs = torch.nn.functional.softmax(logits, dim=-1)
		return probs
	
	def update(self, batch, encoder):
		self.inv_model.train()
		self.optimizer.zero_grad()

		s = torch.stack(batch.state).to(self.params['device'])
		a = torch.stack(batch.action).to(self.params['device'])
		sp = torch.stack(batch.next_state).to(self.params['device'])

		# detach the gradients of the encodings so that the encoder weights aren't affected
		z = encoder(s).detach()
		zp = encoder(sp).detach()
		loss = self.inv_model.compute_loss(z, a, zp)

		loss.backward()
		self.optimizer.step()

		return loss.detach().item()
