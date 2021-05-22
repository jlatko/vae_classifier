import torch
import pyro

cross_entropy = torch.nn.CrossEntropyLoss()
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

def get_losses(vae, y, x, x_unsupervised):
  # TODO: to average or not to average?
  # classification
  if x is not None:
      n = len(x)
      y_logits = vae.classifier(x)
      loss_class = cross_entropy(y_logits, y)

      # supervised ELBO
      loss_sup = loss_fn(
          vae.model_supervised,
          vae.guide_supervised,
          x, y) / n # AVERAGE !!!
  else:
    y_logits = None
    loss_class=0
    loss_sup = 0

  # unsupervised ELBO (missing y)
  if x_unsupervised is not None:
    n = len(x_unsupervised)
    loss_unsup = loss_fn(
        vae.model_unsupervised,
        vae.guide_unsupervised,
        x_unsupervised) / n # AVERAGE !!!
  else:
    loss_unsup = 0

  return loss_unsup, loss_sup, loss_class, y_logits
