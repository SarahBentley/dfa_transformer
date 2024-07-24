import torch
import wandb
from .inference_bilinear import infer
from ..utils.metrics import train_eval_metrics
from ..utils.visualizations import visualize_final_metrics

# Create training loop
def train_bilinear(model, optimizer, config):

  criterion = torch.nn.CrossEntropyLoss()
  train_loss = []
  train_ce_loss = []
  train_l1_loss = []

  for epoch in range(config.num_epochs):
    model.train()
    # Generate a training example:
    input, output = config.dataloader.generate()

    total_loss = 0
    total_ce_loss = 0
    total_l1_loss = 0

    # Predict next 'token' for each character in training example
    for i in range(len(input)):
      symbol = input[i]
      last_state = output[i]
      pred = model(last_state, symbol)
      target = output[i+1]

      l1_loss = config.l1_penalty * torch.norm(model.bilinear.weight, p=1)
      ce_loss = criterion(pred, target)

      loss = ce_loss + l1_loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      total_ce_loss += ce_loss.item()
      total_l1_loss += l1_loss.item()

    avg_loss, ce_loss, l1_loss = total_loss/len(input), total_ce_loss/len(input), total_l1_loss/len(input)
    train_loss.append(avg_loss)
    train_ce_loss.append(ce_loss)
    train_l1_loss.append(l1_loss)

    # Log the loss to wandb
    print(f'Epoch {epoch+1}/{config.num_epochs}, Avg Training Loss: {avg_loss}')

    model.eval()
    tgt, pred = infer(model, config.dataloader)
    metrics = train_eval_metrics(tgt, pred)
    wandb.log({"epoch": epoch, "loss": avg_loss, "ce_loss": ce_loss, "l1_loss": l1_loss} | metrics)

  return train_loss, train_ce_loss, train_l1_loss


if __name__ == "__main__":
  from ..DFA.ParityDFA import ParityDFA
  from .dataloader import DFADataloader
  from .config import TrainConfig
  from .bilinear_function import BilinearFunction
  from .inference_bilinear import metrics_by_length
  seq_len = 10

  dataloader = DFADataloader(ParityDFA, seq_len)
  trainconf = TrainConfig(dataloader=dataloader)
  model = BilinearFunction(len(dataloader.alphabet), len(dataloader.states))
  optimizer = torch.optim.Adam(model.parameters(), lr=trainconf.lr)
    
  # Training loop
  train_loss, train_ce_loss, train_l1_loss = train_bilinear(model, optimizer, trainconf)

  model.eval()
  tgt, pred = infer(model, dataloader)
  visualizations = visualize_final_metrics(tgt, pred)

  plots = metrics_by_length(model, ParityDFA, [i for i in range(4,20)])
  print(plots)
