import wandb
from .inference_GPT import infer
from ..utils.metrics import train_eval_metrics

def train_GPT(model, config, device_type, log=True):

    optimizer = model.configure_optimizers(config.weight_decay, config.lr, (config.beta1, config.beta2), device_type) 

    train_loss = []

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        for _ in range(config.num_batches):
            # Generate a batch of training examples
            src, tgt = config.dataloader.generate_batch(config.batch_size)

            # if _ == 0:
            #   print("src:", src)
            #   print("tgt:", tgt)

            # Forward pass
            optimizer.zero_grad()
            logits, loss = model(src, targets=tgt)

            # if _ == 0 and epoch==5:
            #   print("logits", logits)
            #   print("tgt output:", tgt)
            #   print("loss:", loss)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (config.num_batches)
        train_loss.append(avg_loss)
        
        # Print epoch and loss
        print(f'Epoch {epoch+1}/{config.num_epochs}, Avg Training Loss: {avg_loss}')

        # Log evaluation metrics and loss to wandb
        model.eval()
        tgt, pred = infer(model, config.dataloader, config.batch_size)
        metrics = train_eval_metrics(tgt, pred)
        if log:
            wandb.log({"epoch": epoch, "loss": avg_loss} | metrics)
        else:
            print(metrics) # while running this file locally, print metrics on every epoch
        
    return train_loss


if __name__ == "__main__":
    from ..DFA.ParityDFA import ParityDFA
    from .dataloader import DFADataloader
    from .config import TrainConfig, GPTConfig
    from .GPT import GPT
    import torch
    from .inference_GPT import metrics_by_length
    from .attention import visualize_all_attentions

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = 10

    # Initialize dataloader
    dataloader = DFADataloader(ParityDFA, max_seq_len=max_seq_len, pad_idx=0)

    # Organize args
    trainconf = TrainConfig(
        num_epochs=10,
        dataloader=dataloader,
        batch_size=10,
        num_batches=100
    ) # betas left as default

    gptconf = GPTConfig(
        vocab_size = dataloader.vocab_size,
        block_size = 6*max_seq_len,
        trigram=True
    )

    # Initialize
    model = GPT(gptconf).to(device)

    # Training loop
    train_loss = train_GPT(model, trainconf, device_type=device, log=False)
    # print(metrics_by_length(model, ParityDFA, [i for i in range(4, 10)], 10))
    # visualize_all_attentions(model, dataloader)
    # print(visualize_all_attentions(model, dataloader))