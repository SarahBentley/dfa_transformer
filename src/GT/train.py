import wandb
from .inference import infer
from ..utils.metrics import train_eval_metrics
from .embeddings import visualize_wpe, visualize_wte
from ..DFA.ParityDFA import ParityDFA
from ..DFA.BinaryAdditionDFA import BinaryAdditionDFA


# Function to choose a DFA
def choose_DFA(args):
    if args.DFA == 'ParityDFA':
        return ParityDFA
    elif args.DFA == 'BinaryAdditionDFA':
        return BinaryAdditionDFA
    
def train(model, config, device_type, early_stopping_threshold= 1e-6, log=True):
    optimizer = model.configure_optimizers(config.weight_decay, config.lr, (config.beta1, config.beta2), device_type)
    if not early_stopping_threshold:
        early_stopping_threshold = -float('inf')
    
    train_loss = []
    best_loss = float('inf')  # Initialize best loss to infinity
    best_model_state = None  # To store the best model state
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        for _ in range(config.num_batches):
            # Generate a batch of training examples
            src, tgt = config.dataloader.generate_batch(config.batch_size)
            
            # Forward pass
            optimizer.zero_grad()
            logits, loss = model(src, targets=tgt)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / config.num_batches
        train_loss.append(avg_loss)
        
        # Check if the current model is the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()  # Save the best model state
        
        # Print epoch and loss
        print(f'Epoch {epoch+1}/{config.num_epochs}, Avg Training Loss: {avg_loss}')
        
        # Log evaluation metrics and loss to wandb
        model.eval()
        tgt, pred = infer(model, config.dataloader, 50)
        metrics = train_eval_metrics(tgt, pred)
        if log:
            wandb.log({"epoch": epoch, "loss": avg_loss} | metrics)
        else:
            print(metrics)  # while running this file locally, print metrics on every epoch

        # Early stopping check
        if best_loss < early_stopping_threshold:
            print(f"Early stopping at epoch {epoch+1} with loss {best_loss}")
            break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Using best model loaded with loss:", best_loss)
    
    return train_loss



if __name__ == "__main__":
    from ..DFA.ParityDFA import ParityDFA
    from .dataloader import DFADataloader
    from .config import TrainConfig, GTConfig, WGTConfig, RWGTConfig
    from .GT import GT
    from .WGT import WGT
    from .RWGT import RWGT
    import torch
    from .inference import metrics_by_length
    from .attention import visualize_all_attentions

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = 10
    state_freq = 1

    # Initialize dataloader
    dataloader = DFADataloader(ParityDFA, max_seq_len=max_seq_len, pad_idx=0, state_freq=state_freq)

    # Organize args
    trainconf = TrainConfig(
        num_epochs=10,
        dataloader=dataloader,
        batch_size=10,
        num_batches=100,
        lr=0.001
    ) # betas left as default

    gtconf = GTConfig(
        vocab_size = dataloader.vocab_size,
        block_size = 6*max_seq_len,
        n_head=1,
        n_layer=1,
        n_embd=10
    )

    wgtconf = WGTConfig(
        vocab_size = dataloader.vocab_size,
        window_size=state_freq+1,
        n_head=1,
        n_layer=1,
        n_embd=10
    )

    rwgtconf = RWGTConfig(
        vocab_size = dataloader.vocab_size,
        window_size=state_freq+1,
        n_head=3,
        n_layer=3,
        n_embd=60,
    )

    # Initialize
    # model = GT(gtconf).to(device)
    model = WGT(wgtconf).to(device)
    # model = RWGT(rwgtconf).to(device)

    # Training loop
    train_loss = train(model, trainconf, device_type=device, log=False)
    by_length = metrics_by_length(model, ParityDFA, [i for i in range(4, max_seq_len*5)], 50)
    for title, fig in by_length.items():
        if title == "Performance metrics by length":
            fig.show()
    wte = visualize_wte(model, dataloader)
    for title, fig in wte.items():
        fig.show()
    # visualize_wpe(model)
    attentions = visualize_all_attentions(model, dataloader)
    for title, fig in attentions.items():
        fig.show()