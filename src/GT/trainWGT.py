import argparse
import os
import wandb
import torch
from .config import TrainConfig, WGTConfig, RWGTConfig
from .dataloader import DFADataloader
from .WGT import WGT
from .RWGT import RWGT
from .train import train, choose_DFA
from .inference import infer, metrics_by_length
from datetime import datetime
from ..utils.visualizations import visualize_final_metrics
from .attention import visualize_all_attentions
from .embeddings import visualize_wte

# Function to ensure the outputs directory is created
def create_output_dir(args):
    # Get the current working directory
    if args.rel:
        name = "RWGT_outputs"
    else:
        name = "WGT_outputs"
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Function to generate a unique run name
def generate_run_name(args):
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"run_{current_time}_WGT_{args.DFA}_{args.name}_lr{args.lr}_s{args.max_seq_len}"
    return run_name

def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--name", type=str, default='', help="Name of project")
    parser.add_argument("--DFA", type=str, default='ParityDFA', help="Name of DFA to be trained")
    parser.add_argument("--rel", type=bool, default=False, help="Whether the model should use relative positional encoding")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max_seq_len", type=int, default=10, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for training")
    parser.add_argument("--num_batches", type=int, default=50, help="Number of batches")
    parser.add_argument("--n_layer", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--n_head", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=40, help="Embedding dimension")
    parser.add_argument("--MLP_width", type=int, default=None, help="Number of parallel activations in MLP layer")
    parser.add_argument("--input_freq", type=int, default=1, help="How often states should be interspersed with input symbols. Ex.  1 means every other symbol, etc.")


    parser.add_argument("--gen", type=int, default=20, help="Max sequence length for which we want to test model generalization")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="Beta2 for Adam optimizer")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--bias", type=bool, default=False, help="Use bias in Linears and LayerNorms")
    
    return parser.parse_args()

def main(args):
        
    # Create outputs directory
    output_dir = create_output_dir(args)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize dataloader
    DFA = choose_DFA(args)
    dataloader = DFADataloader(DFA, max_seq_len=args.max_seq_len, pad_idx=0, input_freq=args.input_freq)

    # Organize args
    trainconf = TrainConfig(
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        lr = args.lr,
        num_batches = args.num_batches,
        weight_decay= args.weight_decay,
        batch_size = args.batch_size
    ) # betas left as default

    WGTconf = WGTConfig(
        vocab_size = dataloader.vocab_size,
        n_layer = args.n_layer,
        n_head = args.n_head,
        n_embd = args.n_embd,
        dropout = args.dropout,
        bias = args.bias,
        pad_idx = dataloader.pad_idx,
        MLP_width = args.MLP_width,
        window_size= args.input_freq + 1 # add 1 to see state
    )

    RWGTconf = RWGTConfig(
        vocab_size = dataloader.vocab_size,
        n_layer = args.n_layer,
        n_head = args.n_head,
        n_embd = args.n_embd,
        dropout = args.dropout,
        bias = args.bias,
        pad_idx = dataloader.pad_idx,
        MLP_width = args.MLP_width,
        window_size= args.input_freq + 1 # add 1 to see state
    )
    
    # Initialize Weights and Biases
    if args.rel == True:
        wandb.init(project='RWGT_' + args.DFA + '_' + args.name, config=args)
        # Initialize model
        model = RWGT(RWGTconf).to(device)
    elif args.rel == False:
        wandb.init(project='WGT_' + args.DFA + '_' + args.name, config=args)
        # Initialize model. Use same config for relative
        model = WGT(WGTconf).to(device)
    config = wandb.config
    
    # Training loop
    train_loss = train(model, trainconf, device_type=device)
    
    # Save the model
    run_name = generate_run_name(args)
    model_path = os.path.join(output_dir, f"{run_name}_model.pth")
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)

    # Final Inference for Evaluation
    model.eval()
    tgt_pred, pred = infer(model, dataloader, args.batch_size)
    visualizations = visualize_final_metrics(tgt_pred, pred)
    wandb.log(visualizations)

    by_length = metrics_by_length(model, DFA, [i for i in range(4,args.gen)], batch_size=args.batch_size, pad_idx=0)
    wandb.log(by_length)

    attentions = visualize_all_attentions(model, dataloader)
    wandb.log(attentions)

    wte = visualize_wte(model, dataloader)
    wandb.log(wte)



if __name__ == "__main__":
    args = get_args()
    main(args)