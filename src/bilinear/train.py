import argparse
import os
import wandb
import torch
from .config import TrainConfig
from .dataloader import DFADataloader
from ..DFA.ParityDFA import ParityDFA
from ..DFA.BinaryAdditionDFA import BinaryAdditionDFA
from .train_bilinear import train_bilinear
from .bilinear_function import BilinearFunction
from .inference_bilinear import infer, metrics_by_length
from datetime import datetime
from ..utils.visualizations import visualize_final_metrics

# Function to ensure the outputs directory is created
def create_output_dir():
    # Get the current working directory
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, "bilinear_outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Function to generate a unique run name
def generate_run_name(args):
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"run_{current_time}_bilinear_{args.DFA}_lr{args.lr}_s{args.seq_len}"
    return run_name

# Function to choose a DFA
def choose_DFA(args):
    if args.DFA == 'ParityDFA':
        return ParityDFA
    elif args.DFA == 'BinaryAdditionDFA':
        return BinaryAdditionDFA

def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--DFA", type=str, default='ParityDFA', help="Name of DFA to be trained")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length to train on")
    parser.add_argument("--l1_penalty", type=float, default=0.01, help="L1 penalty")
    parser.add_argument("--name", type=str, default='', help="Optional model name.")
    
    return parser.parse_args()

def main(args):
    # Initialize Weights and Biases
    wandb.init(project= 'bilinear_' + args.name + '_' + args.DFA, config=args)
    config = wandb.config
        
    # Create outputs directory
    output_dir = create_output_dir()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize dataloader
    DFA = choose_DFA(args)
    dataloader = DFADataloader(DFA, args.seq_len)

    # Organize args
    trainconf = TrainConfig(
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        lr = args.lr,
        seq_len = args.seq_len,
        l1_penalty= args.l1_penalty
    )

    # Initialize
    model = BilinearFunction(len(dataloader.alphabet), len(dataloader.states)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=trainconf.lr)
    
    # Training loop
    train_loss, train_ce_loss, train_l1_loss = train_bilinear(model, optimizer, trainconf)
    
    # Save the model
    run_name = generate_run_name(args)
    model_path = os.path.join(output_dir, f"{run_name}_model.pth")
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)

    # Final Inference for Evaluation
    model.eval()
    tgt_pred, pred = infer(model, dataloader)
    visualizations = visualize_final_metrics(tgt_pred, pred)
    wandb.log(visualizations)

    by_length = metrics_by_length(model, DFA, [i for i in range(4, args.seq_len*10)])
    wandb.log(by_length)



if __name__ == "__main__":
    args = get_args()
    main(args)