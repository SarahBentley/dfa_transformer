import plotly.graph_objects as go
import torch

def visualize_all_attentions(model, dataloader):
    model.eval()
    src, tgt = dataloader.generate_batch(batch_size=1) # generate only one example
    decoded_src = dataloader.decode(src[0])
    with torch.no_grad():
        # Forward pass through the model
        logits, loss = model(src, targets=tgt, use_flash=False)

    # Dictionary to store plots
    plots_dict = {}

    # Loop through each layer and each head
    for layer_idx, layer in enumerate(model.transformer.h):
        attention_weights = layer.attn.attention_weights # shape is (B, nh, T, T)

        
        # Loop through each head
        for head_idx in range(model.config.n_head):
            # Select the attention weights for the specified head
            attn = attention_weights[0, head_idx].cpu().numpy()

            # Plot the attention heatmap using Plotly
            fig = go.Figure(data=go.Heatmap(
                z=list(attn),
                colorscale='Viridis'
            ))

            fig.update_layout(
                title=f"Attention Weights (Layer {layer_idx}, Head {head_idx})",
                xaxis_title="Key Tokens",
                yaxis_title="Query Tokens",
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(decoded_src))),
                    ticktext=decoded_src,
                    tickangle=45  # Adjust angle for better readability
                ),
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(decoded_src))),
                    ticktext=decoded_src,
                    tickangle=45  # Adjust angle for better readability
                )
            )

            # Store the plot in the dictionary with an appropriate name
            plot_name = f"layer_{layer_idx}_head_{head_idx}"
            plots_dict[plot_name] = fig
    
    return plots_dict
