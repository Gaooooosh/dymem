import os
import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from dymem.transformer.qwen2_dymem import register_customized_qwen2
from dymem.utils import CacheWithMem


def save_heatmap(matrix, out_path, title=None, highlight_col=None):
    plt.figure(figsize=(10, 8), dpi=150)
    plt.imshow(matrix, aspect='auto', interpolation='nearest', cmap='viridis')
    cbar = plt.colorbar()
    cbar.set_label('Attention Score', rotation=270, labelpad=15)
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Key index', fontsize=12)
    plt.ylabel('Query index', fontsize=12)
    
    if highlight_col is not None and 0 <= highlight_col < matrix.shape[1]:
        # Use a very thin vertical line to indicate the sink position precisely
        plt.axvline(x=highlight_col, color='red', linewidth=0.5, alpha=0.5, linestyle='--')
        
        # Add a marker at the top x-axis to indicate the column without obscuring data
        plt.scatter([highlight_col], [-0.5], marker='v', color='red', s=50, clip_on=False, zorder=10)
        plt.text(highlight_col, -matrix.shape[0] * 0.02, 'Sink', ha='center', va='bottom', 
                 color='red', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_zoomed_heatmap(matrix, out_path, highlight_col, window_size=20, title=None):
    """
    Saves a zoomed-in heatmap around the highlight_col.
    window_size: Number of columns to show on left and right of highlight_col.
    """
    if highlight_col is None or highlight_col >= matrix.shape[1]:
        return

    start_col = max(0, highlight_col - window_size)
    end_col = min(matrix.shape[1], highlight_col + window_size + 1)
    
    zoomed_matrix = matrix[:, start_col:end_col]
    
    plt.figure(figsize=(10, 8), dpi=150)
    plt.imshow(zoomed_matrix, aspect='auto', interpolation='nearest', cmap='viridis',
               extent=[start_col, end_col, matrix.shape[0], 0]) # Set extent to keep original indices
    cbar = plt.colorbar()
    cbar.set_label('Attention Score', rotation=270, labelpad=15)
    
    if title:
        plt.title(f"{title} (Zoomed)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Key index', fontsize=12)
    plt.ylabel('Query index', fontsize=12)
    
    # Highlight the sink column
    plt.axvline(x=highlight_col, color='red', linewidth=1.5, linestyle='--')
    plt.scatter([highlight_col], [-0.5], marker='v', color='red', s=50, clip_on=False, zorder=10)
    plt.text(highlight_col, -matrix.shape[0] * 0.02, 'Sink', ha='center', va='bottom', 
                color='red', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_line(sink_vec, other_mean, other_max, out_path, title=None):
    plt.figure(figsize=(10, 5), dpi=150)
    
    # Plot Sink Attention
    plt.plot(sink_vec, label='Sink Token', color='#d62728', linewidth=2, alpha=0.9)
    
    # Plot Others
    if other_mean is not None:
        plt.plot(other_mean, label='Avg Other Tokens', color='#1f77b4', linewidth=1.5, linestyle='--', alpha=0.8)
    
    if other_max is not None:
        plt.plot(other_max, label='Max Other Tokens', color='#7f7f7f', linewidth=1, linestyle=':', alpha=0.6)

    if title:
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        
    plt.xlabel('Query index', fontsize=12)
    plt.ylabel('Attention Score (Log Scale)', fontsize=12)
    plt.yscale('log') # Use log scale to make small values visible
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.3, which='both') # Grid for both major and minor ticks in log scale
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def visualize_attentions(attentions, sink_len, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # attentions: tuple(len=num_layers) of [B,H,T_q,T_k]
    assert isinstance(attentions, (tuple, list))
    for layer_idx, layer_attn in enumerate(attentions):
        # B=1 assumed
        attn = layer_attn[0].detach().float().cpu()  # [H,T_q,T_k]
        heads, tq, tk = attn.shape
        layer_dir = os.path.join(out_dir, f"layer_{layer_idx:02d}")
        os.makedirs(layer_dir, exist_ok=True)

        # sum over heads heatmap
        sum_matrix = attn.sum(dim=0)  # [T_q,T_k]
        save_heatmap(
            sum_matrix.numpy(),
            os.path.join(layer_dir, 'heads_sum_heatmap.png'),
            title=f'Layer {layer_idx} Heads Sum',
            highlight_col=sink_len if sink_len < tk else None,
        )

        if sink_len < tk:
            # Zoomed heatmap for sum
            save_zoomed_heatmap(
                sum_matrix.numpy(),
                os.path.join(layer_dir, 'heads_sum_heatmap_zoomed.png'),
                highlight_col=sink_len,
                title=f'Layer {layer_idx} Heads Sum',
            )

            mem_vec_sum = sum_matrix[:, sink_len].numpy()
            
            # Calculate stats for other tokens
            mask = torch.ones(tk, dtype=torch.bool)
            mask[sink_len] = False
            others = sum_matrix[:, mask]
            other_mean = others.mean(dim=1).numpy() if others.shape[1] > 0 else None
            other_max = others.max(dim=1).values.numpy() if others.shape[1] > 0 else None

            save_line(
                mem_vec_sum,
                other_mean,
                other_max,
                os.path.join(layer_dir, 'heads_sum_mem_column.png'),
                title=f'Layer {layer_idx} Sum: attention to mem column',
            )

        # per-head heatmaps and mem column curves
        for h in range(heads):
            head_matrix = attn[h]
            save_heatmap(
                head_matrix.numpy(),
                os.path.join(layer_dir, f'head_{h:02d}_heatmap.png'),
                title=f'Layer {layer_idx} Head {h}',
                highlight_col=sink_len if sink_len < tk else None,
            )
            
            if sink_len < tk:
                # Zoomed heatmap for head
                save_zoomed_heatmap(
                    head_matrix.numpy(),
                    os.path.join(layer_dir, f'head_{h:02d}_heatmap_zoomed.png'),
                    highlight_col=sink_len,
                    title=f'Layer {layer_idx} Head {h}',
                )

                mem_vec = head_matrix[:, sink_len].numpy()
                
                # Calculate stats for other tokens per head
                mask = torch.ones(tk, dtype=torch.bool)
                mask[sink_len] = False
                others = head_matrix[:, mask]
                other_mean = others.mean(dim=1).numpy() if others.shape[1] > 0 else None
                other_max = others.max(dim=1).values.numpy() if others.shape[1] > 0 else None

                save_line(
                    mem_vec,
                    other_mean,
                    other_max,
                    os.path.join(layer_dir, f'head_{h:02d}_mem_column.png'),
                    title=f'Layer {layer_idx} Head {h}: attention to mem column',
                )


def main():
    parser = argparse.ArgumentParser(description='Visualize DyMem attentions (layer sum and per-head), focusing mem/sink token')
    parser.add_argument('--model_dir', type=str, required=True, help='HF model directory',default='qwen2.5-3b-compressor-instruct-sft')
    parser.add_argument('--exp_id', type=str, required=True, help='Experiment id for output path')
    parser.add_argument('--prompt_file', type=str, required=True, help='Prompt text file path')
    parser.add_argument('--max_input_tokens', type=int, default=4096, help='Truncate input to at most N tokens')
    parser.add_argument('--device', type=str, default=None, help='cuda:N or cpu')
    args = parser.parse_args()

    register_customized_qwen2(exist_ok=True)
    device = torch.device(args.device or (os.environ.get('CUDA_DEVICE', 'cuda:1') if torch.cuda.is_available() else 'cpu'))
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.set_device(device)

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, padding_side='right')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    with open(args.prompt_file, 'r') as f:
        prompt_text = f.read()

    messages = [
        {"role": "system", "content": "You are Qwen-Compressor, created by yg Xiao."},
        {"role": "user", "content": prompt_text},
    ]
    chat_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(chat_str, return_tensors='pt')
    # truncate
    if inputs['input_ids'].shape[-1] > args.max_input_tokens:
        inputs['input_ids'] = inputs['input_ids'][:, :args.max_input_tokens]
        if 'attention_mask' in inputs:
            inputs['attention_mask'] = inputs['attention_mask'][:, :args.max_input_tokens]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    past_key_values = CacheWithMem(model.config, dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=False,
        )
    attentions = outputs.attentions  # tuple[num_layers] of [B,H,T_q,T_k]

    model_name = os.path.basename(os.path.abspath(args.model_dir))
    base_out = os.path.join('/home/xiaoyonggao/dymem/viz', model_name, args.exp_id)
    os.makedirs(base_out, exist_ok=True)

    sink_len = getattr(model.config, 'num_attn_sinks', 128)
    visualize_attentions(attentions, sink_len, base_out)

    print(f'Visualization saved to: {base_out}')


if __name__ == '__main__':
    main()
