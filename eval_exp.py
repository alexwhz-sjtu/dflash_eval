from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
import torch
import json
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from model.dflash_exp import DFlashDraftModel
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def register_local_dflash_model():
    """Register local DFlashDraftModel so AutoModel loads local implementation."""
    try:
        AutoModel.register(Qwen3Config, DFlashDraftModel, exist_ok=True)
    except TypeError:
        AutoModel.register(Qwen3Config, DFlashDraftModel)

def load_mtbench101_questions(question_file, begin=None, end=None):
    """Load questions from mtbench101.jsonl format"""
    questions = []
    with open(question_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Convert mtbench101 format to standard format
                turns = []
                for hist in data['history']:
                    turns.append(hist['user'])
                
                question = {
                    'question_id': data['id'],
                    'category': data.get('task', 'mtbench101'),
                    'turns': turns,
                    'history': data['history']  # Keep original history for reference
                }
                questions.append(question)
    
    if begin is not None and end is not None:
        questions = questions[begin:end]
    
    return questions


def multi_turn_dialogue(
    draft_model,
    target_model,
    tokenizer,
    turns,
    max_new_tokens=4096,
    temperature=0.0,
    log_file=None,
    thinking=False,
    use_spec_decode=False,
):
    """
    Execute multi-turn dialogue with the model
    
    Args:
        draft_model: The DFlash draft model instance
        target_model: The target model instance
        tokenizer: The tokenizer
        turns: List of user inputs (strings)
        max_new_tokens: Maximum new tokens to generate
        temperature: Temperature for generation
        log_file: File object to write logs to (optional)
    
    Returns:
        List of assistant responses and their statistics
    """
    conversation_history = []
    responses = []
    turn_stats = []
    response_lengths = []
    
    for turn_idx, user_input in enumerate(turns):
        if log_file:
            log_file.write(f"Turn {turn_idx + 1}: {user_input[:50]}...\n")
            log_file.flush()
        
        # Add user message to conversation history
        conversation_history.append({"role": "user", "content": "Answer the following question as detailed as possible: " + user_input})
        
        # Apply chat template with full conversation history
        text = tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking
        )
        
        # Tokenize and move to GPU
        input_ids = tokenizer([text], return_tensors="pt").input_ids.to(draft_model.device)
        
        # Generate response with selected decoding mode.
        if use_spec_decode:
            output_ids = draft_model.spec_generate(
                target=target_model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=temperature,
            )
        else:
            output_ids = draft_model.naive_generate(
                target=target_model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=temperature,
            )
        
        # Decode output (remove input tokens)
        output = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Add assistant response to conversation history for next turn
        conversation_history.append({"role": "assistant", "content": output})
        responses.append(output)
        
        # Calculate response length in tokens
        response_token_ids = tokenizer([output]).input_ids[0]
        response_length = len(response_token_ids)
        response_lengths.append(response_length)
        get_stats_fn = getattr(draft_model, "get_last_decode_stats", None)
        stats = get_stats_fn() if callable(get_stats_fn) else None
        stats = stats or {
            "accept_lengths": [],
            "target_total_time": 0.0,
            "draft_total_time": 0.0,
            "steps": 0,
        }
        
        # Add computed metrics to stats
        stats["response_length"] = response_length
        accept_lengths = stats.get("accept_lengths", [])
        stats["mean_accept_length"] = float(np.mean(accept_lengths)) if accept_lengths else 0.0
        
        # Calculate throughput (tokens per second)
        total_time = stats.get("target_total_time", 0.0) + stats.get("draft_total_time", 0.0)
        if total_time > 0 and response_length > 0:
            stats["throughput"] = response_length / total_time
        else:
            stats["throughput"] = 0.0
        
        turn_stats.append(stats)

        if log_file:
            log_file.write(f"Assistant: {output[:100]}...\n")
            log_file.write(f"  Response Length: {response_length} tokens\n")
            log_file.write(f"  Accept Lengths: {stats['accept_lengths']}\n")
            log_file.write(f"  Mean Accept Length: {stats['mean_accept_length']:.4f}\n")
            log_file.write(f"  Target Time: {stats['target_total_time']:.4f}s | Draft Time: {stats['draft_total_time']:.4f}s\n")
            log_file.write(f"  Throughput: {stats['throughput']:.2f} tokens/sec\n\n")
            log_file.flush()
    
    return responses, turn_stats


def summarize_question_stats(turn_stats, responses=None):
    all_accept_lengths = []
    response_lengths = []
    mean_accept_lengths = []
    throughputs = []
    target_total = 0.0
    draft_total = 0.0
    total_steps = 0
    total_response_length = 0

    for one_turn in turn_stats:
        all_accept_lengths.extend(one_turn.get("accept_lengths", []))
        response_lengths.append(one_turn.get("response_length", 0))
        mean_accept_lengths.append(one_turn.get("mean_accept_length", 0.0))
        throughputs.append(one_turn.get("throughput", 0.0))
        target_total += one_turn.get("target_total_time", 0.0)
        draft_total += one_turn.get("draft_total_time", 0.0)
        total_steps += one_turn.get("steps", 0)
        total_response_length += one_turn.get("response_length", 0)

    mean_accept = float(np.mean(all_accept_lengths)) if all_accept_lengths else 0.0
    mean_response_length = float(np.mean(response_lengths)) if response_lengths else 0.0

    mean_per_turn_accept = float(np.mean(mean_accept_lengths)) if mean_accept_lengths else 0.0
    mean_throughput = float(np.mean(throughputs)) if throughputs else 0.0
    total_time = target_total + draft_total
    overall_throughput = total_response_length / total_time if total_time > 0 else 0.0
    
    return {
        "all_accept_lengths": all_accept_lengths,
        "mean_accept_length": mean_accept,
        "response_lengths": response_lengths,
        "mean_response_length": mean_response_length,
        "total_response_length": total_response_length,
        "mean_per_turn_accept_length": mean_per_turn_accept,
        "mean_throughput": mean_throughput,
        "overall_throughput": overall_throughput,
        "target_total_time": target_total,
        "draft_total_time": draft_total,
        "total_time": total_time,
        "total_steps": total_steps,
    }


def plot_hidden_similarity_for_one_answer(turn_stats, output_path, log_file=None):
    """Plot mean/variance of cosine similarity by token distance and layer for one answer."""
    if not turn_stats:
        if log_file:
            log_file.write("No turn stats found, skip hidden similarity plot.\n")
        return False

    stat = turn_stats[0]
    groups = stat.get("layer_group_cosine_similarity", None)
    group_size = int(stat.get("hidden_similarity_group_size", 16))
    if not groups:
        if log_file:
            log_file.write("No hidden similarity data found, skip plot.\n")
        return False

    first_group = groups[0]
    cosine_by_layer = first_group.get("cosine_by_layer", [])
    num_layers = len(cosine_by_layer)
    if num_layers == 0:
        if log_file:
            log_file.write("Empty hidden similarity data, skip plot.\n")
        return False

    mean_mat = np.full((num_layers, group_size), np.nan, dtype=np.float32)
    var_mat = np.full((num_layers, group_size), np.nan, dtype=np.float32)

    for layer_idx in range(num_layers):
        by_dist = [[] for _ in range(group_size)]
        for g in groups:
            layer_values = g["cosine_by_layer"][layer_idx]
            for dist, value in enumerate(layer_values):
                if dist < group_size:
                    by_dist[dist].append(float(value))
        for dist in range(group_size):
            if by_dist[dist]:
                arr = np.array(by_dist[dist], dtype=np.float32)
                mean_mat[layer_idx, dist] = float(arr.mean())
                var_mat[layer_idx, dist] = float(arr.var())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    im0 = axes[0].imshow(mean_mat, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_title("Mean Cosine Similarity")
    axes[0].set_xlabel("Token Distance In Group")
    axes[0].set_ylabel("Layer Index")
    axes[0].set_xticks(np.arange(group_size))
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(var_mat, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Cosine Similarity Variance")
    axes[1].set_xlabel("Token Distance In Group")
    axes[1].set_ylabel("Layer Index")
    axes[1].set_xticks(np.arange(group_size))
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Hidden-State Similarity by Layer and Token Distance (First Answer)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    if log_file:
        log_file.write(f"Hidden similarity plot saved to: {output_path}\n")
        log_file.flush()
    return True


def main():
    parser = argparse.ArgumentParser()
    
    # Predefined model pairs
    MODEL_PAIRS = {
        "qwen3-8b": {
            "target": "/share/public/public_models/Qwen3-8B",
            "draft": "z-lab/Qwen3-8B-DFlash-b16"
        },
        "longwriter-llama3.1-8b": {
            "target": "THUDM/LongWriter-llama3.1-8b",
            "draft": "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
        }
    }
    
    parser.add_argument("--model-pair", type=str, default="qwen3-8b", 
                        choices=list(MODEL_PAIRS.keys()),
                        help="Choose a predefined model pair: qwen3-8b or longwriter-llama3.1-8b")
    parser.add_argument("--target-model-path", type=str, default=None,
                        help="Override target model path (if not using predefined pairs)")
    parser.add_argument("--draft-model-path", type=str, default=None,
                        help="Override draft model path (if not using predefined pairs)")
    parser.add_argument("--question-file", type=str, default="/share/wanghanzhen/SpeculativeDecoding/NIPS26/dataset/mtbench101/question.jsonl")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="./model_answer")
    parser.add_argument("--thinking", type=bool, default=False)
    parser.add_argument("-sd", "--spec-decode", action="store_true", help="Enable speculative decoding")
    args = parser.parse_args()
    
    # Use predefined model pair or custom paths
    if args.target_model_path is None:
        args.target_model_path = MODEL_PAIRS[args.model_pair]["target"]
    if args.draft_model_path is None:
        args.draft_model_path = MODEL_PAIRS[args.model_pair]["draft"]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract model and dataset names for output filename
    target_model_name = os.path.basename(args.target_model_path)
    draft_model_name = os.path.basename(args.draft_model_path)
    dataset_name = os.path.basename(os.path.dirname(args.question_file))
    
    # Create output filename
    output_file = os.path.join(
        args.output_dir,
        f"{target_model_name}-{'think' if args.thinking else ''}-DFlash-{dataset_name}-temperature-{args.temperature}-max_length_{args.max_length}.jsonl"
    )
    log_file_path = os.path.join(
        args.output_dir,
        f"{target_model_name}-{'think' if args.thinking else ''}-DFlash-{dataset_name}-temperature-{args.temperature}-max_length_{args.max_length}.log"
    )
    hidden_plot_path = os.path.join(
        args.output_dir,
        f"{target_model_name}-hidden_similarity_first_answer.png",
    )

    # Register local DFlash model implementation to AutoModel
    register_local_dflash_model()

    # Load models (force local DFlash implementation)
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_model_path,
        torch_dtype="auto",
        device_map="cuda:0"
    ).eval()
    
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        dtype="auto",
        device_map="cuda:0"
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    questions = load_mtbench101_questions(args.question_file, begin=args.begin, end=args.end)
    
    # Open output and log files
    with open(output_file, 'w', encoding='utf-8') as f_out, \
         open(log_file_path, 'w', encoding='utf-8') as f_log:
        
        f_log.write(f"Loaded {len(questions)} questions from {args.question_file}\n")
        f_log.write(f"Target Model: {args.target_model_path}\n")
        f_log.write(f"Draft Model: {args.draft_model_path}\n")
        f_log.write(f"Dataset: {dataset_name}\n")
        f_log.write(f"Temperature: {args.temperature}\n")
        f_log.write(f"Max Length: {args.max_length}\n\n")
        f_log.write(f"Decode Mode: {'spec_generate' if args.spec_decode else 'naive_generate'}\n\n")
        f_log.flush()

        hidden_plot_written = False

        for idx, question in enumerate(questions):
            f_log.write("\n" + "=" * 80 + "\n")
            f_log.write(f"Question {idx + 1}/{len(questions)} | ID: {question['question_id']} | Category: {question['category']}\n")
            f_log.write("=" * 80 + "\n")
            f_log.flush()

            responses, turn_stats = multi_turn_dialogue(
                draft_model,
                target_model,
                tokenizer,
                question['turns'],
                max_new_tokens=args.max_length,
                temperature=args.temperature,
                log_file=f_log,
                thinking=args.thinking,
                use_spec_decode=args.spec_decode,
            )

            summary = summarize_question_stats(turn_stats, responses)

            if not hidden_plot_written:
                hidden_plot_written = plot_hidden_similarity_for_one_answer(
                    turn_stats,
                    hidden_plot_path,
                    log_file=f_log,
                )

            f_log.write("-" * 80 + "\n")
            f_log.write("Question Summary\n")
            f_log.write(f"Response Lengths (per turn): {summary['response_lengths']}\n")
            f_log.write(f"Mean Response Length: {summary['mean_response_length']:.1f} tokens\n")
            f_log.write(f"Total Response Length: {summary['total_response_length']} tokens\n")
            f_log.write(f"\n")
            f_log.write(f"Accept Lengths (all verifications): {summary['all_accept_lengths']}\n")
            f_log.write(f"Overall Mean Accept Length: {summary['mean_accept_length']:.4f}\n")
            f_log.write(f"Mean Per-Turn Accept Length: {summary['mean_per_turn_accept_length']:.4f}\n")
            f_log.write(f"\n")
            f_log.write(f"Target Total Time: {summary['target_total_time']:.4f}s\n")
            f_log.write(f"Draft Total Time: {summary['draft_total_time']:.4f}s\n")
            f_log.write(f"Total Time: {summary['total_time']:.4f}s\n")
            f_log.write(f"\n")
            f_log.write(f"Mean Throughput (per turn): {summary['mean_throughput']:.2f} tokens/sec\n")
            f_log.write(f"Overall Throughput: {summary['overall_throughput']:.2f} tokens/sec\n")
            f_log.write(f"Total Verification Steps: {summary['total_steps']}\n")
            f_log.write("-" * 80 + "\n")
            f_log.flush()
            
            # Save result to JSONL file
            result = {
                "question_id": question['question_id'],
                "category": question['category'],
                "turns": question['turns'],
                "responses": responses,
                "statistics": {
                    "turn_stats": turn_stats,
                    "summary": summary
                }
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_out.flush()
        
        f_log.write(f"\n\nResults saved to: {output_file}\n")
        f_log.write(f"Logs saved to: {log_file_path}\n")


if __name__ == "__main__":
    main()