"""
Quick benchmark to estimate training time on current hardware.
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def benchmark():
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU (will be slow)")

    print("\n1. Loading model (this takes a while first time)...")
    start = time.time()

    # Load smaller model for benchmark (Gemma-2B would take longer)
    # Using a smaller model to get a baseline, then we'll extrapolate
    model_name = "google/gemma-2b"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # MPS works better with float32
        )
        model.to(device)
        load_time = time.time() - start
        print(f"   Model loaded in {load_time:.1f}s")
    except Exception as e:
        print(f"   Could not load Gemma-2B: {e}")
        print("   Trying smaller model for benchmark...")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        load_time = time.time() - start
        print(f"   Loaded {model_name} in {load_time:.1f}s")
        print("   NOTE: GPT-2 is much smaller than Gemma-2B, multiply estimates by ~15x")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Training config from train.py
    batch_size = 2
    seq_length = 512
    n_warmup = 3
    n_benchmark = 10

    print(f"\n2. Benchmarking forward+backward pass...")
    print(f"   Batch size: {batch_size}, Seq length: {seq_length}")

    # Create dummy batch
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length)).to(device)
    labels = input_ids.clone()

    # Warmup
    model.train()
    for _ in range(n_warmup):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        model.zero_grad()
        if device.type == "mps":
            torch.mps.synchronize()

    # Benchmark
    times = []
    for i in range(n_benchmark):
        start = time.time()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        model.zero_grad()
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    tokens_per_step = batch_size * seq_length
    tokens_per_sec = tokens_per_step / avg_time

    print(f"   Avg step time: {avg_time:.2f}s")
    print(f"   Tokens/second: {tokens_per_sec:.0f}")

    # Estimate for WikiText-2
    print(f"\n3. Training time estimates for WikiText-2:")

    # WikiText-2 has ~2M tokens in train, ~36k samples after tokenization
    wikitext2_samples = 36000  # approximate
    gradient_accumulation = 8
    epochs = 1

    steps_per_epoch = wikitext2_samples // batch_size
    optimizer_steps = steps_per_epoch // gradient_accumulation

    total_time_seconds = steps_per_epoch * avg_time * epochs
    total_time_minutes = total_time_seconds / 60
    total_time_hours = total_time_minutes / 60

    print(f"   Dataset samples: ~{wikitext2_samples:,}")
    print(f"   Steps per epoch: {steps_per_epoch:,}")
    print(f"   Optimizer steps: {optimizer_steps:,}")
    print(f"")
    print(f"   Estimated time for 1 epoch:")
    if total_time_hours >= 1:
        print(f"     {total_time_hours:.1f} hours")
    else:
        print(f"     {total_time_minutes:.0f} minutes")

    # Memory usage
    print(f"\n4. Memory usage:")
    if device.type == "mps":
        # MPS doesn't have easy memory tracking, estimate based on model size
        param_count = sum(p.numel() for p in model.parameters())
        param_gb = param_count * 4 / 1e9  # float32
        print(f"   Model parameters: {param_count/1e9:.1f}B")
        print(f"   Estimated model memory: ~{param_gb:.1f}GB")
        print(f"   With activations + optimizer: ~{param_gb * 3:.1f}GB")
    elif device.type == "cuda":
        print(f"   GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB")
        print(f"   GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.1f}GB")

    print(f"\n5. Recommendations:")
    if total_time_hours > 4:
        print("   - Consider using cloud GPU for faster iteration")
        print("   - Or reduce max_seq_length to 256 for ~2x speedup")
        print("   - Or use a smaller dataset subset for initial testing")
    else:
        print("   - Training time is reasonable for local development")
        print("   - Consider overnight runs for full training")

    return {
        "device": str(device),
        "step_time_seconds": avg_time,
        "tokens_per_second": tokens_per_sec,
        "estimated_epoch_minutes": total_time_minutes,
    }


if __name__ == "__main__":
    benchmark()
