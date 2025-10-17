#!/usr/bin/env python3
"""
Test script to verify that merged checkpoints load correctly with HuggingFace transformers.

Tests:
1. Model loading from merged checkpoint
2. Tokenizer loading
3. Basic inference/generation
4. Model parameter counts
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_checkpoint(checkpoint_path, test_name):
    """Test loading and basic generation for a checkpoint."""
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"Path: {checkpoint_path}")
    print(f"{'='*70}\n")

    try:
        # Test 1: Load tokenizer
        print("1️⃣ Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {len(tokenizer)}")
        print(f"  Pad token: {tokenizer.pad_token}")
        print(f"  EOS token: {tokenizer.eos_token}")

        # Test 2: Load model
        print("\n2️⃣ Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✓ Model loaded successfully")

        # Test 3: Count parameters
        print("\n3️⃣ Model information...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model dtype: {model.dtype}")
        print(f"  Device: {next(model.parameters()).device}")

        # Test 4: Simple generation
        print("\n4️⃣ Testing generation...")
        test_prompt = "What is the capital of France?"

        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print(f"  Prompt: '{test_prompt}'")
        print(f"  Generating response...")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Response: '{response}'")

        print(f"\n✅ All tests passed for {test_name}")
        return True

    except Exception as e:
        print(f"\n❌ Test failed for {test_name}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if 'model' in locals():
            del model
        if 'inputs' in locals():
            del inputs
        if 'outputs' in locals():
            del outputs
        torch.cuda.empty_cache()


def main():
    """Run tests on both merged checkpoints."""

    # Define checkpoints to test
    checkpoints = [
        (
            "/home/yeopjin/orcd/pool/workspace/RL_KG/verl_checkpoints/webqsp-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn7_rep1-merged-step150",
            "WebQSP (step 150)"
        ),
        (
            "/home/yeopjin/orcd/pool/workspace/RL_KG/verl_checkpoints/cwq-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn5-merged-step400",
            "CWQ (step 400)"
        ),
    ]

    # Allow testing specific checkpoint via command line
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        checkpoint_name = "Custom checkpoint"
        checkpoints = [(checkpoint_path, checkpoint_name)]

    print("\n" + "="*70)
    print("HuggingFace Model Loading Test")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

    # Run tests
    results = {}
    for checkpoint_path, name in checkpoints:
        success = test_checkpoint(checkpoint_path, name)
        results[name] = success

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All checkpoints loaded and tested successfully!")
        print("The merged checkpoints are ready for HuggingFace upload.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
