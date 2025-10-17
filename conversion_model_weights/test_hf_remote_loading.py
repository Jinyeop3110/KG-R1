#!/usr/bin/env python3
"""
Test script to verify that models on HuggingFace Hub load correctly.

Tests loading from: JinyeopSong/KG-R1_test
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_remote_model(repo_id, test_name):
    """Test loading and basic generation from HuggingFace Hub."""
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"Repository: {repo_id}")
    print(f"{'='*70}\n")

    try:
        # Test 1: Load tokenizer
        print("1️⃣ Loading tokenizer from HuggingFace Hub...")
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            trust_remote_code=True
        )
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {len(tokenizer)}")
        print(f"  Pad token: {tokenizer.pad_token}")
        print(f"  EOS token: {tokenizer.eos_token}")

        # Test 2: Load model
        print("\n2️⃣ Loading model from HuggingFace Hub...")
        print("  (This may take a few minutes to download...)")
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
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
        test_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
        ]

        for i, test_prompt in enumerate(test_prompts, 1):
            print(f"\n  Test {i}/{len(test_prompts)}")
            print(f"  Prompt: '{test_prompt}'")

            inputs = tokenizer(test_prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

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
    """Run tests on HuggingFace Hub models."""

    # Model repository to test
    repo_id = "JinyeopSong/KG-R1_test"
    test_name = "KG-R1 Test Model"

    # Allow testing custom repo via command line
    if len(sys.argv) > 1:
        repo_id = sys.argv[1]
        test_name = f"Custom model: {repo_id}"

    print("\n" + "="*70)
    print("HuggingFace Hub Remote Model Loading Test")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

    # Run test
    success = test_remote_model(repo_id, test_name)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"{status}: {test_name}")

    if success:
        print("\n🎉 Remote model loaded and tested successfully!")
        print(f"The model at {repo_id} is working correctly.")
    else:
        print("\n⚠️  Test failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
