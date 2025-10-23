"""
EAGLE Model Test Setup - Refactored for Modularity

This script provides a modular framework for testing EAGLE model inference
with different input lengths and configurations.

Usage:
- Run full test suite: python test_setup.py
- Customize parameters by modifying the CONFIG dictionaries
- Use utility functions for quick tests or custom configurations
"""

import traceback
import torch
import json
from eagle.model.ea_model import EaModel
from transformers import AutoTokenizer
import os
import gc

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Model configuration
MODEL_CONFIG = {
    "base_model_path": "meta-llama/Llama-3.1-8B-Instruct",
    "ea_model_path": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "torch_dtype": torch.float16,
    "low_cpu_mem_usage": True,
    "total_token": -1,
}

# Test configuration
TEST_CONFIG = {
    "context_file_path": "./illusions_perdues.txt",
    "context_max_length": 31000,
    "input_lengths": [1000 + i * 500 for i in range(18)],  # Generate test lengths
    "runs_per_length": 5,
    "context_offset_per_run": 1000,
    "seagle": True
}

# Inference configuration
INFERENCE_CONFIG = {
    "temperature": 0,
    "max_new_tokens": 512,
    "max_length": 31000,
    "log": True,
    "is_llama3": True,
}

# System configuration
DEVICE = os.environ.get("EAGLE_DEVICE", "cuda")  # Default to cuda, but can be overridden

# ============================================================================
# INITIALIZATION AND LOGGING
# ============================================================================

def initialize_system():
    """Initialize system and print configuration info"""
    if DEVICE == "cpu":
        print("Running on CPU for debugging")
    else:
        print("Running on GPU")

    # Check if CUDA_LAUNCH_BLOCKING == 1
    if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
        print("CUDA_LAUNCH_BLOCKING is set to 1")
    else:
        print("CUDA_LAUNCH_BLOCKING is not set to 1")

initialize_system()

def get_prefixed_versionned_filename(filename):
    version = 0
    while os.path.exists(f"{version}_{filename}"):
        version += 1
    return f"{version}_{filename}"

def load_context_file(file_path, tokenizer, max_length=2048):
    text = ""
    with open(file_path, "r") as f:
        text = f.read()
    
    # Only take the first max_length tokens
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = token_ids[:max_length]
    text = tokenizer.decode(token_ids)
    return text

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# ============================================================================
# MODEL LOADING AND INITIALIZATION
# ============================================================================

def load_tokenizer(model_path):
    """Load tokenizer for the specified model"""
    print(f"Loading tokenizer from: {model_path}")
    return AutoTokenizer.from_pretrained(model_path)

def load_model(model_config, device):
    """Load and initialize the EAGLE model"""
    print("=== Loading Model ===")

    model = EaModel.from_pretrained(
        base_model_path=model_config["base_model_path"],
        ea_model_path=model_config["ea_model_path"],
        torch_dtype=model_config["torch_dtype"],
        low_cpu_mem_usage=model_config["low_cpu_mem_usage"],
        device_map="cpu" if device == "cpu" else "auto",
        total_token=model_config["total_token"],
    )
    model.eval()

    # Move model to CPU if needed for debugging
    if device == "cpu":
        model = model.cpu()
        print("Model moved to CPU for debugging")

    print("After model loading:")
    print_gpu_memory()

    return model

# ============================================================================
# PROMPT GENERATION
# ============================================================================

def generate_prompts(tokenizer, test_config):
    """Generate prompts for different input lengths and runs"""
    print("=== Generating Prompts ===")

    # Load context file
    context = load_context_file(
        test_config["context_file_path"], 
        tokenizer, 
        max_length=test_config["context_max_length"]
    )

    input_lengths = test_config["input_lengths"]
    runs_per_length = test_config["runs_per_length"]
    context_offset = test_config["context_offset_per_run"]

    tokenized_prompts = []
    for length in input_lengths:
        length_prompts = []
        for run in range(runs_per_length):
            # Create different context for each run by taking different slices
            context_slice = context[run * context_offset:]

            tokenized_context = tokenizer.encode(context_slice, add_special_tokens=False)
            tokenized_context = tokenized_context[:length]
            context_slice = tokenizer.decode(tokenized_context)

            prompt = tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": f"""
Please answer the following question based on the provided context.
Question: Summarize this text
Context: {context_slice}
                    """.strip()
                }],
                tokenize=False,
                add_generation_prompt=True
            )
            length_prompts.append(prompt)
        tokenized_prompts.append(length_prompts)

    print(f"Generated {len(tokenized_prompts)} length groups with {runs_per_length} runs each")
    return tokenized_prompts

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def run_single_inference(model, prompt, inference_config, device):
    """Run inference on a single prompt and return results"""
    input_ids = model.tokenizer([prompt]).input_ids
    actual_input_length = len(input_ids[0])

    print(f"Starting test case with input length: {actual_input_length}")

    input_ids = torch.as_tensor(input_ids).to(device)

    try:
        output_ids, accept_lengths = model.eagenerate(
            input_ids, 
            temperature=inference_config["temperature"], 
            max_new_tokens=inference_config["max_new_tokens"], 
            max_length=inference_config["max_length"],
            log=inference_config["log"], 
            is_llama3=inference_config["is_llama3"]
        )

        output = model.tokenizer.decode(output_ids[0])
        print("Output length:", len(output_ids[0]) - actual_input_length)

        # Format accept lengths with input length information
        #accept_lengths is a list of (input_length, accept_length) for each sequence in the batch
        formatted_results = {
            "input_length": actual_input_length,
            "dec_pos-acc_len": [((input_length-actual_input_length), accept_length) for input_length, accept_length in accept_lengths]
        }

        return {
            "success": True,
            "output_ids": output_ids,
            "output": output,
            "formatted_results": formatted_results,
            "actual_input_length": actual_input_length
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM error with input length {actual_input_length}: {str(e)}")
            print(traceback.format_exc())
            print("Skipping this test case...")
            clear_gpu_memory()
            return {
                "success": False,
                "error_type": "OOM",
                "error_message": str(e),
                "actual_input_length": actual_input_length
            }
        else:
            print(f"Other runtime error: {str(e)}")
            print(traceback.format_exc())
            return {
                "success": False,
                "error_type": "RuntimeError",
                "error_message": str(e),
                "actual_input_length": actual_input_length
            }
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return {
            "success": False,
            "error_type": "UnexpectedError",
            "error_message": str(e),
            "actual_input_length": actual_input_length
        }

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_test_suite(model_config, test_config, inference_config, device):
    """Run the complete test suite"""
    print("=== Starting Test Suite ===")

    import eagle.model.cnets as cnets
    cnets.ENABLE_SEAGLE = test_config["seagle"]

    # Initialize components
    tokenizer = load_tokenizer(model_config["base_model_path"])
    model = load_model(model_config, device)
    tokenized_prompts = generate_prompts(tokenizer, test_config)

    # Track results
    all_results = []
    successful_tests = 0
    failed_tests = 0
    oom_errors = 0

    # Run tests
    print("=== Running Inference Tests ===")
    for length_idx, length_prompts in enumerate(tokenized_prompts):
        for run_idx, prompt in enumerate(length_prompts):
            print(f"\n--- Test {length_idx + 1}/{len(tokenized_prompts)}, Run {run_idx + 1}/{len(length_prompts)} ---")

            result = run_single_inference(model, prompt, inference_config, device)

            if result["success"]:
                res = result["formatted_results"]
                all_results.append(res)
                successful_tests += 1
                print(f"Average accept length: {sum([acc for pos, acc in res['dec_pos-acc_len']]) / len(res['dec_pos-acc_len'])}")
            else:
                failed_tests += 1
                if result["error_type"] == "OOM":
                    oom_errors += 1
                    # Break out of runs for this length if we hit OOM
                    print("Breaking out of current length group due to OOM")
                    break

    # Print summary
    print("\n=== Test Summary ===")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {failed_tests}")
    print(f"OOM errors: {oom_errors}")
    print(f"Total accept length records: {len(all_results)}")

    # Save results
    output_filename = get_prefixed_versionned_filename(f"results_{'seagle' if test_config['seagle'] else 'eagle3'}.json")
    with open(output_filename, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {output_filename}")

    return {
        "results": all_results,
        "successful_tests": successful_tests,
        "failed_tests": failed_tests,
        "oom_errors": oom_errors
    }

# ============================================================================
# UTILITY FUNCTIONS FOR CUSTOMIZATION
# ============================================================================

def create_custom_test_config(
    context_file_path=None,
    context_max_length=None,
    input_lengths=None,
    runs_per_length=None,
    context_offset_per_run=None,
    seagle=True
):
    """Create a custom test configuration by overriding default values"""
    config = TEST_CONFIG.copy()

    if context_file_path is not None:
        config["context_file_path"] = context_file_path
    if context_max_length is not None:
        config["context_max_length"] = context_max_length
    if input_lengths is not None:
        config["input_lengths"] = input_lengths
    if runs_per_length is not None:
        config["runs_per_length"] = runs_per_length
    if context_offset_per_run is not None:
        config["context_offset_per_run"] = context_offset_per_run
    if seagle is not None:
        config["seagle"] = seagle
    return config

def create_custom_inference_config(
    temperature=None,
    max_new_tokens=None,
    max_length=None,
    log=None,
    is_llama3=None
):
    """Create a custom inference configuration by overriding default values"""
    config = INFERENCE_CONFIG.copy()

    if temperature is not None:
        config["temperature"] = temperature
    if max_new_tokens is not None:
        config["max_new_tokens"] = max_new_tokens
    if max_length is not None:
        config["max_length"] = max_length
    if log is not None:
        config["log"] = log
    if is_llama3 is not None:
        config["is_llama3"] = is_llama3

    return config

def run_quick_test(input_lengths=None, runs_per_length=None, seagle=None):
    """Run a quick test with fewer iterations for debugging"""
    if input_lengths is None:
        input_lengths = [1500, 3000, 6000]  # Just 3 lengths for quick testing
    if runs_per_length is None:
        runs_per_length = 2  # Just 2 runs per length
    if seagle is None:
        seagle = True
    quick_test_config = create_custom_test_config(
        input_lengths=input_lengths,
        runs_per_length=runs_per_length,
        seagle=seagle
    )

    print("=== Running Quick Test ===")
    return run_test_suite(MODEL_CONFIG, quick_test_config, INFERENCE_CONFIG, DEVICE)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the full test suite
    # To run a quick test instead, uncomment the line below:
    # results = run_quick_test(seagle=False)
    results = run_test_suite(MODEL_CONFIG, TEST_CONFIG, INFERENCE_CONFIG, DEVICE)