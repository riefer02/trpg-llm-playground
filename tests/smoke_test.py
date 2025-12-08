import os
import json
import yaml
import subprocess
import sys

def run_smoke_test():
    print("🚀 Starting Local Smoke Test Pipeline...")
    
    # Setup paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST_DIR = os.path.join(BASE_DIR, "tests", "artifacts")
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # 1. Mock Ingest Data
    print("\n[1/3] Creating mock ingested data...")
    # Make text > 200 chars to pass the filter in generate_synthetic.py
    long_text = "The Lancer RPG is a game about mechs and pilots. Combat is turn-based. " * 10
    raw_data = [
        {"page": 1, "text": long_text},
        {"page": 2, "text": long_text}
    ]
    raw_path = os.path.join(TEST_DIR, "mock_raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw_data, f)
    print(f"✓ Created {raw_path}")

    # 2. Mock Synthetic Config
    print("\n[2/3] Configuring synthetic generation test...")
    synth_output_path = os.path.join(TEST_DIR, "mock_synthetic.jsonl")
    synth_config = {
        "ingest": {
            "raw_output_path": raw_path
        },
        "topic": "Test Topic",
        "depth": "shallow",
        "n_samples": 2, # Small number for speed
        "styles": ["simple"],
        "output": {
            "path": synth_output_path
        },
        "llm": {
            "model": "gpt-mock"
        }
    }
    synth_config_path = os.path.join(TEST_DIR, "smoke_synth_config.yaml")
    with open(synth_config_path, "w") as f:
        yaml.dump(synth_config, f)
    
    # 3. Run Generate Synthetic
    print("\n[3/3] Running generation script (simulated LLM)...")
    # We rely on llm_client.py falling back to mock if no API key is present
    env = os.environ.copy()
    if "OPENAI_API_KEY" in env:
        print("  (Temporarily unsetting OPENAI_API_KEY to force mock mode)")
        del env["OPENAI_API_KEY"]
        
    # Run as module to handle relative imports
    cmd = [sys.executable, "-m", "src.data.generate_synthetic", "--config", synth_config_path]
    
    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, env=env, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Generation failed!")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            return
            
        print("Script Output:", result.stdout)
        
        # 4. Validate Output
        if not os.path.exists(synth_output_path):
             print(f"❌ Output file not found at {synth_output_path}!")
             return
             
        with open(synth_output_path, "r") as f:
            lines = f.readlines()
            
        if len(lines) == 0:
            print("❌ Output file is empty! (Mock LLM might have returned empty response)")
            return
            
        try:
            data = json.loads(lines[0])
            required_fields = ["instruction", "output"]
            for field in required_fields:
                if field not in data:
                    print(f"❌ Missing field in output: {field}")
                    return
        except json.JSONDecodeError:
            print("❌ Output is not valid JSONL!")
            return
            
        print(f"✓ Generated {len(lines)} valid samples.")
        
        # 5. Check Training Config (Static Analysis)
        print("\n[Optional] Verifying Training Config integrity...")
        train_config_path = os.path.join(BASE_DIR, "config", "rpg_finetune.yaml")
        try:
            with open(train_config_path, "r") as f:
                t_config = yaml.safe_load(f)
            
            # Check critical keys
            assert 'model' in t_config
            assert 'dataset' in t_config
            assert 'training' in t_config
            assert 'lora' in t_config
            print(f"✓ Training config {train_config_path} is valid YAML and has required sections.")
            
        except Exception as e:
            print(f"❌ Training config validation failed: {e}")
            return

        print("\n✅ SMOKE TEST PASSED! The pipeline logic is sound for local execution.")
        print("  (Note: Actual training requires GPU/Colab, which was skipped)")
        
    except Exception as e:
        print(f"❌ An error occurred during test execution: {e}")

if __name__ == "__main__":
    run_smoke_test()

