import os
import sys

# Configuration
FORBIDDEN_STRINGS = [
    "qwen2.5:3b", "qwen2.5:1.5b", "qwen2.5:14b",
    "llama3.2:3b", "llama3.1:70b",
    "phi3:mini", "mistral:7b", "mixtral:8x7b"
]

# Files explicitly allowed to contain these strings
ALLOWED_FILES = [
    "backend/.model",
    "backend/models_catalog.yaml",
    "verify_changes.py",
    "verify_phase2.py",
    "verify_phase2_with_mocks.py",
    "scripts/ci_gate.py",
    "README.md",
    "task.md",
    "implementation_plan.md",
    "docker-compose.yml",
    "DEPLOYMENT_CHECKLIST.md",
    "TEST_RESULTS.md",
    "backend_test_results.json",
    ".gitignore",
    ".gitconfig"
]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def scan_file(filepath):
    """Scan a single file for forbidden strings."""
    rel_path = os.path.relpath(filepath, ROOT_DIR).replace("\\", "/")
    
    if rel_path in ALLOWED_FILES:
        return []

    found_errors = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                for s in FORBIDDEN_STRINGS:
                    if s in line:
                        found_errors.append(f"{rel_path}:{i} contains forbidden '{s}'")
    except Exception as e:
        print(f"Warning: Could not read {rel_path}: {e}")
        
    return found_errors

def main():
    print(f"Starting CI Gate Scan from {ROOT_DIR}...")
    errors = []
    
    for root, dirs, files in os.walk(ROOT_DIR):
        # Skip hidden dirs and venv
        if ".git" in root or ".venv" in root or "__pycache__" in root or "node_modules" in root:
            continue
            
        for file in files:
            # Skip non-text files roughly
            if file.endswith(('.py', '.js', '.md', '.txt', '.yml', '.yaml', '.json', '.html', '.css', '.env')):
                full_path = os.path.join(root, file)
                file_errors = scan_file(full_path)
                errors.extend(file_errors)

    if errors:
        print("\n[FAIL] Found hardcoded model references usage in unauthorized files:")
        for e in errors:
            print(f" - {e}")
        print("\nPlease use 'backend/.model' or 'models_catalog.yaml' instead.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] No unauthorized hardcoded model references found.")
        sys.exit(0)

if __name__ == "__main__":
    main()
