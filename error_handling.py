import json
import os


# ── PART 1: try / except / else / finally ────────────────────────

def read_file(filepath: str) -> str:
    """Read a text file and return its contents."""
    try:
        with open(filepath) as f:
            contents = f.read()


    except FileNotFoundError:
        print(f"ERROR: File not found - {filepath}")
        return ""
    
    except PermissionError:
        print(f"ERROR: No permission to read - {filepath}")
        return ""
    
    else:
        print(f"OK: File loaded - {len(contents)} characters")
        return contents
    
    finally:
        print(f"DONE: Attempted to read {filepath}")


# ── PART 2: raising your own exceptions ──────────────────────────

def create_experiment(name: str, learning_rate: float, epochs: int) -> dict:
    """Create an experiment config - validates all inputs."""

    if not name or not name.strip():
        raise ValueError("Experiment name cannot be empty")
    
    if learning_rate <= 0 or learning_rate > 1:
        raise ValueError(
            f"Learning rate must be between 0 and 1, got: {learning_rate}"
        )
    
    if epochs < 1:
        raise ValueError(f"Epochs must be at least 1, got: {epochs}")

    return {
        "name": name.strip(),
        "learning_rate": learning_rate,
        "epochs": epochs
    }


# ── PART 3: saving and loading data with json ────────────────────

def save_experiment(config: dict, filepath: str) -> None:
    """Save experiment config to a JSON file."""
    try:
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved: {filepath}")

    except OSError as e:
        print(f"ERROR: Could not save file - {e}")


def load_experiment(filepath: str) -> dict:
    """Load experiment config form JSON file"""
    try:
        with open(filepath) as f:
            return json.load(f)
        
    except FileNotFoundError:
        print(f"ERROR: No saved experiment at {filepath}")
        return {}
    
    except json.JSONDecodeError:
        print(f"ERROR: File exists but is not valid JSON - {filepath}")
        return {}
    
# ── PART 4: test everything ───────────────────────────────────────

print("=" * 50)
print("TEST 1: Reading files")
print("=" * 50)

result = read_file("cars.py")
print()
result = read_file("missing.txt")
print()

print("=" * 50)
print("TEST 2: Creating experiment")
print("=" * 50)

# Valid experiment
try:
    exp = create_experiment("baseline", 0.01, 10)
    print(f"Created: {exp}")
except ValueError as e:
    print(f"Caught: {e}")

print()

# Invalid learning rate
try:
    exp = create_experiment("bad-lr", 5.0, 10)
    print(f"Created: {exp}")
except ValueError as e:
    print(f"Caught: {e}")

print()

# Empty name
try:
    exp = create_experiment("", 0.01, 10)
    print(f"Created: {exp}")
except ValueError as e:
    print(f"Caught: {e}")

print()

print("=" * 50)
print("TEST 3: Saving and loading JSON")
print("=" * 50)

config = create_experiment("my-run", 0.01, 20)
save_experiment(config, "experiment.json")

loaded = load_experiment("experiment.json")
print(f"Loaded back: {loaded}")

print()
loaded_missing = load_experiment("nonexistent.json")
print(f"Missing file returned: {loaded_missing}")