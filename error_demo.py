import logging
import time
from contextlib import contextmanager

# ── Setup logging — do this once at the top of any real script ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── Part 1: Specific exception handling ──────────────────────────
def load_config(filepath: str) -> dict:
    """Load a config file — demonstrates specific exception handling."""
    logger.info(f"Loading config from: {filepath}")

    try:
        with open(filepath) as f:
            content = f.read()
            logger.info("Config loaded successfully")
            return {"content": content}

    except FileNotFoundError:
        logger.error(f"Config file not found: {filepath}")
        raise FileNotFoundError(f"Cannot start — missing config: {filepath}")

    except PermissionError:
        logger.error(f"No permission to read: {filepath}")
        raise PermissionError(f"Cannot read file: {filepath}")


# ── Part 2: Your own context manager ─────────────────────────────
@contextmanager
def training_session(name: str):
    """Context manager for a training run — logs start/end, handles errors."""
    logger.info(f"Starting training session: {name}")
    start = time.time()
    try:
        yield   # your training code runs here
        elapsed = time.time() - start
        logger.info(f"Session complete: {name} in {elapsed:.3f}s")
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Session failed: {name} after {elapsed:.3f}s — {e}")
        raise   # re-raise so the caller knows it failed


# ── Part 3: Try it all ───────────────────────────────────────────
print("=== Test 1: File that exists ===")
try:
    load_config("generators_demo.py")   # this file exists
except FileNotFoundError as e:
    print(f"Caught: {e}")

print()
print("=== Test 2: File that doesn't exist ===")
try:
    load_config("missing_file.json")    # this doesn't exist
except FileNotFoundError as e:
    print(f"Caught: {e}")

print()
print("=== Test 3: Context manager — successful run ===")
with training_session("baseline-v1"):
    time.sleep(0.2)   # simulate training

print()
print("=== Test 4: Context manager — failed run ===")
try:
    with training_session("broken-run"):
        time.sleep(0.1)
        raise ValueError("Loss exploded — NaN detected")
except ValueError as e:
    print(f"Caught outside context manager: {e}")