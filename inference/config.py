"""
Centralized inference configuration.

Model and config can be selected via:
1. Environment variables: MARVIN_MODEL, MARVIN_CONFIG
2. Editing the defaults below

Available models (in inference/ folder):
- marvin_token_bf16.pt      (token-conditioned, ~23M params)

Available configs:
- "auto"  : Auto-detect from checkpoint (recommended)
- "small" : Small token-conditioned (~23M params)
- "large" : Large token-conditioned (~100M params)
"""

import os
from pathlib import Path

# =============================================================================
# DEFAULT CONFIGURATION - Edit these to change the default model
# =============================================================================

# Model checkpoint filename (relative to inference/ folder)
DEFAULT_MODEL = "marvin_small.pt"

# Model config: "auto", "small", "large"
# "auto" will detect from checkpoint keys (recommended)
DEFAULT_CONFIG = "auto"

# =============================================================================
# Environment variable overrides
# =============================================================================

def get_model_name() -> str:
    """Get model filename from env var or default."""
    return os.environ.get("MARVIN_MODEL", DEFAULT_MODEL)


def get_config_name() -> str:
    """Get config name from env var or default."""
    return os.environ.get("MARVIN_CONFIG", DEFAULT_CONFIG)


def get_model_path(repo_root: Path) -> Path:
    """Get full path to model checkpoint."""
    return repo_root / "inference" / get_model_name()


# =============================================================================
# Helper to print current config
# =============================================================================

def print_config(repo_root: Path) -> None:
    """Print current model configuration."""
    model = get_model_name()
    config = get_config_name()
    path = get_model_path(repo_root)
    exists = "✓" if path.exists() else "✗ NOT FOUND"
    print(f"Model: {model} ({exists})")
    print(f"Config: {config}")
