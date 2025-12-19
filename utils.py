import json
import os
from pathlib import Path

def load_api_key(keys_file:str, provider: str) -> str:
    """Load API key for a specific provider from ~/.llm_keys file.

    The file format is:
    <provider name>:<api_key>
    """

    if not keys_file.exists():
        raise FileNotFoundError(f"API keys file not found at {keys_file}")

    with open(keys_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue

            if ':' in line:
                provider_name, api_key = line.split(':', 1)
                if provider_name.strip().lower() == provider.lower():
                    return api_key.strip()

    raise ValueError(f"Provider '{provider}' not found in {keys_file}")

def read_file(filepath: str, encoding: str = 'utf-8') -> str:
    with open(os.path.expanduser(filepath.strip()), 'r', encoding=encoding) as f:
        return f.read()