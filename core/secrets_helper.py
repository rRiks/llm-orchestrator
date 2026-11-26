"""
Universal secret retrieval helper.
Compatible with:
1. Kaggle Notebooks (via kaggle_secrets UserSecretsClient)
2. Local environments & CI/CD (via os.environ)
"""

import os

def get_secret(name: str, default: str = None) -> str:
    """Retrieve a secret from Kaggle UserSecretsClient if available, otherwise os.environ."""
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        return user_secrets.get_secret(name)
    except Exception:
        return os.environ.get(name, default)
