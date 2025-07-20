import os


class _ModelLibrarySettings:
    _key_overrides: dict[str, str]

    def __init__(self):
        self._key_overrides = {}

    def set(self, **keys: str):
        self._key_overrides.update(
            {provider.upper(): key for provider, key in keys.items()}
        )

    def reset(self):
        self._key_overrides = {}

    def __getattr__(self, name: str) -> str:
        # load key from override
        if name in self._key_overrides:
            return self._key_overrides[name]

        # load key from environment
        if name in os.environ:
            return os.environ[name]

        raise AttributeError(f"Missing config key: {name}")
