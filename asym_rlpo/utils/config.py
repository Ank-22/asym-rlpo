from typing import Any, TypeAlias

BasicType: TypeAlias = str | float | int | bool | None
ConfigDict: TypeAlias = dict[str, BasicType]


class Config:
    def __init__(self):
        self._config: ConfigDict = {}

    def _clear(self):
        self._config.clear()

    def _update(self, cd: ConfigDict):
        self._config.update(cd)

    def _get(self, name: str, default=None) -> Any:
        return self._config.get(name, default)

    def _as_dict(self) -> ConfigDict:
        return self._config.copy()

    def __getattr__(self, name: str) -> Any:
        return self._config[name]


_config: Config | None = None


def get_config() -> Config:
    global _config

    if _config is None:
        _config = Config()

    return _config
