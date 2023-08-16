from os import path
import tomli

__config__ = path.abspath(path.join(path.split(__file__)[0], '../pyproject.toml'))
if path.exists(__config__):
    with open(__config__, mode='rb') as f:
        __metadata__ = tomli.load(f)['tool']['poetry']
    version = __metadata__['version']