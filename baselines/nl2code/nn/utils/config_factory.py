import logging


class MetaConfig(type):
    def __getitem__(self, key):
        return config._config[key]

    def __setitem__(self, key, value):
        config._config[key] = value


class config(object):
    _config = {}
    __metaclass__ = MetaConfig

    @staticmethod
    def set(key, val):
        config._config[key] = val

    @staticmethod
    def init_config(file='config.py'):
        if len(config._config) > 0:
            return

        logging.info('use configuration: %s', file)
        data = {}
        execfile(file, data)
        config._config = data['config']