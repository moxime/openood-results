from .configdict import ConfigDict
import logging

DEFAULT_LEVEL = 'warning'

logger = logging.getLogger(__name__)

fmt = '%(asctime)s [%(levelname)s] %(message)s'
fmt = '[%(levelname).1s] %(message)s'
formatter = logging.Formatter(fmt)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)


def log_level(s):

    return getattr(logging, s.upper())


def get_level(name, **levels):
    if not name:
        return
    if name in levels:
        return levels[name]
    names = name.split('.')
    if names[0] not in levels:
        return
    return get_level('.'.join(names[1:]), **levels[names[0]])


def set_loggers(**levels):

    if 'loggers' in levels:
        levels = levels['loggers']

    default_level = levels.get('__default__', DEFAULT_LEVEL)
    logger.setLevel(log_level(get_level(__name__, **levels) or default_level))
    logger.addHandler(stream_handler)

    for name, l in logging.Logger.manager.loggerDict.items():
        if isinstance(l, logging.Logger):
            level = get_level(name, **levels) or default_level
            l.setLevel(log_level(level))
            l.addHandler(stream_handler)
            logger.debug('setting debugger {} level to {}'.format(name, level))


if __name__ == '__main__':
    print('******************')
    set_loggers(**ConfigDict())
