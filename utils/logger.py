from configs.configdict import ConfigDict
import logging
# logging.basicConfig()

DEFAULT_LEVEL = 'warning'

logger = logging.getLogger(__name__)

fmt = '%(asctime)s [%(levelname)s] %(message)s'
fmt = '[%(levelname).1s] %(message)s'
formatter = logging.Formatter(fmt)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

LOG_LEVELS = dict(debug=10, info=20, warning=30, error=40)


def set_loggers(**levels):

    if 'loggers' in levels:
        levels = levels['loggers']

    default = levels.get('__default__', DEFAULT_LEVEL)
    logging_level = LOG_LEVELS[levels.get(__name__) or default]
    logger.setLevel(logging_level)
    logger.addHandler(stream_handler)
    for name, l in logging.Logger.manager.loggerDict.items():
        level = levels.get(name) or default
        logger.log(20, 'setting debugger {} level to {}'.format(name, level))

        if isinstance(l, logging.Logger):
            l.setLevel(LOG_LEVELS[level])
            l.addHandler(stream_handler)


set_loggers(**ConfigDict())
