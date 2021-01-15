import logging


def log_init(name):
    logger = logging.getLogger(name)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[ %(levelname)8s %(filename)18s:%(lineno)4s - %(funcName)20s() ] %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.propagate = 0
    logger.setLevel(logging.DEBUG)

    return logger
