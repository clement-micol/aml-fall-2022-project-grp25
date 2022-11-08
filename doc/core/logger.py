import logging

def custom_logger():
    logger = logging.getLogger()
    logger.setLevel('INFO')
    blue = '\u001b[30m'
    yellow = "\x1b[33;20m"
    reset = "\x1b[0m"
    fmt = blue+'%(asctime)s'+reset+'--'+yellow+'%(levelname)-5s'+reset+': %(message)s'
    fmt_date = '%H:%M:%S'
    formatter = logging.Formatter(fmt, fmt_date)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

if __name__ == "__main__":
    logger = custom_logger()
    logger.info("Hello Word")