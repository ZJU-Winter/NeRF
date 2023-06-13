import logging

logger = logging.getLogger(__package__)
logger.setLevel(logging.INFO)

def setup_logger(path: str = ""):
    # create logger
    # logger.setLevel(logging.DEBUG)

    # log to file
    if path != "":
        logging.basicConfig(filename=path, encoding='utf-8', level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger