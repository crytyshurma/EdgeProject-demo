import logging
import logging.handlers
from config import LOG_FILE
def setup_logging():
    logger = logging.getLogger("Surveillance")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    fh = logging.handlers.RotatingFileHandler(LOG_FILE)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger