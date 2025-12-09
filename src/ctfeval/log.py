import logging


logger = logging.getLogger("ctfeval")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s\t%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
