# coding:utf-8

import datetime
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def colored_print(color, message):
    """
    Formats a message with a specified color using ANSI escape codes.

    Parameters:
    - color (str): The color name (e.g., 'red', 'green', 'yellow', 'blue').
    - message (str): The message to format.

    Returns:
    - str: The formatted message with the specified color.
    """
    color_codes = {
        "black": "\x1b[30;1m",
        "red": "\x1b[31;1m",
        "green": "\x1b[32;1m",
        "yellow": "\x1b[33;1m",
        "blue": "\x1b[34;1m",
        "magenta": "\x1b[35;1m",
        "cyan": "\x1b[36;1m",
        "white": "\x1b[37;1m",
    }

    reset_code = "\x1b[0m"
    color_code = color_codes.get(color.lower(), color_codes["black"])
    print(f"{color_code}{message}{reset_code}")

    
def init_logging(filename_prefix):
    path = Path(__file__).resolve()
    path_root = Path(path).parent.parent.parent
    log_dir = os.path.join(path_root, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger()
    logger.handlers=[]
    # level = logging.INFO
    level = logging.DEBUG
    filename = f'{log_dir}/{filename_prefix}_{datetime.datetime.now().strftime("%Y-%m-%d")}.log'
    #logformat = '%(asctime)s %(levelname)s %(module)s.%(funcName)s Line:%(lineno)d %(message)s'
    logformat = '%(asctime)s %(levelname)s: %(message)s (%(filename)s:%(lineno)d)'
    timeformat = "%m-%d %H:%M:%S"
    logFormatter = logging.Formatter(logformat, timeformat)
    
    hdlr = TimedRotatingFileHandler(filename, "midnight", 1, 14)
    hdlr.setFormatter(logFormatter)
    logger.addHandler(hdlr)
    logger.setLevel(level)
    
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(CustomFormatter())
    
    # consoleHandler.setLevel(logging.INFO)
    consoleHandler.setLevel(level)
    logger.addHandler(consoleHandler)
    logger.setLevel(level)


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s %(levelname)s: %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)    