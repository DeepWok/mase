#! /usr/bin/env python3
# This script configures the logger in mase
import sys, os, time, logging, colorlog, functools


def getLogger(name: str,
              logFile: str = "",
              console: bool = True) -> logging.Logger:
    # add a trace level
    logging.TRACE = logging.DEBUG - 5
    logging.addLevelName(logging.TRACE, 'TRACE')
    logging.Logger.trace = functools.partialmethod(logging.Logger.log,
                                                   logging.TRACE)
    logging.trace = functools.partial(logging.log, logging.TRACE)

    logger = logging.getLogger(name)
    logger.setLevel(logging.TRACE)

    if logFile:
        if os.path.isfile(logFile):
            os.remove(logFile)
        # File handle
        class customFileFormat(logging.Formatter):
            format = "[%(asctime)s][%(levelname)s] %(message)s"

            def format(self, record):
                logformat = "%(message)s" if record.levelno == logging.TRACE else "[%(asctime)s][%(levelname)s] %(message)s"
                formatter = logging.Formatter(logformat, "%Y-%m-%d %H:%M:%S")
                return formatter.format(record)

        fh = logging.FileHandler(logFile)
        fh.setFormatter(customFileFormat())
        fh.setLevel(logging.TRACE)
        logger.addHandler(fh)

    # Console handler
    if console:
        ch = logging.StreamHandler()

        class customConsoleFormat(logging.Formatter):
            format = "[%(asctime)s][%(levelname)s] %(message)s"

            def format(self, record):
                traceformat = logging.Formatter("%(message)s",
                                                "%Y-%m-%d %H:%M:%S")
                colorformat = colorlog.ColoredFormatter(
                    "%(log_color)s[%(asctime)s][%(levelname)s]%(reset)s" +
                    " %(message_log_color)s%(message)s",
                    "%Y-%m-%d %H:%M:%S",
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "red,bg_white",
                    },
                    secondary_log_colors={
                        "message": {
                            "ERROR": "red",
                            "CRITICAL": "red"
                        }
                    },
                )
                logformat = traceformat if record.levelno == logging.TRACE else colorformat
                return logformat.format(record)

        ch.setFormatter(customConsoleFormat())
        ch.setLevel(logging.TRACE)
        logger.addHandler(ch)
    return logger
