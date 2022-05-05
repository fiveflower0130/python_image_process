import os
import logging
import datetime


class LoggerSingleton():
    _instance = None

    def __init__(self):
        if LoggerSingleton._instance is not None:
            raise Exception('instance is exist')
        else:
            LoggerSingleton._instance = Logger()

    @staticmethod
    def get_instance():
        if LoggerSingleton._instance is None:
            LoggerSingleton()
        return LoggerSingleton._instance


class Logger():
    _logger = None

    def __init__(self):
        self._logger = logging.getLogger("logger")
        self._logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s \t [%(levelname)s | %(filename)s | %(funcName)s:%(lineno)s ] -> %(message)s'
        )

        # %Y-%m-%d_%H_%M_%S
        logname = datetime.datetime.now().strftime("%Y-%m-%d.log")
        dirname = "./log"

        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        fileHandler = logging.FileHandler(dirname + "/log_" + logname,
                                          encoding="utf-8",
                                          mode="a")

        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        self._logger.addHandler(fileHandler)
        self._logger.addHandler(streamHandler)

        # print("Init logger instance")

    def get_logger(self):
        return self._logger
