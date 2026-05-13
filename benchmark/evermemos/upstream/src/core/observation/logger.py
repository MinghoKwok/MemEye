import logging
import os
import sys
import traceback
from enum import Enum
from functools import lru_cache
from typing import Optional


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggerProvider:
    _instance: Optional["LoggerProvider"] = None
    _initialized = False

    def __new__(cls) -> "LoggerProvider":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self._setup_logging()
            self._initialized = True

    def _setup_logging(self) -> None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True,
        )
        for name in ("urllib3", "google", "googleapiclient", "httpx", "hpack", "httpcore", "pymongo", "aiokafka"):
            logging.getLogger(name).setLevel(logging.WARNING)

    @lru_cache(maxsize=512)
    def _get_cached_logger(self, module_name: str) -> logging.Logger:
        return logging.getLogger(module_name)

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        if name is None:
            frame = sys._getframe(1)
            name = frame.f_globals.get("__name__", "unknown")
        return self._get_cached_logger(name)

    def _get_caller_logger(self) -> logging.Logger:
        frame = sys._getframe(2)
        module_name = frame.f_globals.get("__name__", "unknown")
        return self._get_cached_logger(module_name)

    def debug(self, message: str, *args, **kwargs):
        self._get_caller_logger().debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        self._get_caller_logger().info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        self._get_caller_logger().warning(message, *args, **kwargs)

    def warn(self, message: str, *args, **kwargs):
        self.warning(message, *args, **kwargs)

    def error(self, message: str, exc_info: bool = True, *args, **kwargs):
        self._get_caller_logger().error(message, exc_info=exc_info, *args, **kwargs)

    def exception(self, message: str, exc_info: bool = True, *args, **kwargs):
        self._get_caller_logger().exception(message, exc_info=exc_info, *args, **kwargs)

    def critical(self, message: str, exc_info: bool = True, *args, **kwargs):
        self._get_caller_logger().critical(message, exc_info=exc_info, *args, **kwargs)

    def log_with_stack(self, level: LogLevel, message: str):
        logger = self._get_caller_logger()
        stack_trace = traceback.format_stack()
        full_message = f"{message}\nStack trace:\n{''.join(stack_trace)}"
        getattr(logger, level.value.lower())(full_message)


logger_provider = LoggerProvider()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logger_provider.get_logger(name)


def debug(message: str, *args, **kwargs):
    logger_provider.debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs):
    logger_provider.info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs):
    logger_provider.warning(message, *args, **kwargs)


def warn(message: str, *args, **kwargs):
    logger_provider.warn(message, *args, **kwargs)


def error(message: str, exc_info: bool = True, *args, **kwargs):
    logger_provider.error(message, exc_info=exc_info, *args, **kwargs)


def exception(message: str, exc_info: bool = True, *args, **kwargs):
    logger_provider.exception(message, exc_info=exc_info, *args, **kwargs)


def critical(message: str, exc_info: bool = True, *args, **kwargs):
    logger_provider.critical(message, exc_info=exc_info, *args, **kwargs)


def log_with_stack(level: LogLevel, message: str):
    logger_provider.log_with_stack(level, message)
