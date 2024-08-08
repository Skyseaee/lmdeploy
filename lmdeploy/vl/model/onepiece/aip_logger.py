# Copyright (c) Shopee. All rights reserved.
from os import environ


def env(key, type_, default=None):
    if key not in environ:
        return default

    val = environ[key]

    if type_ == str:
        return val
    elif type_ == bool:
        if val.lower() in ["1", "true", "yes", "y", "ok", "on"]:
            return True
        if val.lower() in ["0", "false", "no", "n", "nok", "off"]:
            return False
        raise ValueError(
            "Invalid environment variable '%s' (expected a boolean): '%s'" % (key, val)
        )
    elif type_ == int:
        try:
            return int(val)
        except ValueError:
            raise ValueError(
                "Invalid environment variable '%s' (expected an integer): '%s'"
                % (key, val)
            ) from None

from lmdeploy.utils import get_logger
import logging
logger = get_logger("lmdeploy", log_level=logging.INFO)

# AIP_LOG_LEVEL = env("AIP_LOG_LEVEL", str, "DEBUG")
# logger.remove()
# logger.add(sys.stderr, level=AIP_LOG_LEVEL)
TRT_LOG_LEVEL = env("TRT_LOG_LEVEL", str, "INFO")
