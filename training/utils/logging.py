# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import sys
import atexit

import functools
from .general import safe_makedirs


# cache the opened file object, so that different calls
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    log_buffer_kb = 1 * 1024  # 1KB
    io = open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io



def setup_logging(name, output_dir=None, log_level_primary="INFO"):
    """
    Setup the logging streams: stdout, plus `{output_dir}/log.txt` when output_dir is set.
    """
    # get the filename if we want to log to the file as well
    log_filename = None
    if output_dir:
        safe_makedirs(output_dir)
        log_filename = f"{output_dir}/log.txt"

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    # create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # clean up any pre-existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []
    logging.root.handlers = []

    # setup the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level_primary)
    logger.addHandler(console_handler)

    # we log to file as well if user wants
    if log_filename is not None:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger
