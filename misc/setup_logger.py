import logging, os


def setup_logger(config: dict):

    logger = logging.getLogger(config["System"]["logging"]["logger_name"])
    logger.setLevel(config["System"]["logging"]["level"])


    # Add a consol logger
    if config["System"]["logging_console"]:
        console = logging.StreamHandler()
        level_dict = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR,
                      "critical": logging.CRITICAL}
        console.setLevel(level_dict.get(config["System"]["logging"]["level"].lower(), logging.INFO))
        formatter = logging.Formatter(config["System"]["logging"]["format"])
        console.setFormatter(formatter)
        logger.addHandler(console)
        config["System"]["logging_console"] = False

    # Write logs to file
    file_handler = logging.FileHandler(config["System"]["logging"]["filename"])
    file_handler.setLevel(config["System"]["logging"]["level"])
    formatter = logging.Formatter(config["System"]["logging"]["format"])
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return config
