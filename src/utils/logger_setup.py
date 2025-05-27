# PROJECT_ROOT/utils/logger_setup.py
import logging
import sys

def setup_logger(logger_name: str = 'NucliaEval', level: int = logging.WARNING) -> logging.Logger:
    """
    Sets up a standardized logger.

    Args:
        logger_name (str): The name for the logger.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger is already configured
    if not logger.handlers:
        # Create a handler (console handler in this case)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
        )
        ch.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(ch)

    return logger

# Example of a global logger instance for the application
# You can import this specific logger in other modules.
app_logger = setup_logger()