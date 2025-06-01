import json
import logging

def set_logger(log_path):
    """
    Configures the Python logger to output logs to both the terminal and a specified log file.

    This is useful for saving all printed logs during training or evaluation to a permanent file,
    while still displaying them in the terminal.

    Example:
        set_logger("logs/train.log")
        logging.info("Training started...")

    Args:
        log_path (str): The path to the log file where logs will be saved.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:'
                                                    ' %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """
    Saves a dictionary of numeric values to a JSON file.

    Args:
        d (dict): A dictionary with float-convertible values (e.g., float, int, numpy.float).
        json_path (str): The path to the JSON file where the dictionary will be saved.
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept
        # np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

