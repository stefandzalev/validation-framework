import logging

# Create a logger
ml_logger = logging.getLogger("data-lake-machine-learning-logger")
ml_logger.setLevel(logging.INFO)
# Create a handler and set its level
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add the formatter to the handler
handler.setFormatter(formatter)

# Add the handler to the logger
ml_logger.addHandler(handler)
