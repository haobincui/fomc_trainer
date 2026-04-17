import logging
import os
from datetime import datetime


os.makedirs("logs", exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
        logging.StreamHandler()
]
)



logging.info("✅ Global logging configuration loaded.")
