'''
Written by Sidi Liang
'''

import logging
import sys
from datetime import datetime

now = datetime.now() 
date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_{0}.log".format(date_time), mode="w"),
    ],
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)

log = logging.getLogger()


