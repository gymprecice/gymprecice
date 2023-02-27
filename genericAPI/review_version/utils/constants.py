
from os import environ
import math

EPSILON = 1e-6
LOG_EPSILON = math.log(EPSILON)
BASE_PATH = environ.get("PWD", "")