
import math
import multiprocessing as mp

LOCK = mp.Lock()
EPSILON = 1e-6
LOG_EPSILON = math.log(EPSILON)
SLEEP_TIME = 0.5