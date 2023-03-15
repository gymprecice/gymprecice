import os
import random
import numpy as np


def fix_randseeds(seed=1234):
    try:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = True
    except Exception as e:
        print(e)
        pass
