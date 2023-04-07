import json
from os import path
# from gymprecice.utils.fileutils import make_result_dir

with open(path.join(__path__[0], 'gymprecice-config.json')) as config_file:
  content = config_file.read()

environment_config = json.loads(content)
environment_config['environment']['src'] = __path__[0]

# make_result_dir(environment_config)

