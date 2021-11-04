import configparser
import os
import sys
from datetime import datetime

import numpy as np

os.system("mkdir -p generated_configs/")

#usage: python launch_config.py rgb config.ini

# dst domain
domain_id = sys.argv[1]
# base template
cfg_template = sys.argv[2]

cfg_out = "generated_configs/launch_ensembles_%s_%s_%s.ini" % (
    domain_id, str(datetime.now()), cfg_template[:-4])
config = configparser.ConfigParser()
config.read(cfg_template)

config.set("GraphStructure", "only_edges_to_dst", domain_id)
tensorboard_prefix = config.get("Logs", "tensorboard_prefix")

# run with different similarity functions
sim_fcts = ['lpips', 'l1']
for sim_fct in sim_fcts:

    config.set('Ensemble', 'similarity_fct', sim_fct)
    with open(cfg_out, "w") as fd:
        config.write(fd)

    os.system('python main.py "%s"' % (cfg_out))
