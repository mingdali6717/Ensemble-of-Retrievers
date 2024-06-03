import argparse
import warnings
import debugpy
import torch

from eor.config import Config
from eor.main import run_eor

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='a project on Ensemble of Retrievers (EoR)')
    parser.add_argument('-c', '--config', type=str, default='config/base.yaml', help='config file(yaml) path')
    parser.add_argument('-d', '--debug', action='store_true',help='use valid dataset to debug your system')

    args, _ = parser.parse_known_args()

    if args.debug:
        debugpy.listen(("0.0.0.0", 14327))
        debugpy.wait_for_client()
    
    # set up configs based on yaml file
    config = Config(args.config, debug=args.debug)
    
    if config['ddp']:
        
        world_size = len(config["device"])
        torch.multiprocessing.spawn(run_eor, args=(config, args.debug), nprocs=world_size, join=True)
        
    else:
        
        run_eor(None, config)
            
