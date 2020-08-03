import os
import sys
import argparse
import torch
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker


def run_vot(tracker_name, tracker_param, run_id=None, debug=0, visdom_info=None):
    # initSeed = 1
    # torch.manual_seed(initSeed)
    # torch.cuda.manual_seed(initSeed)
    # torch.cuda.manual_seed_all(initSeed)
    # np.random.seed(initSeed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # os.environ['PYTHONHASHSEED'] = str(initSeed)

    tracker = Tracker(tracker_name, tracker_param, run_id)
    tracker.run_vot(debug, visdom_info)


def main():

    parser = argparse.ArgumentParser(description='Run VOT.')
    parser.add_argument('tracker_name', type=str)
    parser.add_argument('tracker_param', type=str)
    parser.add_argument('--run_id', type=int, default=None)
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom')

    args = parser.parse_args()

    visdom_info = {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}
    run_vot(args.tracker_name, args.tracker_param, args.run_id, args.debug, visdom_info)


if __name__ == '__main__':
    main()
