import warnings
from base import Config, train
import argparse
import os


def warn(*args, **kwargs):
    pass


warnings.warn = warn

if __name__ == '__main__':

    rep = int(os.environ['SLURM_ARRAY_TASK_ID'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--tss', default=100, type=int,
                        help='training sample size')
    parser.add_argument('--ssd', default='/SampleSplits/',
                        help='sample splits directory (default: /SampleSplits/)')
    parser.add_argument('--nw', default=8, type=int,
                        help='number of workers')
    parser.add_argument('--bs', default=16, type=int,
                        help='batch size#')
    parser.add_argument('--sn', default='age', type=str,
                        help='scorename')
    parser.add_argument('--lr', default=0.01, type=float)
    args = parser.parse_args()

    cfg = Config(sample_size=args.tss, repetition_num=rep, sample_splits_dir=args.ssd,
                 num_workers=args.nw, batch_size=args.bs, scorename=args.sn, learning_rate=args.lr)

    train(cfg)
