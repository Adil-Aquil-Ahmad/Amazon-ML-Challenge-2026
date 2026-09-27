import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline
if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default=os.path.join(here, '..', '..', 'student_resource'), help='folder containing dataset/train and dataset/test')
    ap.add_argument('--out-dir', default=os.path.join(here, '..', '..', 'student_resource', 'output'))
    ap.add_argument('--model-dir', default=os.path.join(here, '..', 'models'))
    ap.add_argument('--threads', type=int, default=os.cpu_count())
    ap.add_argument('--top-k', type=int, default=pipeline.DEFAULTS['top_k'])
    ap.add_argument('--key-cap', type=int, default=pipeline.DEFAULTS['key_cap'])
    ap.add_argument('--train-frac', type=float, default=pipeline.DEFAULTS['train_frac'])
    ap.add_argument('--stage', choices=['all', 'train', 'test'], default='all', help='train: fit + save models only; test: reuse saved models and write outputs')
    a = ap.parse_args()
    pipeline.run_all(a.base_dir, a.out_dir, a.model_dir, stage=a.stage, n_threads=a.threads, top_k=a.top_k, key_cap=a.key_cap, train_frac=a.train_frac)
