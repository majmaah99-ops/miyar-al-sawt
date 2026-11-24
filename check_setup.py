"""Check local readiness without downloading weights or exposing private data."""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path

MODULES={'numpy':'numpy','pandas':'pandas','scikit-learn':'sklearn',
         'scipy':'scipy','soundfile':'soundfile','streamlit':'streamlit',
         'pytest':'pytest'}


def check(model_dir):
    print('MIYAR V1 | Local readiness')
    for label,mod in MODULES.items():
        print(f"[{'OK' if importlib.util.find_spec(mod) else 'MISSING'}] Python {label}")
    root=Path(model_dir)
    required=['best_model.pth','xlsr2_300m.pt','config.yaml','run_inference.py']
    for name in required:
        print(f"[{'OK' if (root/name).is_file() else 'MISSING'}] Model {name}")
    print('NOTE: model file presence does not guarantee DeepFense compatibility or successful inference.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',default='models/ArA-DF-Baseline')
    args=p.parse_args()
    check(args.model_dir)
