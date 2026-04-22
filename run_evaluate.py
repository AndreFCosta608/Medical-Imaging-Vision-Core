import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from scripts.Segmentation import evaluate_model

if __name__ == '__main__':
    evaluate_model.run()
