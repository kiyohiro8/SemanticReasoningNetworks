import argparse
import yaml

from trainer import SRNTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--params", help="path to parameter file")

    args = parser.parse_args()
    param_file = args.params

    with open(param_file, "r") as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    
    trainer = SRNTrainer()
    trainer.train(params)