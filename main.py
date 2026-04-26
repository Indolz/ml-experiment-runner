import argparse
import json
from ml_runner import MLExperiment, RegularisedExperiment

def parse_args():
    parser = argparse.ArgumentParser(
        description="ML Experiment Runner - run and track training experiments"
    )
    parser.add_argument("--name", type=str, required=True,
                        help="Name of the experiment")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to run (default: 10)")
    parser.add_argument("--regularise", action="store_true",
                        help="Use regularised experiment")
    parser.add_argument("--lambda-reg", type=float, default=0.001,
                        help="Regularisation penalty (default: 0.001)")
    return parser.parse_args()

def run_experiment(args):
    if args.regularise:
        exp = RegularisedExperiment(
            name=args.name,
            learning_rate=args.lr,
            lambda_reg=args.lambda_reg
        )
    else:
        exp = MLExperiment(name=args.name, learning_rate=args.lr)

    print(f"\n{'='*50}")
    print(f" Running: {exp}")
    print(f"{'='*50}")

    results = exp.run(epochs=args.epochs)

    print(f"\n Epochs: {len(exp)}")
    print(f" Best loss: {exp.best_result()}")
    print(f" Final loss: {results[-1]}")
    print(f"\n Loss curve: {results}")
    print(f"{'='*50}\n")

    return exp

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)