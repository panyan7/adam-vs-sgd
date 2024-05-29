import sys
from train_utils import *
from landscape_utils import *
from dataset_utils import *
from algorithm_list import algorithm_list_convergence_machine_translation as algorithm_list
from algorithm_list import algorithm_lists
from train_configs import *
import argparse

import huggingface_hub

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_key", type=str, help="Huggingface key required for access to the stack dataset")
    parser.add_argument("--num_epochs", default=20, type=int, help="Number of epochs")
    parser.add_argument("--task", default="dbg", type=str, help="Task to run on")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--algorithm_list_id", default=-1, type=int)
    parser.add_argument("--num_epoch_to_save", default=1000000, type=int)

    args = parser.parse_args()
    if not args.use_sample_iterations:
        sample_iterations = None

    if args.algorithm_list_id >= 0:
        algorithm_list = algorithm_lists[args.algorithm_list_id]

    huggingface_hub.login(args.hf_key)
    out = train_optimizers(args.task, args.num_epochs, args.batch_size, algorithm_list, num_epoch_to_save=args.num_epoch_to_save)

if __name__ == "__main__":
    main()
