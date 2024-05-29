import sys
from train_utils import *
from landscape_utils import *
from dataset_utils import *
from algorithm_list import algorithm_lists
from algorithm_list import algorithm_list_sgd_machine_translation
from train_configs import *
import argparse

import huggingface_hub

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_key", type=str, help="Huggingface key required for access to the stack dataset")
    parser.add_argument("--num_epochs", default=20, type=int, help="Number of epochs")
    parser.add_argument("--lr", default=2e-4, type=float, help="Learning rate")
    parser.add_argument("--train_optimizer", default="adam", type=str, help="Train optimizer")
    parser.add_argument("--task", default="dbg", type=str, help="Task to run on")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--save_model", default=False, type=bool)
    parser.add_argument("--momentum", default=True, type=bool)
    parser.add_argument("--use_test_batch", default=True, type=bool)
    parser.add_argument("--num_sample_sharpness", default=1, type=int)
    parser.add_argument("--skip_search", default=False, type=bool)
    parser.add_argument("--use_sample_iterations", default=False, type=bool)
    parser.add_argument("--optimizer_device", default='cpu', type=str)
    parser.add_argument("--algorithm_list_id", default=-1, type=int)

    args = parser.parse_args()
    if not args.use_sample_iterations:
        sample_iterations = None

    if args.algorithm_list_id >= 0:
        algorithm_list = algorithm_lists[args.algorithm_list_id]

    huggingface_hub.login(args.hf_key)
    model, dataloader = load_model(args.task), load_dataloader(args.task)
    out = train(model,
                dataloader,
                args.num_epochs,
                args.lr,
                args.batch_size,
                algorithm_list,
                args.task,
                save_model=args.save_model,
                momentum=args.momentum,
                train_optimizer=args.train_optimizer,
                use_test_batch=args.use_test_batch,
                test_batch_size=args.batch_size,
                num_sample_sharpness=args.num_sample_sharpness,
                thresholds=thresholds,
                skip_search=args.skip_search,
                sample_iterations=sample_iterations,
                optimizer_device=args.optimizer_device)

if __name__ == "__main__":
    main()
