"""Train the Liu receiver-side adaptation (M3) with the shared DeepSC runner."""

from train_multi_vocab_one_student import build_receiver_parser, run_training
from utils.liu_kd import (
    METHOD_ID,
    configure_liu_objective,
    liu_distillation_terms,
    liu_objective_manifest,
)


def parse_args(argv=None):
    parser = build_receiver_parser(f"Liu receiver-side adaptation ({METHOD_ID})")
    parser.add_argument("--liu-decoder-weight", type=float, required=True)
    parser.add_argument("--liu-logit-weight", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        configure_liu_objective(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    return run_training(
        args,
        model_type="Liu et al. receiver-side adaptation",
        loss_config=liu_objective_manifest(args),
        checkpoint_key="ce",
        distill=liu_distillation_terms,
        method_files=("train_liu_rx.py", "utils/liu_kd.py"),
    )


if __name__ == "__main__":
    main()
