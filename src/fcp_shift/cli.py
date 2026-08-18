from __future__ import annotations

import argparse
import logging

from fcp_shift.config import filter_config, load_config
from fcp_shift.experiments import run_asymptotic, run_covariate_shift, run_transport_shift
from fcp_shift.reporting import make_grouped_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fcp-shift")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run an experiment from YAML")
    run.add_argument("--config", required=True)
    run.add_argument("--dataset")
    run.add_argument("--weight")
    run.add_argument("--rho", type=float)
    run.add_argument("--seed", type=int)
    run.add_argument("--force", action="store_true")
    run.add_argument("--log-level", default="INFO")
    figures = subparsers.add_parser(
        "figures", help="Aggregate completed weights into shared main figures"
    )
    figures.add_argument("--config", required=True)
    figures.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = load_config(args.config)
    if args.command == "figures":
        generated = make_grouped_figures(config)
        for path in generated:
            print(path)
        return
    config = filter_config(config, args.dataset, args.weight, args.rho, args.seed)
    kind = config["experiment"]["kind"]
    runners = {
        "covariate_shift": run_covariate_shift,
        "transport_shift": run_transport_shift,
        "asymptotic": run_asymptotic,
    }
    runners[kind](config, force=args.force)


if __name__ == "__main__":
    main()
