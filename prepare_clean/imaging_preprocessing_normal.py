import argparse

from imaging_preprocessing import run_pipeline

def parse_args():

    parser = argparse.ArgumentParser(description="Run imaging preprocessing (pydevconsole-safe).")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="0 means process all patients; >0 means process first N patients.",
    )
    args, _unknown = parser.parse_known_args()
    return args

def main():

    args = parse_args()
    run_pipeline(args.max_cases)

if __name__ == "__main__":
    main()
