import argparse

from processing.downloader import Banking77Downloader


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BANKING77 to local raw folder")
    parser.add_argument("--output_dir", type=str, default="dataset/raw/banking77")
    args = parser.parse_args()

    stats = Banking77Downloader(output_dir=args.output_dir).download()
    print(f"Saved BANKING77 to: {stats['output_dir']}")
    print(
        f"Train rows: {stats['train_rows']} | "
        f"Test rows: {stats['test_rows']} | Intents: {stats['num_intents']}"
    )


if __name__ == "__main__":
    main()
