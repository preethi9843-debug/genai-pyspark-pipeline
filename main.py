"""Generate synthetic e-commerce data and save it as Parquet files."""

import argparse
import logging
import sys
from time import perf_counter

from src.config import configure_logging, get_config
from src.data_generator import SyntheticDataGenerator

logger = logging.getLogger(__name__)


def main() -> int:
    """Generate the data, save Parquet files, and report elapsed time and sizes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=100_000)
    parser.add_argument("--products", type=int, default=10_000)
    parser.add_argument("--orders", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging()
    started_at = perf_counter()
    try:
        config = get_config()
        output_dir = config.raw_data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = SyntheticDataGenerator(
            num_customers=args.customers,
            num_products=args.products,
            num_orders=args.orders,
            seed=args.seed,
        )
        datasets = generator.generate_all()

        for dataset_name, dataframe in datasets.items():
            if dataset_name == "orders":
                dataframe["order_date"] = dataframe["order_date"].astype(
                    "datetime64[us]"
                )
            output_path = output_dir / f"{dataset_name}.parquet"
            dataframe.to_parquet(output_path, index=False)
            size_bytes = output_path.stat().st_size
            print(f"{output_path.name}: {size_bytes:,} bytes")
            logger.info("Saved %d rows to %s", len(dataframe), output_path)

        print(f"Generation and saving time: {perf_counter() - started_at:.2f} seconds")
        return 0
    except Exception:
        logger.exception("Failed to generate or save synthetic data.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
