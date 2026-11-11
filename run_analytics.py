"""Run the PySpark sales analytics against generated Parquet data."""

from pathlib import Path
from time import perf_counter
from typing import Callable

from pyspark.sql import DataFrame, SparkSession

from src.spark_analytics import SalesAnalytics


def _show_timed(label: str, operation: Callable[[], DataFrame]) -> None:
    """Run an analytics operation, display its results, and print elapsed time."""
    started_at = perf_counter()
    result = operation()
    result.show()
    print(f"{label} execution time: {perf_counter() - started_at:.2f} seconds")


def main() -> None:
    """Load generated Parquet data and display each sales analytics result."""
    spark: SparkSession = SalesAnalytics.create_spark_session()
    try:
        raw_data_dir = Path(__file__).resolve().parent / "data" / "raw"
        SalesAnalytics.load_parquet(raw_data_dir / "customers.parquet")
        products_df = SalesAnalytics.load_parquet(raw_data_dir / "products.parquet")
        orders_df = SalesAnalytics.load_parquet(raw_data_dir / "orders.parquet")

        _show_timed(
            "Top customers by revenue",
            lambda: SalesAnalytics.top_customers_by_revenue(orders_df, products_df),
        )
        _show_timed(
            "Sales by category",
            lambda: SalesAnalytics.sales_by_category(orders_df, products_df),
        )
        _show_timed(
            "Monthly trends",
            lambda: SalesAnalytics.monthly_trends(orders_df, products_df),
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
