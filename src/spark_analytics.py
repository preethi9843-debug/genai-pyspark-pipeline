"""Analyze generated e-commerce data with PySpark."""

import argparse
import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.config import configure_logging, get_config

logger = logging.getLogger(__name__)


class SalesAnalytics:
    """Provide Spark setup and e-commerce sales analysis operations."""

    @staticmethod
    def create_spark_session() -> SparkSession:
        """Create or retrieve a local Spark session configured with 4 GB memory."""
        return (
            SparkSession.builder.appName("EcommerceSalesAnalytics")
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.legacy.parquet.nanosAsLong", "true")
            .config(
                "spark.serializer",
                "org.apache.spark.serializer.KryoSerializer",
            )
            .getOrCreate()
        )

    @staticmethod
    def load_parquet(path: str | Path) -> DataFrame:
        """Load a Parquet file or directory into a Spark DataFrame."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Parquet input path was not found: {source}")
        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError(
                "No active Spark session. Call create_spark_session() first."
            )
        spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")
        dataframe = spark.read.parquet(str(source))
        if (
            "order_date" in dataframe.columns
            and isinstance(dataframe.schema["order_date"].dataType, LongType)
        ):
            dataframe = dataframe.withColumn(
                "order_date",
                F.expr("timestamp_micros(order_date DIV 1000)"),
            )
        return dataframe

    @staticmethod
    def _orders_with_revenue(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Join orders to products and calculate line-item revenue."""
        return (
            orders_df.alias("orders")
            .join(
                products_df.select("product_id", "price").alias("products"),
                on="product_id",
                how="inner",
            )
            .withColumn(
                "line_revenue",
                F.col("quantity").cast("double") * F.col("price").cast("double"),
            )
        )

    @staticmethod
    def top_customers_by_revenue(
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10,
    ) -> DataFrame:
        """Return the top N customers ranked by revenue from their orders."""
        if n < 1:
            raise ValueError("n must be a positive integer.")
        return (
            SalesAnalytics._orders_with_revenue(orders_df, products_df)
            .groupBy("customer_id")
            .agg(F.round(F.sum("line_revenue"), 2).alias("total_spend"))
            .orderBy(F.desc("total_spend"))
            .limit(n)
        )

    @staticmethod
    def sales_by_category(
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Aggregate revenue and units sold for each product category."""
        category_products = products_df.select("product_id", "category", "price")
        return (
            orders_df.join(category_products, on="product_id", how="inner")
            .groupBy("category")
            .agg(
                F.round(
                    F.sum(
                        F.col("quantity").cast("double")
                        * F.col("price").cast("double")
                    ),
                    2,
                ).alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
            )
            .orderBy(F.desc("total_revenue"))
        )

    @staticmethod
    def monthly_trends(
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Calculate monthly revenue and month-over-month revenue growth."""
        revenue_by_month = (
            SalesAnalytics._orders_with_revenue(orders_df, products_df)
            .withColumn("month", F.date_format(F.to_date("order_date"), "yyyy-MM"))
            .groupBy("month")
            .agg(F.sum("line_revenue").alias("total_revenue"))
        )
        month_window = Window.orderBy("month")
        with_previous_month = revenue_by_month.withColumn(
            "previous_month_revenue",
            F.lag("total_revenue").over(month_window),
        )
        return (
            with_previous_month.withColumn(
                "revenue_growth_pct",
                F.when(
                    F.col("previous_month_revenue").isNull()
                    | (F.col("previous_month_revenue") == 0),
                    F.lit(None).cast("double"),
                ).otherwise(
                    F.round(
                        (
                            F.col("total_revenue")
                            - F.col("previous_month_revenue")
                        )
                        / F.col("previous_month_revenue")
                        * 100,
                        2,
                    )
                ),
            )
            .select(
                "month",
                F.round("total_revenue", 2).alias("total_revenue"),
                F.round("previous_month_revenue", 2).alias(
                    "previous_month_revenue"
                ),
                "revenue_growth_pct",
            )
            .orderBy("month")
        )


CUSTOMER_SCHEMA = StructType(
    [
        StructField("customer_id", StringType(), False),
        StructField("first_name", StringType(), True),
        StructField("last_name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("city", StringType(), True),
        StructField("country", StringType(), True),
        StructField("signup_date", DateType(), True),
    ]
)
PRODUCT_SCHEMA = StructType(
    [
        StructField("product_id", StringType(), False),
        StructField("product_name", StringType(), True),
        StructField("category", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("stock_quantity", IntegerType(), True),
    ]
)
ORDER_SCHEMA = StructType(
    [
        StructField("order_id", StringType(), False),
        StructField("customer_id", StringType(), False),
        StructField("product_id", StringType(), False),
        StructField("order_date", DateType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("unit_price", DoubleType(), True),
        StructField("total_amount", DoubleType(), True),
        StructField("status", StringType(), True),
    ]
)


def _read_csv(spark: SparkSession, path: Path, schema: StructType) -> DataFrame:
    """Read a headered CSV file using its explicit schema."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input file was not found: {path}")
    return (
        spark.read.option("header", True)
        .option("dateFormat", "yyyy-MM-dd")
        .schema(schema)
        .csv(str(path))
    )


def _write_report(dataframe: DataFrame, output_path: Path) -> Path:
    """Write a Spark DataFrame as a CSV output directory."""
    dataframe.write.mode("overwrite").option("header", True).csv(str(output_path))
    logger.info("Wrote analytics report to %s", output_path)
    return output_path


def analyze_data(
    raw_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
    spark: SparkSession | None = None,
) -> dict[str, Path]:
    """Create sales and customer reports from the generated CSV data.

    Spark CSV output paths are directories containing one or more part files.
    A Spark session passed by the caller is left running; a session created
    here is stopped before returning.
    """
    config = get_config()
    input_dir = raw_data_dir or config.raw_data_dir
    output_dir = processed_data_dir or config.processed_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    owns_spark_session = spark is None
    active_spark = spark or (
        SparkSession.builder.appName("EcommerceDataPipeline")
        .master("local[*]")
        .getOrCreate()
    )

    try:
        customers = _read_csv(
            active_spark, input_dir / "customers.csv", CUSTOMER_SCHEMA
        )
        products = _read_csv(active_spark, input_dir / "products.csv", PRODUCT_SCHEMA)
        orders = _read_csv(active_spark, input_dir / "orders.csv", ORDER_SCHEMA)
        sales_orders = orders.filter(F.col("status") != "cancelled")

        product_sales = (
            sales_orders.join(products, "product_id")
            .groupBy("product_id", "product_name", "category")
            .agg(
                F.round(F.sum("total_amount"), 2).alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
                F.countDistinct("order_id").alias("order_count"),
            )
            .orderBy(F.desc("total_revenue"))
        )
        category_sales = (
            sales_orders.join(products, "product_id")
            .groupBy("category")
            .agg(
                F.round(F.sum("total_amount"), 2).alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
                F.countDistinct("order_id").alias("order_count"),
            )
            .orderBy(F.desc("total_revenue"))
        )
        monthly_sales = (
            sales_orders.withColumn("month", F.date_format("order_date", "yyyy-MM"))
            .groupBy("month")
            .agg(
                F.round(F.sum("total_amount"), 2).alias("total_revenue"),
                F.countDistinct("order_id").alias("order_count"),
            )
            .orderBy("month")
        )
        top_customers = (
            sales_orders.join(customers, "customer_id")
            .groupBy("customer_id", "first_name", "last_name")
            .agg(
                F.round(F.sum("total_amount"), 2).alias("total_spend"),
                F.countDistinct("order_id").alias("order_count"),
            )
            .orderBy(F.desc("total_spend"))
        )

        reports = {
            "sales_by_product": product_sales,
            "sales_by_category": category_sales,
            "sales_by_month": monthly_sales,
            "top_customers": top_customers,
        }
        return {
            name: _write_report(dataframe, output_dir / name)
            for name, dataframe in reports.items()
        }
    finally:
        if owns_spark_session:
            active_spark.stop()


def main() -> None:
    """Parse command-line paths and run the PySpark analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    args = parser.parse_args()

    configure_logging()
    analyze_data(
        raw_data_dir=args.raw_dir,
        processed_data_dir=args.processed_dir,
    )


if __name__ == "__main__":
    main()
