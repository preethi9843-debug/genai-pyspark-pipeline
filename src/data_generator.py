"""Generate synthetic e-commerce data with pandas, NumPy, and Faker."""

import argparse
import logging
import math
from pathlib import Path
import random

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from src.config import configure_logging, get_config

logger = logging.getLogger(__name__)

CATEGORIES = ("Electronics", "Clothing", "Home", "Sports", "Books")
ORDER_STATUSES = ("delivered", "shipped", "processing", "cancelled")
CUSTOMER_COLUMNS = (
    "customer_id",
    "name",
    "email",
    "age",
    "city",
    "country",
    "registration_date",
)
PRODUCT_COLUMNS = (
    "product_id",
    "name",
    "category",
    "price",
    "stock",
    "rating",
)
ORDER_COLUMNS = (
    "order_id",
    "customer_id",
    "product_id",
    "quantity",
    "order_date",
)


class SyntheticDataGenerator:
    """Create related synthetic customer, product, and order DataFrames."""

    def __init__(
        self,
        num_customers: int = 100_000,
        num_products: int = 10_000,
        num_orders: int = 1_000_000,
        seed: int = 42,
    ) -> None:
        """Initialize record counts and reproducible random generators."""
        self._validate_count(num_customers, "num_customers")
        self._validate_count(num_products, "num_products")
        self._validate_count(num_orders, "num_orders")
        self.num_customers = num_customers
        self.num_products = num_products
        self.num_orders = num_orders
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._faker = Faker()
        self._faker.seed_instance(seed)
        self._customers: pd.DataFrame | None = None
        self._products: pd.DataFrame | None = None

    @staticmethod
    def _validate_count(count: int, name: str) -> None:
        """Reject negative record counts with a clear error."""
        if count < 0:
            raise ValueError(f"{name} must be non-negative.")

    def generate_customers(self, count: int | None = None) -> pd.DataFrame:
        """Generate customers, including ages normally distributed around 35."""
        customer_count = self.num_customers if count is None else count
        self._validate_count(customer_count, "count")
        logger.info("Generating %d customers.", customer_count)

        ages = np.clip(
            np.rint(self._rng.normal(loc=35, scale=10, size=customer_count)),
            18,
            90,
        ).astype(np.int16)
        registration_dates = [
            self._faker.date_between(start_date="-5y", end_date="today")
            for _ in range(customer_count)
        ]
        customer_rows = [
            {
                "customer_id": f"C{index:06d}",
                "name": self._faker.name(),
                "email": self._faker.unique.email(),
                "age": int(ages[index - 1]),
                "city": self._faker.city(),
                "country": self._faker.country(),
                "registration_date": registration_dates[index - 1],
            }
            for index in tqdm(
                range(1, customer_count + 1),
                desc="Generating customers",
                unit="customer",
            )
        ]
        self._customers = pd.DataFrame(customer_rows, columns=CUSTOMER_COLUMNS)
        logger.info("Generated %d customers.", len(self._customers))
        return self._customers

    def generate_products(self, count: int | None = None) -> pd.DataFrame:
        """Generate products with prices, stock levels, and ratings."""
        product_count = self.num_products if count is None else count
        self._validate_count(product_count, "count")
        logger.info("Generating %d products.", product_count)

        products = [
            {
                "product_id": f"P{index:05d}",
                "name": f"{self._faker.word().title()} {self._faker.word().title()}",
                "category": str(self._rng.choice(CATEGORIES)),
                "price": round(float(self._rng.uniform(10, 500)), 2),
                "stock": int(self._rng.integers(0, 501)),
                "rating": round(float(self._rng.uniform(1, 5)), 1),
            }
            for index in tqdm(
                range(1, product_count + 1),
                desc="Generating products",
                unit="product",
            )
        ]
        self._products = pd.DataFrame(products, columns=PRODUCT_COLUMNS)
        logger.info("Generated %d products.", len(self._products))
        return self._products

    def generate_orders(
        self,
        customers: pd.DataFrame | None = None,
        products: pd.DataFrame | None = None,
        count: int | None = None,
    ) -> pd.DataFrame:
        """Generate orders with an 80/20 customer ordering distribution."""
        order_count = self.num_orders if count is None else count
        self._validate_count(order_count, "count")

        customer_data = customers
        if customer_data is None:
            customer_data = self._customers
        if customer_data is None:
            customer_data = self.generate_customers()

        product_data = products
        if product_data is None:
            product_data = self._products
        if product_data is None:
            product_data = self.generate_products()

        if order_count and customer_data.empty:
            raise ValueError("Orders require at least one customer.")
        if order_count and product_data.empty:
            raise ValueError("Orders require at least one product.")
        for required_column in ("customer_id",):
            if required_column not in customer_data.columns:
                raise ValueError(f"Customer data must include {required_column!r}.")
        for required_column in ("product_id",):
            if required_column not in product_data.columns:
                raise ValueError(f"Product data must include {required_column!r}.")

        logger.info("Generating %d orders.", order_count)
        customer_count = len(customer_data)
        if order_count == 0:
            return pd.DataFrame(columns=ORDER_COLUMNS)

        top_customer_count = max(1, math.ceil(customer_count * 0.2))
        pareto_weights = self._rng.pareto(a=1.16, size=customer_count) + 1
        top_customer_indexes = np.argpartition(
            pareto_weights, customer_count - top_customer_count
        )[-top_customer_count:]
        top_customer_mask = np.zeros(customer_count, dtype=bool)
        top_customer_mask[top_customer_indexes] = True
        customer_weights = np.zeros(customer_count, dtype=float)
        if top_customer_count == customer_count:
            customer_weights = pareto_weights / pareto_weights.sum()
        else:
            customer_weights[top_customer_mask] = (
                0.8
                * pareto_weights[top_customer_mask]
                / pareto_weights[top_customer_mask].sum()
            )
            customer_weights[~top_customer_mask] = (
                0.2
                * pareto_weights[~top_customer_mask]
                / pareto_weights[~top_customer_mask].sum()
            )
        sampled_customer_indexes = self._rng.choice(
            customer_count, size=order_count, p=customer_weights
        )
        sampled_product_indexes = self._rng.integers(
            0, len(product_data), size=order_count
        )
        sampled_customers = customer_data.iloc[sampled_customer_indexes]
        sampled_products = product_data.iloc[sampled_product_indexes]
        today = pd.Timestamp.today().normalize()

        quantities = self._rng.integers(1, 11, size=order_count)
        order_dates = today - pd.to_timedelta(
            self._rng.integers(0, 366, size=order_count), unit="D"
        )
        order_batches: list[pd.DataFrame] = []
        for start in tqdm(
            range(0, order_count, 10_000),
            desc="Generating orders",
            unit="10k orders",
        ):
            end = min(start + 10_000, order_count)
            order_batches.append(
                pd.DataFrame(
                    {
                        "order_id": [
                            f"O{index:07d}" for index in range(start + 1, end + 1)
                        ],
                        "customer_id": sampled_customers["customer_id"]
                        .iloc[start:end]
                        .to_numpy(),
                        "product_id": sampled_products["product_id"]
                        .iloc[start:end]
                        .to_numpy(),
                        "quantity": quantities[start:end],
                        "order_date": order_dates[start:end],
                    },
                    columns=ORDER_COLUMNS,
                )
            )
        orders = pd.concat(order_batches, ignore_index=True)

        logger.info("Generated %d orders.", len(orders))
        return orders

    def generate_all(self) -> dict[str, pd.DataFrame]:
        """Generate and return customer, product, and order DataFrames."""
        customers = self.generate_customers()
        products = self.generate_products()
        orders = self.generate_orders(customers=customers, products=products)
        return {"customers": customers, "products": products, "orders": orders}


def _write_csv(file_path: Path, dataframe: pd.DataFrame) -> Path:
    """Write a DataFrame to a UTF-8 CSV file and return its path."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(file_path, index=False, encoding="utf-8")
    logger.info("Wrote %d rows to %s", len(dataframe), file_path)
    return file_path


def generate_data(
    num_customers: int = 100_000,
    num_products: int = 10_000,
    num_orders: int = 1_000_000,
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate the data and write CSVs compatible with the Spark pipeline."""
    generated = SyntheticDataGenerator(
        num_customers=num_customers,
        num_products=num_products,
        num_orders=num_orders,
        seed=seed,
    ).generate_all()
    destination = output_dir or get_config().raw_data_dir

    customers = generated["customers"].copy()
    names = customers["name"].str.split(" ", n=1, expand=True)
    if names.shape[1] == 0:
        customers["first_name"] = pd.Series(dtype="object")
        customers["last_name"] = pd.Series(dtype="object")
    else:
        customers["first_name"] = names[0]
        customers["last_name"] = names[1] if names.shape[1] > 1 else ""
    customers = customers.rename(columns={"registration_date": "signup_date"})
    customers = customers[
        [
            "customer_id",
            "first_name",
            "last_name",
            "email",
            "city",
            "country",
            "signup_date",
        ]
    ]

    products = generated["products"].rename(
        columns={"name": "product_name", "stock": "stock_quantity"}
    )[["product_id", "product_name", "category", "price", "stock_quantity"]]
    orders = generated["orders"].merge(
        products[["product_id", "price"]].rename(columns={"price": "unit_price"}),
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    orders["total_amount"] = (
        pd.to_numeric(orders["unit_price"])
        * pd.to_numeric(orders["quantity"])
    ).round(2)
    rng = random.Random(seed)
    orders["status"] = [
        rng.choice(ORDER_STATUSES) for _ in range(len(orders))
    ]
    orders = orders[
        [
            "order_id",
            "customer_id",
            "product_id",
            "order_date",
            "quantity",
            "unit_price",
            "total_amount",
            "status",
        ]
    ]
    paths = {
        "customers": _write_csv(destination / "customers.csv", customers),
        "products": _write_csv(destination / "products.csv", products),
        "orders": _write_csv(destination / "orders.csv", orders),
    }
    logger.info(
        "Generated %d customers, %d products, and %d orders.",
        num_customers,
        num_products,
        num_orders,
    )
    return paths


def main() -> None:
    """Parse command-line options and generate CSV data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=100_000)
    parser.add_argument("--products", type=int, default=10_000)
    parser.add_argument("--orders", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging()
    generate_data(
        num_customers=args.customers,
        num_products=args.products,
        num_orders=args.orders,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
