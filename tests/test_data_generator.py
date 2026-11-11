"""Tests for synthetic e-commerce data generation."""

import csv
from pathlib import Path

import pytest

from src.data_generator import SyntheticDataGenerator, generate_data


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read a generated CSV file as dictionaries."""
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def test_generate_data_creates_linked_csv_files(tmp_path: Path) -> None:
    """Generated orders should reference existing customers and products."""
    paths = generate_data(
        num_customers=4,
        num_products=3,
        num_orders=10,
        output_dir=tmp_path,
        seed=7,
    )

    assert set(paths) == {"customers", "products", "orders"}
    customers = _read_rows(paths["customers"])
    products = _read_rows(paths["products"])
    orders = _read_rows(paths["orders"])
    customer_ids = {row["customer_id"] for row in customers}
    product_ids = {row["product_id"] for row in products}

    assert len(customers) == 4
    assert len(products) == 3
    assert len(orders) == 10
    assert all(row["customer_id"] in customer_ids for row in orders)
    assert all(row["product_id"] in product_ids for row in orders)
    assert all(
        round(float(row["unit_price"]) * int(row["quantity"]), 2)
        == float(row["total_amount"])
        for row in orders
    )


def test_generate_data_supports_empty_output(tmp_path: Path) -> None:
    """Zero record counts should produce header-only CSVs."""
    paths = generate_data(0, 0, 0, output_dir=tmp_path)

    assert all(_read_rows(path) == [] for path in paths.values())


def test_generate_data_rejects_negative_counts(tmp_path: Path) -> None:
    """Negative record counts should fail with a clear validation error."""
    with pytest.raises(ValueError, match="non-negative"):
        generate_data(num_customers=-1, output_dir=tmp_path)


def test_generate_data_requires_entities_for_orders(tmp_path: Path) -> None:
    """Orders cannot be created without customers and products."""
    with pytest.raises(ValueError, match="at least one customer"):
        generate_data(
            num_customers=0,
            num_products=1,
            num_orders=1,
            output_dir=tmp_path,
        )


def test_synthetic_generator_returns_valid_dataframes() -> None:
    """The generator should return the requested pandas schemas and values."""
    generator = SyntheticDataGenerator(
        num_customers=100,
        num_products=10,
        num_orders=2_000,
        seed=17,
    )
    generated = generator.generate_all()
    customers = generated["customers"]
    products = generated["products"]
    orders = generated["orders"]

    assert list(customers.columns) == [
        "customer_id",
        "name",
        "email",
        "age",
        "city",
        "country",
        "registration_date",
    ]
    assert list(products.columns) == [
        "product_id",
        "name",
        "category",
        "price",
        "stock",
        "rating",
    ]
    assert list(orders.columns) == [
        "order_id",
        "customer_id",
        "product_id",
        "quantity",
        "order_date",
    ]
    assert customers["age"].between(18, 90).all()
    assert products["price"].between(10, 500).all()
    assert products["rating"].between(1, 5).all()
    assert products["category"].isin(
        {"Electronics", "Clothing", "Home", "Sports", "Books"}
    ).all()
    assert orders["quantity"].between(1, 10).all()
    assert set(orders["customer_id"]).issubset(set(customers["customer_id"]))
    assert set(orders["product_id"]).issubset(set(products["product_id"]))

    top_customer_count = 20
    top_customer_order_share = (
        orders["customer_id"]
        .value_counts()
        .nlargest(top_customer_count)
        .sum()
        / len(orders)
    )
    assert 0.76 <= top_customer_order_share <= 0.84
