# E-commerce Data Pipeline

Generate synthetic customers, products, and orders, then analyze sales trends
with PySpark. The sample data is fake and intended for development and testing.

## Project structure

```text
genai-pyspark-pipeline/
├── data/
│   ├── raw/                 # Generated customers, products, and orders CSVs
│   └── processed/           # PySpark CSV report directories
├── notebooks/               # Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   └── spark_analytics.py
├── tests/
│   └── test_data_generator.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Prerequisites

- Python 3.10 or newer
- Java 17 (required by the supported PySpark versions)

On Windows, install an OpenJDK distribution from PowerShell:

```powershell
winget install -e --id EclipseAdoptium.Temurin.17.JDK
```

Restart the terminal after installation and verify Java is available:

```powershell
java -version
```

If PowerShell still cannot find `java`, add the JDK's `bin` directory to `PATH`
and set `JAVA_HOME` to the JDK installation directory, then open a new terminal.

## Setup (PowerShell)

Run these commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the pipeline

Generate the default 100,000 customers, 10,000 products, and 1,000,000 orders:

```powershell
python main.py
```

Choose different data sizes or a seed:

```powershell
python main.py --customers 500 --products 50 --orders 2000 --seed 7
```

The generated customer, product, and order Parquet files are written to
`data/raw/`. Run the PySpark analysis:

```powershell
python -m src.spark_analytics
```

Generated input files are written to `data/raw/`. Reports are written to
`data/processed/` as Spark CSV output directories: `sales_by_product`,
`sales_by_category`, `sales_by_month`, and `top_customers`. Each report
directory contains a headered CSV part file. Cancelled orders are excluded
from revenue and customer-spend reports.

## Run tests

```powershell
python -m pytest
```

The test suite checks record counts, order references, totals, and invalid
generator inputs.
