# BTCUSDT Data Pipeline

This directory contains scripts for the Project Compound's data acquisition and feature engineering pipeline for BTC/USDT trading data.

## Overview

The data pipeline consists of the following components:

1. **download_btcusdt_data.py**: Downloads historical BTC/USDT data from Binance
2. **process_btcusdt_data.py**: Applies feature engineering to the raw data
3. **data_pipeline.py**: End-to-end pipeline that runs both download and processing steps
4. **validate_data.py**: Validates data quality and performs exploratory data analysis

## Prerequisites

Before running these scripts, ensure you have:

1. Installed all required dependencies (`requirements.txt`)
2. Set up your Binance API keys in `config.py`
3. Ensured all core Project Compound modules are in place (config.py, binance_interface.py, data_handler.py, feature_engineering.py, utils.py)

## Usage

### Complete Pipeline

For most use cases, you'll want to run the complete pipeline:

```bash
python data_pipeline.py --start_date 2023-01-01 --end_date 2023-12-31 --timeframes 1h 4h 1d --visualize
```

This will:
- Download BTC/USDT data for 1-hour, 4-hour and daily timeframes for 2023
- Process and engineer features for each dataset
- Generate visualizations for the features

### Individual Components

You can also run each component separately:

#### Download Data Only

```bash
python download_btcusdt_data.py --start_date 2023-01-01 --end_date 2023-12-31 --timeframe 1h
```

#### Process Data Only

```bash
python process_btcusdt_data.py --input_file data/raw/BTCUSDT_1h_2023-01-01_2023-12-31.csv --visualize
```

#### Validate Data Quality

```bash
python validate_data.py --input_file data/processed/BTCUSDT_1h_2023-01-01_2023-12-31_processed.csv --timeframe 1h
```

## Command Line Arguments

### data_pipeline.py

| Argument | Description | Default |
|----------|-------------|---------|
| --start_date | Start date in YYYY-MM-DD format | 1 year ago |
| --end_date | End date in YYYY-MM-DD format | Today |
| --timeframes | List of timeframes to download | ['1h', '4h', '1d'] |
| --raw_dir | Directory to save raw data | data/raw |
| --processed_dir | Directory to save processed data | data/processed |
| --visualize | Generate visualizations of features | False |
| --viz_dir | Directory to save visualizations | visualizations |
| --symbol | Trading pair symbol | BTCUSDT |

### download_btcusdt_data.py

| Argument | Description | Default |
|----------|-------------|---------|
| --start_date | Start date in YYYY-MM-DD format | 1 year ago |
| --end_date | End date in YYYY-MM-DD format | Today |
| --timeframe | Timeframe for the data | 1h |
| --output_dir | Directory to save downloaded data | data/raw |
| --symbol | Trading pair symbol | BTCUSDT |

### process_btcusdt_data.py

| Argument | Description | Default |
|----------|-------------|---------|
| --input_file | Path to input CSV file with raw data | Required |
| --output_dir | Directory to save processed data | data/processed |
| --visualize | Generate visualizations of features | False |
| --viz_dir | Directory to save visualizations | visualizations |

### validate_data.py

| Argument | Description | Default |
|----------|-------------|---------|
| --input_file | Path to input CSV file (raw or processed) | Required |
| --output_dir | Directory to save validation reports | reports/data_validation |
| --timeframe | Timeframe of the data (for validation checks) | None |

## Output Structure

The pipeline creates the following directory structure:

```
project_compound/
│
├── data/
│   ├── raw/                  # Raw OHLCV data from Binance
│   └── processed/            # Processed data with engineered features
│
├── visualizations/           # Feature visualizations
│   ├── distributions/        # Distribution plots for features
│   └── timeseries/           # Time series plots
│
└── reports/
    └── data_validation/      # Data validation reports
        └── visualizations/   # EDA visualizations
```

## Data Quality Checks

The `validate_data.py` script performs the following checks:

1. Missing values analysis
2. Data type validation
3. Time series continuity (missing timestamps)
4. Price anomalies detection (extreme price changes)
5. Volume anomalies detection (unusual volume spikes)
6. Basic statistical analysis
7. Exploratory data visualizations

## Troubleshooting

### Common Issues

1. **API Rate Limits**: If you're downloading a lot of data, you may hit Binance's API rate limits. The scripts have built-in retry logic, but you may need to add delays or split your date ranges.

2. **Missing Data**: For some timeframes, especially during exchange maintenance periods, data may be missing. The validation script helps identify these gaps.

3. **Performance Issues**: Processing very large datasets can be memory-intensive. If you encounter memory errors, try processing smaller date ranges or using a lower resolution timeframe.

### Logs

All scripts use the logger from `utils.py`. Logs are written to both the console and log files in the `logs/` directory. Check these logs for troubleshooting information.

## Next Steps

After running the data pipeline:

1. Use the validation reports to assess data quality
2. Examine feature visualizations to understand their distributions and relationships
3. Feed the processed data into the model training pipeline
4. Periodically update your dataset with new data to keep your models current

## Contributing

When adding new features to the data pipeline, please follow these guidelines:

1. Add appropriate logging for all operations
2. Include proper error handling
3. Add unit tests for new functionality
4. Update this README with new arguments or components