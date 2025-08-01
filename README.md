# When Mixture-of-Experts Meets Time Series Forecasting: A Shift-Resilient Gate Framework

This is the official PyTorch implementation of our paper: **"When Mixture-of-Experts Meets Time Series Forecasting: A Shift-Resilient Gate Framework"**. Our model introduces a novel Mixture-of-Experts (MoE) architecture with EMA-based gating mechanism specifically designed for time series forecasting.


## Key Features

Our framework addresses the critical challenge of distribution shifts in time series forecasting through innovative architectural design:

### **Mixture-of-Experts Architecture**
- **Multi-Expert Integration**: Supports diverse expert models (PatchTST, NLinear, TCN)
- **Dynamic Expert Selection**: Novel gating mechanism for optimal expert combination


### **EMA-based Shift-Resilient Gating**
- **Exponential Moving Average**: Trend-seasonal decomposition with SC-EMA
- **Dual-Path Gating**: Separate pathways for trend and seasonal components
- **Monte Carlo Dropout**: Uncertainty quantification for robust predictions

### **Expert Models**
1. **PatchTST Expert**: Patch-based Transformer for capturing local patterns
2. **NLinear Expert**: Normalized linear model for stable predictions
3. **TCN Expert**: Temporal Convolutional Network for sequential modeling


## Requirements

- Python 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.0 (optional, for GPU acceleration)

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data

### Supported Datasets

The following datasets are supported for long-term forecasting tasks:

- **ETT (Electricity Transformer Temperature)**: ETTh1, ETTh2, ETTm1, ETTm2
- **Weather**: Weather forecast data with 21 meteorological indicators
- **Exchange Rate**: Exchange rates of 8 countries
- **Traffic**: California traffic flow data from 862 sensors  
- **Solar**: Solar power generation data from 137 power plants

### Data Download

The download links for the datasets can be found in the code repositories of Informer, Autoformer, and TimeMixer.

Place the downloaded data files in the `./dataset/` directory.


### Data Preprocessing for Solar Dataset

The original Solar dataset requires preprocessing for time series analysis. We provide a preprocessed version `solar_AL_dates.csv` with the following modifications:

- **Date column added**: Time indexing with daily frequency starting from "2016-07-01"
- **Variable naming**: Columns renamed to 'date', 'sensor_0', 'sensor_1', ..., 'sensor_n' for clarity

The preprocessed dataset is ready for direct use in forecasting tasks without additional preprocessing steps.

## Usage

### Basic Training Command

```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id ETTh1_96_96 \
    --model SREMC_MoE \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --data ETTh1 \
    --enc_in 7 \
    --c_out 7 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --num_experts 3 \
    --activated_experts 2 \
    --expert_type TCN \
    --alpha 0.3 \
    --entropy_weight 0.1
```

### Parameter Description

We provide a detailed command description for training and testing the model:

```python
python run.py --model <model> --data <data> --task_name <task_name>
--seq_len <seq_len> --label_len <label_len> --pred_len <pred_len>
--enc_in <enc_in> --dec_in <dec_in> --c_out <c_out>
--num_experts <num_experts> --activated_experts <activated_experts>
--expert_type <expert_type> --alpha <alpha> --entropy_weight <entropy_weight>
--train_epochs <train_epochs> --batch_size <batch_size>
--learning_rate <learning_rate> --patience <patience>
--features <features> --des <des> --itr <itr>
```

The detailed descriptions about the arguments are as follows:

| Parameter name | Description of parameter |
| --- | --- |
| model | The model of experiment (defaults to `SREMC_MoE`) |
| data | The dataset type |
| task_name | The forecasting task (defaults to `long_term_forecast`) |
| seq_len | Input sequence length (defaults to 96) |
| pred_len | Prediction sequence length (defaults to 96) |
| features | The forecasting task (defaults to `M`). Can be set to `M`,`S`,`MS` |
| **MoE Parameters** | |
| num_experts | Number of experts in MoE (defaults to 3) |
| activated_experts | Number of activated experts (defaults to 2) |
| expert_type | Type of expert models (PatchTST, DLinear, NLinear, TCN) |
| alpha | EMA smoothing parameter (defaults to 0.3) |
| entropy_weight | Weight for entropy loss (defaults to 0.1) |
| **Training Parameters** | |
| train_epochs | Training epochs (defaults to 100) |
| batch_size | Batch size (defaults to 32) |
| learning_rate | Learning rate (defaults to 0.0001) |
| patience | Early stopping patience (defaults to 3) |

### Example Commands

#### Weather Dataset Forecasting
```bash
python run.py --task_name long_term_forecast --is_training 1 --model_id Weather_96_96 --model SREMC_MoE --expert_type NLinear --root_path ./dataset/ --data_path weather.csv --data custom --features M --seq_len 96 --pred_len 96 --enc_in 21 --dec_in 21 --c_out 21
```


## Results

We experiment on three benchmarks covering diverse real-world applications and achieves satisfactory performance in long-term forecasting tasks.

