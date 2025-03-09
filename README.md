# 🏞️ PRCF Reservation


## 📊 Project Overview

**PRCF Reservation** is a comprehensive data science project analyzing Parks, Recreation and Community Facilities (PRCF) data from Mesa, Arizona. This project leverages open government data to understand visitation patterns and forecast future attendance levels at various parks and tourist centers throughout Mesa.

## ✨ Key Features

- **Data Extraction & Exploration**: Comprehensive sourcing from government open data portals
- **SQL-Based EDA**: In-depth exploratory data analysis using SQL queries
- **Time Series Forecasting**: Advanced machine learning models to predict annual attendance levels
- **Comparative Model Analysis**: Evaluation of different forecasting approaches

## 📁 Project Structure

The project consists of the following key notebooks:

### 1. Data Extraction & Initial EDA
- Sourcing data from government open data portals
- Preliminary data cleaning and exploration
- Identification of key patterns and trends

### 2. SQL-Based Exploratory Data Analysis
- Advanced data querying and transformation
- Statistical analysis of visitation patterns
- Correlation analysis between different facilities and time periods

### 3. Machine Learning Models
Two supervised learning models for time series forecasting using TimeSeriesSplit:

#### Model A: Split-Train-Evaluate Approach
- Implements multiple time series splits
- Trains and evaluates models on each split independently
- Final model selection based on weighted average performance scores across splits

#### Model B: Train-Split-Evaluate Approach
- Trains models using the complete dataset
- Implements time series splits for evaluation purposes only
- Uses weighted performance scoring for final model selection

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/demarquezs/PRCF-Reservation-DS-Project.git

# Navigate to project directory
cd prcf-reservation

# Install required dependencies
pip install -r requirements.txt

# Run Jupyter notebooks
jupyter notebook
