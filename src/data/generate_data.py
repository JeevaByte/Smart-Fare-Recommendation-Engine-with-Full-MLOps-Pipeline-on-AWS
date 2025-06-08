#!/usr/bin/env python
"""
Data simulation script for UK train fare recommendation engine.
Generates synthetic data representing train journeys with various features.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import argparse

# UK train operators
TRAIN_OPERATORS = [
    'LNER', 'GWR', 'Avanti West Coast', 'CrossCountry', 
    'TransPennine Express', 'ScotRail', 'Northern', 
    'South Western Railway', 'Southeastern', 'East Midlands Railway'
]

# Major UK stations
STATIONS = [
    'London Kings Cross', 'London Euston', 'London Paddington', 'London Waterloo',
    'Manchester Piccadilly', 'Birmingham New Street', 'Edinburgh Waverley',
    'Glasgow Central', 'Leeds', 'Liverpool Lime Street', 'Bristol Temple Meads',
    'Cardiff Central', 'Newcastle', 'Sheffield', 'York', 'Nottingham',
    'Reading', 'Oxford', 'Cambridge', 'Brighton', 'Southampton Central'
]

# User types
USER_TYPES = ['standard', 'business', 'student', 'senior', 'family']

def generate_data(num_samples=10000, output_path=None):
    """
    Generate synthetic train fare data
    
    Args:
        num_samples: Number of data points to generate
        output_path: Path to save the CSV file
        
    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate random data
    data = {
        'origin_station': np.random.choice(STATIONS, num_samples),
        'destination_station': np.random.choice(STATIONS, num_samples),
        'booking_days_ahead': np.random.randint(0, 90, num_samples),
        'travel_time_minutes': np.random.randint(15, 480, num_samples),
        'time_of_day': np.random.randint(0, 24, num_samples),
        'day_of_week': np.random.randint(0, 7, num_samples),
        'train_operator': np.random.choice(TRAIN_OPERATORS, num_samples),
        'class': np.random.choice(['standard', 'first'], num_samples, p=[0.8, 0.2]),
        'user_type': np.random.choice(USER_TYPES, num_samples),
        'is_peak': np.random.choice([0, 1], num_samples),
        'is_weekend': np.random.choice([0, 1], num_samples),
        'is_holiday': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
        'distance_miles': np.random.randint(5, 500, num_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Remove same origin-destination pairs
    mask = df['origin_station'] == df['destination_station']
    df.loc[mask, 'destination_station'] = df.loc[mask, 'destination_station'].apply(
        lambda x: random.choice([s for s in STATIONS if s != x])
    )
    
    # Generate base fare based on distance and other factors
    df['base_fare'] = (
        df['distance_miles'] * 0.2 +  # Distance factor
        df['is_peak'] * 15 +          # Peak time premium
        (df['class'] == 'first') * df['distance_miles'] * 0.3 +  # First class premium
        (90 - df['booking_days_ahead']) * 0.2 +  # Advance booking discount
        (df['is_weekend'] * -5) +     # Weekend discount
        (df['is_holiday'] * 10) +     # Holiday premium
        np.random.normal(0, 5, num_samples)  # Random noise
    )
    
    # Ensure base fare is positive
    df['base_fare'] = np.maximum(df['base_fare'], 5.0)
    
    # Apply user type discounts
    user_type_factors = {
        'standard': 1.0,
        'business': 1.1,
        'student': 0.7,
        'senior': 0.8,
        'family': 0.9
    }
    
    for user_type, factor in user_type_factors.items():
        df.loc[df['user_type'] == user_type, 'base_fare'] *= factor
    
    # Round fare to 2 decimal places
    df['base_fare'] = np.round(df['base_fare'], 2)
    
    # Add timestamp
    now = datetime.now()
    df['data_timestamp'] = now.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to CSV if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic train fare data')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/raw/train_fares.csv', help='Output CSV file path')
    parser.add_argument('--upload-s3', action='store_true', help='Upload to S3 after generation')
    parser.add_argument('--s3-bucket', type=str, help='S3 bucket name')
    parser.add_argument('--s3-prefix', type=str, default='raw/', help='S3 key prefix')
    
    args = parser.parse_args()
    
    # Generate data
    df = generate_data(args.samples, args.output)
    
    # Upload to S3 if requested
    if args.upload_s3 and args.s3_bucket:
        try:
            import boto3
            s3_client = boto3.client('s3')
            s3_path = f"{args.s3_prefix}train_fares_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            s3_client.upload_file(args.output, args.s3_bucket, s3_path)
            print(f"Data uploaded to s3://{args.s3_bucket}/{s3_path}")
        except ImportError:
            print("boto3 not installed. Skipping S3 upload.")
        except Exception as e:
            print(f"Error uploading to S3: {e}")
    
    print(f"Generated {len(df)} samples with {df.shape[1]} features")
    print(f"Sample data:\n{df.head()}")
    print(f"Data statistics:\n{df.describe()}")

if __name__ == "__main__":
    main()