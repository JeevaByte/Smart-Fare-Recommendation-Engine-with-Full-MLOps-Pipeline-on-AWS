#!/usr/bin/env python
"""
PySpark job to process train fare data.
Reads data from S3, performs feature engineering, and saves processed data to S3.
"""

import os
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, hour, dayofweek, when, lit, expr, 
    udf, to_timestamp, datediff, current_date
)
from pyspark.sql.types import DoubleType, StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

def create_spark_session(app_name="TrainFareProcessing"):
    """Create or get a Spark session"""
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.sql.parquet.compression.codec", "snappy")
            .getOrCreate())

def read_data(spark, input_path):
    """Read data from CSV file or S3 location"""
    return spark.read.csv(input_path, header=True, inferSchema=True)

def feature_engineering(df):
    """Perform feature engineering on the input DataFrame"""
    
    # Convert time_of_day to categorical time buckets
    df = df.withColumn(
        "time_bucket",
        when((col("time_of_day") >= 6) & (col("time_of_day") < 10), "morning_peak")
        .when((col("time_of_day") >= 10) & (col("time_of_day") < 16), "day_off_peak")
        .when((col("time_of_day") >= 16) & (col("time_of_day") < 20), "evening_peak")
        .otherwise("night_off_peak")
    )
    
    # Bucket booking days ahead
    df = df.withColumn(
        "booking_window",
        when(col("booking_days_ahead") < 7, "last_minute")
        .when((col("booking_days_ahead") >= 7) & (col("booking_days_ahead") < 14), "one_week")
        .when((col("booking_days_ahead") >= 14) & (col("booking_days_ahead") < 30), "two_weeks")
        .when((col("booking_days_ahead") >= 30) & (col("booking_days_ahead") < 60), "one_month")
        .otherwise("advance")
    )
    
    # Bucket travel time
    df = df.withColumn(
        "journey_length",
        when(col("travel_time_minutes") < 60, "short")
        .when((col("travel_time_minutes") >= 60) & (col("travel_time_minutes") < 120), "medium")
        .when((col("travel_time_minutes") >= 120) & (col("travel_time_minutes") < 240), "long")
        .otherwise("very_long")
    )
    
    # Create categorical features for popular routes
    df = df.withColumn(
        "route",
        expr("concat(origin_station, '-', destination_station)")
    )
    
    return df

def prepare_ml_features(df):
    """Prepare features for machine learning"""
    
    # Define categorical columns for encoding
    categorical_cols = [
        "origin_station", "destination_station", "train_operator", 
        "class", "user_type", "time_bucket", "booking_window", 
        "journey_length"
    ]
    
    # Define numerical columns
    numerical_cols = [
        "booking_days_ahead", "travel_time_minutes", "time_of_day",
        "day_of_week", "is_peak", "is_weekend", "is_holiday", "distance_miles"
    ]
    
    # Create pipeline stages
    stages = []
    
    # String indexing for categorical features
    for col_name in categorical_cols:
        indexer = StringIndexer(
            inputCol=col_name,
            outputCol=f"{col_name}_idx",
            handleInvalid="keep"
        )
        encoder = OneHotEncoder(
            inputCols=[f"{col_name}_idx"],
            outputCols=[f"{col_name}_enc"]
        )
        stages += [indexer, encoder]
    
    # Assemble all features into a single vector
    assembler_inputs = [f"{col}_enc" for col in categorical_cols] + numerical_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="keep"
    )
    stages += [assembler]
    
    # Create and apply the pipeline
    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(df)
    df_transformed = pipeline_model.transform(df)
    
    # Select relevant columns for ML
    ml_df = df_transformed.select(
        "features", 
        col("base_fare").alias("label"),
        *[col(c) for c in df.columns]  # Keep original columns
    )
    
    return ml_df, pipeline_model

def process_data(input_path, output_path, save_pipeline_path=None):
    """Main data processing function"""
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Read data
        print(f"Reading data from {input_path}")
        df = read_data(spark, input_path)
        
        # Print schema and sample data
        print("Original schema:")
        df.printSchema()
        print(f"Original data count: {df.count()}")
        
        # Perform feature engineering
        print("Performing feature engineering...")
        df_engineered = feature_engineering(df)
        
        # Prepare ML features
        print("Preparing ML features...")
        ml_df, pipeline_model = prepare_ml_features(df_engineered)
        
        # Save processed data
        print(f"Saving processed data to {output_path}")
        ml_df.write.mode("overwrite").parquet(output_path)
        
        # Save pipeline model if path is provided
        if save_pipeline_path:
            print(f"Saving pipeline model to {save_pipeline_path}")
            pipeline_model.write().overwrite().save(save_pipeline_path)
        
        print("Data processing completed successfully")
        
    finally:
        # Stop Spark session
        spark.stop()

def main():
    parser = argparse.ArgumentParser(description='Process train fare data with PySpark')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input path (local or s3://bucket/path)')
    parser.add_argument('--output', type=str, required=True, 
                        help='Output path (local or s3://bucket/path)')
    parser.add_argument('--save-pipeline', type=str, 
                        help='Path to save the pipeline model')
    
    args = parser.parse_args()
    
    process_data(args.input, args.output, args.save_pipeline)

if __name__ == "__main__":
    main()