#!/usr/bin/env python3

import atexit
from pathlib import Path
from sys import exit, stderr

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.regression import (
    DecisionTreeRegressor,  # GBTRegressor,
    LinearRegression,
    RandomForestRegressor,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import FloatType, IntegerType, StructField, StructType

from lib import create_spark_session # type: ignore

boston_housing_file = Path(".") / "data" / "HousingData.csv"
if not boston_housing_file.exists():
    print(f"DataFile {boston_housing_file} doesnt exist", file=stderr)
    exit(1)

boston_housing_schema = StructType(
    [
        StructField("CRIM", FloatType(), True),
        StructField("ZN", FloatType(), True),
        StructField("INDUS", FloatType(), True),
        StructField("CHAS", IntegerType(), True),
        StructField("NOX", FloatType(), True),
        StructField("RM", FloatType(), True),
        StructField("AGE", FloatType(), True),
        StructField("DIS", FloatType(), True),
        StructField("RAD", IntegerType(), True),
        StructField("TAX", FloatType(), True),
        StructField("PTRATIO", FloatType(), True),
        StructField("B", FloatType(), True),
        StructField("LSTAT", FloatType(), True),
        StructField("MEDV", FloatType(), True),  # -> Target variable
    ]
)

feature_cols = [
    "CRIM",  # per capita crime rate by town
    "ZN",  # proportion of residential land zoned for lots over 25,000 sq. ft.
    "INDUS",  # proportion of non-retail business acres per town
    "CHAS",  # Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    "NOX",  # nitric oxides concentration (parts per 10 million)
    "RM",  # average number of rooms per dwelling
    "AGE",  # proportion of owner-occupied units built prior to 1940
    "DIS",  # weighted distances to five Boston employment centers
    "RAD",  # index of accessibility to radial highways
    "TAX",  # full-value property tax rate per $10,000
    "PTRATIO",  # pupil-teacher ratio by town
    "B",  # 1000(Bk - 0.63)^2 where Bk is the proportion of black residents by town
    "LSTAT",  # percentage of lower status of the population
]


def main():

    spark = create_spark_session("PySpark-Regression")
    atexit.register(lambda: spark.stop())

    df = spark.read.csv(
        str(boston_housing_file), schema=boston_housing_schema, header=True
    )

    # Impute missing values with the mean
    imputer = Imputer(
        inputCols=feature_cols, outputCols=[f"{col}_imp" for col in feature_cols]
    ).setStrategy("mean")

    df_imputed = imputer.fit(df).transform(df)

    # Create the feature vector
    vectorAssembler = VectorAssembler(
        inputCols=[f"{col}_imp" for col in feature_cols],
        outputCol="features",
        handleInvalid="error",
    )

    df_transformed = vectorAssembler.transform(df_imputed)

    final_df = df_transformed.select("features", "MEDV")

    # Split into train and test sets
    train_df, test_df = final_df.randomSplit([0.5, 0.5], seed=42)

    # Initialize models
    lr = LinearRegression(labelCol="MEDV")
    dt = DecisionTreeRegressor(labelCol="MEDV")
    rf = RandomForestRegressor(labelCol="MEDV")
    # gbt = GBTRegressor(labelCol="MEDV")

    # Hyperparameter tuning for each model
    paramGrid_lr = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1, 0.5])
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
        .build()
    )

    paramGrid_dt = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10, 20]).build()

    paramGrid_rf = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [20, 50, 100])
        .addGrid(rf.maxDepth, [5, 10, 20])
        .build()
    )

    # paramGrid_gbt = (
    #     ParamGridBuilder()
    #     .addGrid(gbt.maxDepth, [5, 10, 20])
    #     .addGrid(gbt.maxIter, [20, 50, 100])
    #     .build()
    # )

    # CrossValidator for each model
    crossval_lr = CrossValidator(
        estimator=lr,
        estimatorParamMaps=paramGrid_lr,
        evaluator=RegressionEvaluator(labelCol="MEDV", metricName="rmse"),
        numFolds=3,
    )

    crossval_dt = CrossValidator(
        estimator=dt,
        estimatorParamMaps=paramGrid_dt,
        evaluator=RegressionEvaluator(labelCol="MEDV", metricName="rmse"),
        numFolds=3,
    )

    crossval_rf = CrossValidator(
        estimator=rf,
        estimatorParamMaps=paramGrid_rf,
        evaluator=RegressionEvaluator(labelCol="MEDV", metricName="rmse"),
        numFolds=3,
    )

    # crossval_gbt = CrossValidator(
    #     estimator=gbt,
    #     estimatorParamMaps=paramGrid_gbt,
    #     evaluator=RegressionEvaluator(labelCol="MEDV", metricName="rmse"),
    #     numFolds=3,
    # )

    # Model training with hyperparameter tuning
    lr_model = crossval_lr.fit(train_df)
    dt_model = crossval_dt.fit(train_df)
    rf_model = crossval_rf.fit(train_df)
    # gbt_model = crossval_gbt.fit(train_df)

    # Make predictions on the test set
    lr_predictions = lr_model.transform(test_df)
    dt_predictions = dt_model.transform(test_df)
    rf_predictions = rf_model.transform(test_df)
    # gbt_predictions = gbt_model.transform(test_df)

    # Initialize evaluators for different metrics
    evaluator_rmse = RegressionEvaluator(labelCol="MEDV", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol="MEDV", metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol="MEDV", metricName="r2")

    # Evaluate the models
    lr_rmse = evaluator_rmse.evaluate(lr_predictions)
    lr_mae = evaluator_mae.evaluate(lr_predictions)
    lr_r2 = evaluator_r2.evaluate(lr_predictions)

    dt_rmse = evaluator_rmse.evaluate(dt_predictions)
    dt_mae = evaluator_mae.evaluate(dt_predictions)
    dt_r2 = evaluator_r2.evaluate(dt_predictions)

    rf_rmse = evaluator_rmse.evaluate(rf_predictions)
    rf_mae = evaluator_mae.evaluate(rf_predictions)
    rf_r2 = evaluator_r2.evaluate(rf_predictions)

    # gbt_rmse = evaluator_rmse.evaluate(gbt_predictions)
    # gbt_mae = evaluator_mae.evaluate(gbt_predictions)
    # gbt_r2 = evaluator_r2.evaluate(gbt_predictions)

    # Print the metrics for each model
    print("Linear Regression Metrics:")
    print(f"RMSE: {lr_rmse}")
    print(f"MAE: {lr_mae}")
    print(f"R²: {lr_r2}")

    print("\nDecision Tree Regressor Metrics:")
    print(f"RMSE: {dt_rmse}")
    print(f"MAE: {dt_mae}")
    print(f"R²: {dt_r2}")

    print("\nRandom Forest Regressor Metrics:")
    print(f"RMSE: {rf_rmse}")
    print(f"MAE: {rf_mae}")
    print(f"R²: {rf_r2}")

    # print("\nGradient-Boosted Tree Regressor Metrics:")
    # print(f"RMSE: {gbt_rmse}")
    # print(f"MAE: {gbt_mae}")
    # print(f"R²: {gbt_r2}")


if __name__ == "__main__":
    main()
