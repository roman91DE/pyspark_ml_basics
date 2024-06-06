#!/usr/bin/env python3

import atexit
from pathlib import Path
from sys import exit, stderr

from pyspark.ml.clustering import GaussianMixture, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, when
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from lib import create_spark_session  # type: ignore

iris_file = Path(".") / "data" / "Iris.csv"
if not iris_file.exists():
    print(f"DataFile {iris_file} doesn't exist", file=stderr)
    exit(1)

iris_schema = StructType(
    [
        StructField("Id", IntegerType(), False),
        StructField("SepalLengthCm", FloatType(), True),
        StructField("SepalWidthCm", FloatType(), True),
        StructField("PetalLengthCm", FloatType(), True),
        StructField("PetalWidthCm", FloatType(), True),
        StructField("Species", StringType(), True),  # -> Target variable
    ]
)

feature_cols = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]


def main():

    spark = create_spark_session("PySpark-Clustering")
    atexit.register(lambda: spark.stop())

    df = spark.read.csv(str(iris_file), schema=iris_schema, header=True).drop("Id")

    df.show()

    # Assemble features into a single vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    # Map species to integers
    species_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    df = df.withColumn(
        "SpeciesIndex",
        when(col("Species") == "Iris-setosa", 0)
        .when(col("Species") == "Iris-versicolor", 1)
        .when(col("Species") == "Iris-virginica", 2),
    )

    # Define KMeans model
    kmeans = KMeans(k=3, seed=1, featuresCol="features")
    kmeans_model = kmeans.fit(df)
    kmeans_predictions = kmeans_model.transform(df)

    # Evaluate KMeans clustering
    evaluator = ClusteringEvaluator()
    kmeans_silhouette = evaluator.evaluate(kmeans_predictions)
    print(f"KMeans Silhouette Score: {kmeans_silhouette}")

    # Define GaussianMixture model
    gmm = GaussianMixture(k=3, seed=1, featuresCol="features")
    gmm_model = gmm.fit(df)
    gmm_predictions = gmm_model.transform(df)

    # Evaluate GaussianMixture clustering
    gmm_silhouette = evaluator.evaluate(gmm_predictions)
    print(f"GaussianMixture Silhouette Score: {gmm_silhouette}")

    # Show clustered data
    kmeans_predictions.show()
    gmm_predictions.show()

    # Evaluate clustering results against actual species labels
    kmeans_evaluation = (
        kmeans_predictions.groupBy("SpeciesIndex", "prediction")
        .count()
        .orderBy("SpeciesIndex", "prediction")
    )
    gmm_evaluation = (
        gmm_predictions.groupBy("SpeciesIndex", "prediction")
        .count()
        .orderBy("SpeciesIndex", "prediction")
    )

    print("KMeans Clustering Results:")
    kmeans_evaluation.show()

    print("GaussianMixture Clustering Results:")
    gmm_evaluation.show()


if __name__ == "__main__":
    main()
