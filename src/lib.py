from pyspark.sql import SparkSession


def create_spark_session(app_name: str = "MySparkApp") -> SparkSession:
    """
    Creates a SparkSession for local use.

    Args:
        app_name (str): The name of the Spark application.

    Returns:
        SparkSession: An instance of SparkSession.
    """
    spark = SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()

    return spark


if __name__ == "__main__":

    spark = create_spark_session()
    data = [("John", "Doe", 30), ("Jane", "Doe", 25), ("Mike", "Smith", 40)]
    columns = ["First Name", "Last Name", "Age"]

    df = spark.createDataFrame(data, columns)
    print("This is a test pySpark DataFrame!")
    df.show()
    spark.stop()
