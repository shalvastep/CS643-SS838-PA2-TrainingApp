package com.cs643.project2;

import org.apache.spark.SparkConf;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class Main {

    public static void main(String[] args) throws IOException {

        System.out.println("Starting in Trainer application");

        Properties props = new Properties();

        try (InputStream input = Main.class.getClassLoader().getResourceAsStream("config.properties")) {
            if (input == null) {
                throw new FileNotFoundException("config.properties not found in classpath");
            }
            props.load(input);
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Creating spark session");

        System.out.println(props.getProperty("spark.hadoop.fs.s3a.access.key"));

        SparkSession spark = SparkSession.builder()
                .appName("TrainingDataApplication")
                .master(props.getProperty("spark.master"))
                //.master("local[*]")  // I used thi for lcoal development
                .config("spark.hadoop.fs.s3a.access.key", props.getProperty("spark.hadoop.fs.s3a.access.key"))
                .config("spark.hadoop.fs.s3a.secret.key", props.getProperty("spark.hadoop.fs.s3a.secret.key"))
                .config("spark.hadoop.fs.s3a.endpoint", props.getProperty("spark.hadoop.fs.s3a.endpoint"))
                .config("spark.hadoop.fs.s3a.path.style.access", props.getProperty("spark.hadoop.fs.s3a.path.style.access"))
                .getOrCreate();

        // Load training data
        System.out.println("Reading Data...");
        Dataset<Row> trainingData = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .option("inferSchema", "true")
                .option("delimiter", ";")
                .load(props.getProperty("s3.training.data"));


        System.out.println("Data Total Row count: " + trainingData.count());

        long rowCount = trainingData.count();
        if (rowCount == 0) {
            System.out.println("DataFrame is empty. No data to process.");
        } else {
            System.out.println("DataFrame loaded with " + rowCount + " rows.");
        }

        System.out.println("5 rows of data");
        trainingData.printSchema(5);

        System.out.println("Number of rows: " + trainingData.count());
        String schemaString = trainingData.schema().treeString();
        System.out.println(schemaString);


        String[] columns = {
                "fixed acidity",
                "volatile acidity",
                "citric acid",
                "residual sugar",
                "chlorides",
                "free sulfur dioxide",
                "total sulfur dioxide",
                "density",
                "pH",
                "sulphates",
                "alcohol",
                "quality"
        };


        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(columns)
                .setOutputCol("features");

        Dataset<Row> transformedData = vectorAssembler.transform(trainingData);

        // This is good to see printed table
        transformedData.show(10);


        // Logistic regression model
        LogisticRegression logisticRegression = new LogisticRegression()
                // Specifies the column in the dataset that contains the feature vectors.
                .setFeaturesCol("features")
                .setLabelCol("quality");


        // Train
        LogisticRegressionModel logisticRegressionModel = logisticRegression.fit(transformedData);


        // Ensure only the driver writes the model to S3
        SparkConf conf = spark.sparkContext().getConf();
        String executorId = conf.get("spark.executor.id", "not_set");

        if ("driver".equals(executorId)) {
            System.out.println("This is the driver.");
            try {
                // Only the driver writes the model to S3
                logisticRegressionModel.write().overwrite().save(props.getProperty("s3.model.destination"));
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("This is a worker.");
        }


        // Stop Spark session
        System.out.println("Closing Spark Session");
        spark.stop();
    }
}