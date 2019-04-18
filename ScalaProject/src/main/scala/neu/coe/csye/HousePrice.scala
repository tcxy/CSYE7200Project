package neu.coe.csye

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{exp, log}

object HousePrice extends App {

  def buildPipeline() = {
    val assembler = new VectorAssembler().setInputCols(Array("bedrooms", "floors", "grade", "lat", "sqft_living", "view"))
      .setOutputCol("features")
    val rf = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("features").setNumTrees(40).setMaxBins(36).setMaxDepth(14)
    val pipeline = new Pipeline().setStages(Array(assembler, rf))
    pipeline
  }

  def evaluateRMS(prediction: DataFrame) = {
    val price = prediction.select(exp(prediction("prediction")).as("prediction"), exp(prediction("label")).as("label"))

    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
    evaluator.evaluate(price)
  }

  def evaluateRSquare(prediction: DataFrame) = {
    val price = prediction.select(exp(prediction("prediction")).as("prediction"), exp(prediction("label")).as("label"))

    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("r2")
    evaluator.evaluate(price)
  }

  override def main(args: Array[String]): Unit = {
    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("HousePrice")
      .getOrCreate;

    val df = spark.read
      .format("csv")
      .option("header", "true") //first line in file has headers
      .option("inferSchema", "true")
      .option("mode", "DROPMALFORMED")
      .load("../kc_house_data.csv")
    val data = df.select(log(df("price")).as("label") , df("bedrooms"), df("floors"), df("grade"), df("lat"), log(df("sqft_living")).as("sqft_living"), df("view"))

    val Array(train, test) = data.randomSplit(Array(0.7, 0.3))

    val pipeline = buildPipeline()
    val model = pipeline.fit(train)
    val predictions = model.transform(test)
    val rmse = evaluateRMS(predictions)
    val r2 = evaluateRSquare(predictions)

    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    println(s"R squared error on test data = $r2")
  }
}