package neu.coe.csye.Utils

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.exp

object EvaluateUtil {
  def evaluateModel(predictions: DataFrame) = {
    val rmse = evaluateRMS(predictions)
    val r2 = evaluateRSquare(predictions)

    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    println(s"R squared error on test data = $r2")
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
}