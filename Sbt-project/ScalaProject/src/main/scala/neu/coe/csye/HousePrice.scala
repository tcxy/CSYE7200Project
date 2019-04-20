package neu.coe.csye

import neu.coe.csye.Utils.ModelUtil
import neu.coe.csye.Utils.EvaluateUtil


object HousePrice extends App {

  override def main(args: Array[String]): Unit = {
    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("HousePrice")
      .getOrCreate;

    val data = ModelUtil.loadFile(spark, "kc_house_data.csv")
    val Array(train, test) = data.randomSplit(Array(0.7, 0.3))

    val pipeline = ModelUtil.buildPipeline()
    val predictions = ModelUtil.fitAndTransform(pipeline, train, test)

    EvaluateUtil.evaluateModel(predictions)
  }
}