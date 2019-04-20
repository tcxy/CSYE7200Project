package neu.coe.csye.Utils

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql.functions.log
import org.apache.spark.sql.{DataFrame, SparkSession}

object ModelUtil {
  def loadFile(spark: SparkSession, filePath: String) = {
    val df = spark.read
      .format("csv")
      .option("header", "true") //first line in file has headers
      .option("inferSchema", "true")
      .option("mode", "DROPMALFORMED")
      .load(filePath)
    dataIngest(df)
  }

  def trainTestSplit(df: DataFrame) = df.randomSplit(Array(0.6,0.4))

  def dataIngest(df: DataFrame) = df.select(log(df("price")).as("label") , df("bedrooms"), df("floors"), df("grade"), df("lat"), log(df("sqft_living")).as("sqft_living"), df("view"))

  def buildPipeline() = {
    val assembler = new VectorAssembler().setInputCols(Array("bedrooms", "floors", "grade", "lat", "sqft_living", "view"))
      .setOutputCol("features")
    val rf = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("features").setNumTrees(40).setMaxBins(36).setMaxDepth(14)
    val pipeline = new Pipeline().setStages(Array(assembler, rf))
    pipeline
  }

  def fitAndTransform(pipeline: Pipeline, trainData:DataFrame, testData: DataFrame) = {
    val model = pipeline.fit(trainData)
    model.transform(testData)
  }
}
