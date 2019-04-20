import neu.coe.csye.Utils.{EvaluateUtil, ModelUtil}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.log
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class HousePriceSpec extends FlatSpec with Matchers with BeforeAndAfter {
  implicit var spark: SparkSession = _
  before {
    spark = org.apache.spark.sql.SparkSession.builder
      .master("local[*]")
      .appName("HousePrice")
      .getOrCreate()
  }

  after {
    if (spark != null) {
      spark.stop()
    }
  }

  behavior of "Training"

  "The dataset" should "not be null" in {
    val data = ModelUtil.loadFile(spark, "kc_house_data.csv")
    assert(data.count() == 21613)
  }

  "The train data" should "not be null" in {
    val data = ModelUtil.loadFile(spark, "kc_house_data.csv")
    val train = ModelUtil.trainTestSplit(data)(0)
    assert(train.count() > 0)
  }

  "The test data" should "not be null" in {
    val data = ModelUtil.loadFile(spark, "kc_house_data.csv")
    val test = ModelUtil.trainTestSplit(data)(1)
    assert(test.count() > 0)
  }

  "Execution time" should "with in 3 seconds" in {
    val t0 = System.currentTimeMillis()
    val data = ModelUtil.loadFile(spark, "kc_house_data.csv")

    val Array(train, test) = data.randomSplit(Array(0.6, 0.4))

    val pipeline = ModelUtil.buildPipeline()
    val model = pipeline.fit(train)
    val t1 = System.currentTimeMillis()
    assert((t1-t0) <= 3000)
  }

  "RMS" should "less or equal than 180000" in {
    val df = spark.read
      .format("csv")
      .option("header", "true") //first line in file has headers
      .option("inferSchema", "true")
      .option("mode", "DROPMALFORMED")
      .load("kc_house_data.csv")
    val data = df.select(log(df("price")).as("label") , df("bedrooms"), df("floors"), df("grade"), df("lat"), log(df("sqft_living")).as("sqft_living"), df("view"))

    val Array(train, test) = data.randomSplit(Array(0.6, 0.4))

    val pipeline = ModelUtil.buildPipeline()
    val model = pipeline.fit(train)
    val predictions = model.transform(test)

    assert(EvaluateUtil.evaluateRMS(predictions)<= 180000)
  }

  "R-Sqaured" should "larger than 0.7" in {
    val df = spark.read
      .format("csv")
      .option("header", "true") //first line in file has headers
      .option("inferSchema", "true")
      .option("mode", "DROPMALFORMED")
      .load("kc_house_data.csv")
    val data = df.select(log(df("price")).as("label") , df("bedrooms"), df("floors"), df("grade"), df("lat"), log(df("sqft_living")).as("sqft_living"), df("view"))

    val Array(train, test) = data.randomSplit(Array(0.6, 0.4))

    val pipeline = ModelUtil.buildPipeline()
    val model = pipeline.fit(train)
    val predictions = model.transform(test)
    assert(EvaluateUtil.evaluateRSquare(predictions) >= 0.7)
  }
}