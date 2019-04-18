import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.Pipeline

val spark = org.apache.spark.sql.SparkSession.builder
        .master("local")
        .appName("HousePrice")
        .getOrCreate;

val df = spark.read 
        .format("csv")
        .option("header", "true") //first line in file has headers
        .option("inferSchema", "true")         
        .option("mode", "DROPMALFORMED")         
        .load("../../kc_house_data.csv")
// Select data and take log scale of price and sqft_living
val data = df.select(log(df("price")).as("label") , df("bedrooms"), df("floors"), df("grade"), df("lat"), log(df("sqft_living")).as("sqft_living"), df("view"))

// Split data into training and testing set
val Array(training, test) = data.randomSplit(Array(0.7, 0.3))

def buildPipeline() = {
    val assembler = new VectorAssembler().setInputCols(Array("bedrooms", "floors", "grade", "lat", "sqft_living", "view"))
                        .setOutputCol("features")    
    val rf = new RandomForestRegressor().setLabelCol(\"label\").setFeaturesCol(\"features\").setNumTrees(40).setMaxBins(36).setMaxDepth(14)      
    val pipeline = new Pipeline().setStages(Array(assembler, gb))
    pipeline
}

val t0 = System.nanoTime()

val model = buildPipeline().fit(training)
val predictions = model.transform(test)
val originalPrice = predictions.select(exp($"prediction").as("prediction"), exp($"label").as("label"), $"features")
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(originalPrice)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
val r2evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("r2")
val r2 = r2evaluator.evaluate(originalPrice)
println(s"R squared error on test data = $r2")

val t1 = System.nanoTime()

print("Execution time: " + (t1 - t0) + " ns")