name := "ScalaProject"

version := "0.1"

scalaVersion := "2.11.12"

val scalaTestVersion = "2.2.4"

libraryDependencies += "org.scalatest" %% "scalatest" % scalaTestVersion % "test"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.3"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.3.3"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.3"