package com.mccorby.photolabeller.ml

import com.mccorby.photolabeller.ml.trainer.SharedConfig
import com.mccorby.photolabeller.ml.trainer.ImageTrainer
import org.bytedeco.javacpp.opencv_core
import org.datavec.image.loader.NativeImageLoader
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import java.io.File
import java.util.*


fun main(args: Array<String>) {
    if (args.isNotEmpty() && args[0] == "train") {
        val seed = 123
        val iterations = 1

        // Harvard Weather Image Recognition
//        val numLabels = 11
//        val numEpochs = 20 // 0.6255
//        val batchSize = 25
//        val saveFile = "hv_weather_federated-${Date().time}.zip"
//        val trainFileDir = "E:\\dataset\\WeatherImageRecognition" // https://www.kaggle.com/jehanbhathena/weather-dataset

        // multi weather
        val numLabels = 4
        val numEpochs = 5
        val batchSize = 10
        val saveFile = "weather_federated_beta3-${Date().time}.zip"
        val trainFileDir = "E:\\dataset\\MultiClassWeatherDataset" // https://www.kaggle.com/pratik2901/multiclass-weather-dataset

        // sp-weather
//        val numLabels = 5
//        val numEpochs = 5
//        val batchSize = 20
//        val saveFile = "sp_weather_federated-${Date().time}.zip"
//        val trainFileDir = "E:\\dataset\\SP-Weather" // https://github.com/ZebaKhanam91/SP-Weather

        // customize cifar
//        val numLabels = 10
//        val numEpochs = 50
//        val batchSize = 100
//        val saveFile = "cifar10_federated-${Date().time}.zip"
//        val trainFileDir = "E:\\dataset\\cifar10_dl4j.v1\\train" // https://www.cs.toronto.edu/~kriz/cifar.html

        // customized car body type
//        val numLabels = 2 // largeCar: Wagon,SUV,Minivan,Cap,Van; smallCar: Coupe,Sedan,Hatchback
//        val numEpochs = 4
//        val batchSize = 30
//        val saveFile = "car_body_federated-${Date().time}.zip"
//        val trainFileDir = "E:\\dataset\\StanfordCarBodyTypeData\\stanford_cars_type" // https://www.kaggle.com/mayurmahurkar/stanford-car-body-type-data

        // Vehicle Detection Image Set
//        val numLabels = 2 // vehicle, non-vehicle
//        val numEpochs = 4
//        val batchSize = 30
//        val saveFile = "vehicle_detection-${Date().time}.zip"
//        val trainFileDir = "E:\\dataset\\VehicleDetectionImageSet" // https://www.kaggle.com/brsdincer/vehicle-detection-image-set


        val config = SharedConfig(32, 3, 100)
        val trainer = ImageTrainer(config)
        var model = trainer.createModel(seed, iterations, numLabels)
//        model = trainer.train(model, numSamples, numEpochs, getVisualization(args.getOrNull(2)))
        model = trainer.train(model, numEpochs, batchSize, trainFileDir)

        if (args[1].isNotEmpty()) {
            println("Saving model to ${args[1]}")
            trainer.saveModel(model, args[1] + "/$saveFile")
        }

//        val eval = trainer.eval(model, numSamples)
//        println(eval.stats())

    } else {
//        predict(args[0], args[1])
    }
}

//fun predict(modelFile: String, imageFile: String) {
//    val config = SharedConfig(32, 3, 100)
//    val trainer = CifarTrainer(config)
//
//    val model = ModelSerializer.restoreMultiLayerNetwork(modelFile)
//
//    val eval = trainer.eval(model, 100)
//    println(eval.stats())
//
//    val file = File(imageFile)
//    val resizedImage = opencv_core.Mat()
//    val sz = opencv_core.Size(32, 32)
//    val opencvImage = org.bytedeco.javacpp.opencv_imgcodecs.imread(file.absolutePath)
//    org.bytedeco.javacpp.opencv_imgproc.resize(opencvImage, resizedImage, sz)
//
//    val nativeImageLoader = NativeImageLoader()
//    val image = nativeImageLoader.asMatrix(resizedImage)
//    val reshapedImage = image.reshape(1, 3, 32, 32)
//    val result = model.predict(reshapedImage)
//    println(result.joinToString(", ", prefix = "[", postfix = "]"))
//}

//private fun getVisualization(visualization: String?): IterationListener {
//    return when (visualization) {
//        "web" -> {
//            //Initialize the user interface backend
//            val uiServer = UIServer.getInstance()
//
//            //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
//            val statsStorage =
//                InMemoryStatsStorage()         //Alternative: new FileStatsStorage(File), for saving and loading later
//
//            //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
//            uiServer.attach(statsStorage)
//
//            //Then add the StatsListener to collect this information from the network, as it trains
//            StatsListener(statsStorage)
//        }
//        else -> ScoreIterationListener(50)
//    }
//}