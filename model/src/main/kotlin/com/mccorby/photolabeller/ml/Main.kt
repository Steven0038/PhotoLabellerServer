package com.mccorby.photolabeller.ml

import com.mccorby.photolabeller.ml.trainer.CifarTrainer
import com.mccorby.photolabeller.ml.trainer.SharedConfig
import org.bytedeco.javacpp.opencv_core
import org.datavec.image.loader.CifarLoader
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
    val argumentOne = "train"
    val argumentTwo = "D:/workspace/phModel"
    val argumentThree = "web"

//    if (args.isNotEmpty() && args[0] == "train") {
    if (argumentOne == "train") {
        val seed = 123
        val iterations = 1
        val numLabels = CifarLoader.NUM_LABELS
        val saveFile = "cifar_federated-${Date().time}.zip"

//        val numEpochs = 50
        val numEpochs = 512
//        val numEpochs = 1 // TODO
        val numSamples = 10000

        val config = SharedConfig(32, 3, 100)
        val trainer = CifarTrainer(config)
//        var model = trainer.createModel(seed, iterations, numLabels)
        var model = trainer.createModel2(seed, iterations, numLabels) //TODO
//        model = trainer.train(model, numSamples, numEpochs, getVisualization(args.getOrNull(2)))
        model = trainer.train(model, numSamples, numEpochs, getVisualization(argumentThree))
//
////        if (args[1].isNotEmpty()) {
        if (argumentTwo.isNotEmpty()) {
//            println("Saving model to ${args[1]}")
            println("Saving model to $argumentTwo")
//            trainer.saveModel(model, args[1] + "/$saveFile")
            trainer.saveModel(model, argumentTwo + "/$saveFile")
        }

//        val eval = trainer.eval(model, numSamples)//FIXME: error
//        println(eval.stats())

    } else {
//        predict(args[0], args[1])
        predict(argumentOne, argumentTwo)
    }


    predict("D:/workspace/phModel/cifar_federated-1618113579660.zip", "D:/workspace/phModel/cat.jpg") //TODO
//    predict("D:/workspace/phModel/model.zip", "D:/workspace/phModel/cat.jpg") //TODO
}

fun predict(modelFile: String, imageFile: String) {
    val config = SharedConfig(32, 3, 100)
    val trainer = CifarTrainer(config)

    val model = ModelSerializer.restoreMultiLayerNetwork(modelFile)

    val eval = trainer.eval(model, 100)
    println(eval.stats())

    val file = File(imageFile)
    val resizedImage = opencv_core.Mat()
    val sz = opencv_core.Size(32, 32)
    val opencvImage = org.bytedeco.javacpp.opencv_imgcodecs.imread(file.absolutePath)
    org.bytedeco.javacpp.opencv_imgproc.resize(opencvImage, resizedImage, sz)

    val nativeImageLoader = NativeImageLoader()
    val image = nativeImageLoader.asMatrix(resizedImage)
    val reshapedImage = image.reshape(1, 3, 32, 32)
    val result = model.predict(reshapedImage)
    println(result.joinToString(", ", prefix = "[", postfix = "]"))
}

private fun getVisualization(visualization: String?): IterationListener {
    return when (visualization) {
        "web" -> {
            //Initialize the user interface backend
            val uiServer = UIServer.getInstance()

            //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
            val statsStorage = InMemoryStatsStorage()         //Alternative: new FileStatsStorage(File), for saving and loading later

            //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
            uiServer.attach(statsStorage)

            //Then add the StatsListener to collect this information from the network, as it trains
            StatsListener(statsStorage)
        }
        else -> ScoreIterationListener(50)
    }
}