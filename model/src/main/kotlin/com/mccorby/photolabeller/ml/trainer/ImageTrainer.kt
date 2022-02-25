package com.mccorby.photolabeller.ml.trainer

import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.util.*


class ImageTrainer(private val config: SharedConfig) {

    fun createModel(seed: Int, iterations: Int, numLabels: Int): MultiLayerNetwork {
        val modelConf = NeuralNetConfiguration.Builder()
                .seed(seed.toLong())
                .updater(Adam())
//                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l1(1e-4)
//                .regularization(true)
                .l2(5 * 1e-4)
                .list()
                .layer(0, ConvolutionLayer.Builder(intArrayOf(4, 4), intArrayOf(1, 1), intArrayOf(0, 0))
                        .name("cnn1")
                        .convolutionMode(ConvolutionMode.Same)
                        .nIn(3)
                        .nOut(32)
                        .weightInit(WeightInit.XAVIER_UNIFORM)
                        .activation(Activation.RELU)
//                        .learningRate(1e-2)
                        .biasInit(1e-2)
//                        .biasLearningRate(1e-2 * 2)
                        .build())
                .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, intArrayOf(3, 3))
                        .name("pool1")
                        .build())
                .layer(2, LocalResponseNormalization.Builder(3.0, 5e-05, 0.75)
                        .build())
                .layer(3, DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(64)
                        .dropOut(0.5)
                        .build())
                .layer(4, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(config.imageSize.toLong(), config.imageSize.toLong(),
                    config.channels.toLong()
                ))
                .build()

        return MultiLayerNetwork(modelConf)
            .also { it.init() }
    }

    fun train(model: MultiLayerNetwork, epochs:Int, batchSize: Int = 10, fileDir: String): MultiLayerNetwork {
        //R,G,B channels
        val channels = 3

        //load files and split
        val parentDir = File(fileDir)
        val fileSplit = FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS, Random(42))
        val numLabels = fileSplit.rootDir.listFiles { obj: File -> obj.isDirectory }.size

        //identify labels in the path
        val parentPathLabelGenerator = ParentPathLabelGenerator()

        //file split to train/test using the weights.
        val balancedPathFilter =
            BalancedPathFilter(Random(42), NativeImageLoader.ALLOWED_FORMATS, parentPathLabelGenerator)
        val inputSplits = fileSplit.sample(balancedPathFilter, 80.0, 20.0)

        //get train/test data
        val trainData = inputSplits[0]
        val testData = inputSplits[1]

        //Data augmentation
//        val randNumGen = Random(12345)
//        val transform: ImageTransform = MultiImageTransform(
//            randNumGen,
//            FlipImageTransform(Random(42)),
//            FlipImageTransform(Random(123)),
//            WarpImageTransform(Random(42), 42F),
//            RotateImageTransform(Random(42), 40F)
//        )

        val scaler: DataNormalization = ImagePreProcessingScaler(0.0, 1.0)

        //train without transformations
        val imageRecordReader = ImageRecordReader(config.imageSize.toLong(), config.imageSize.toLong(), channels.toLong(), parentPathLabelGenerator)
        imageRecordReader.initialize(trainData, null)
        val dataSetIterator: DataSetIterator = RecordReaderDataSetIterator(imageRecordReader, batchSize, 1, numLabels)
        scaler.fit(dataSetIterator)
        dataSetIterator.preProcessor = scaler

        model.setListeners(ScoreIterationListener(100)) //PerformanceListener for optimized training

        for (i in 0 until epochs) {
            println("Epoch=====================$i")
            model.fit(dataSetIterator)
        }

        //train with transformations
//        imageRecordReader.initialize(trainData, transform)
//        val dataSetIterator2: DataSetIterator = RecordReaderDataSetIterator(imageRecordReader, batchSize, 1, numLabels)
//        scaler.fit(dataSetIterator2)
//        dataSetIterator.preProcessor = scaler
//        for (i in 0 until epochs) {
//            println("Epoch=====================$i")
//            model.fit(dataSetIterator)
//        }

        // evaluation of model
        imageRecordReader.initialize(testData)
        val evaluation: org.nd4j.evaluation.classification.Evaluation = model.evaluate(dataSetIterator)
        println("args = [" + evaluation.stats().toString() + "]")

        return model
    }

    fun saveModel(model: MultiLayerNetwork, location: String) {
        ModelSerializer.writeModel(model, File(location), true)
    }

    fun eval(model: MultiLayerNetwork, batchSize: Int, trainFileDir: String){
        //R,G,B channels
        val channels = 3

        //load files and split
        val parentDir = File(trainFileDir)
        val fileSplit = FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS, Random(42))
        val numLabels = fileSplit.rootDir.listFiles { obj: File -> obj.isDirectory }.size

        //identify labels in the path
        val parentPathLabelGenerator = ParentPathLabelGenerator()

        //file split to train/test using the weights.
        val balancedPathFilter =
            BalancedPathFilter(Random(42), NativeImageLoader.ALLOWED_FORMATS, parentPathLabelGenerator)
        val inputSplits = fileSplit.sample(balancedPathFilter, 80.0, 20.0)

        //get train/test data
        val trainData = inputSplits[0]
        val testData = inputSplits[1]

        val scaler: DataNormalization = ImagePreProcessingScaler(0.0, 1.0)

        //train without transformations
        val imageRecordReader = ImageRecordReader(config.imageSize.toLong(), config.imageSize.toLong(), channels.toLong(), parentPathLabelGenerator)
        imageRecordReader.initialize(trainData, null)
        val dataSetIterator: DataSetIterator = RecordReaderDataSetIterator(imageRecordReader, batchSize, 1, numLabels)
        scaler.fit(dataSetIterator)
        dataSetIterator.preProcessor = scaler

        model.setListeners(ScoreIterationListener(100)) //PerformanceListener for optimized training

        // evaluation of model
        imageRecordReader.initialize(testData)
        val evaluation: org.nd4j.evaluation.classification.Evaluation = model.evaluate(dataSetIterator)
        println("args = [" + evaluation.stats().toString() + "]")
    }
}