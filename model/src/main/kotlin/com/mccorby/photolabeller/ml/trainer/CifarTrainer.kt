package com.mccorby.photolabeller.ml.trainer

import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File


class CifarTrainer(private val config: SharedConfig) {
    //    fun createModel(seed: Int, iterations: Int, numLabels: Int): MultiLayerNetwork? {
//
//
////        val fullModel: String = ClassPathResource("D:\\workspace\\phModel\\flower_model.h5").getFile().getPath()
////        val model: ComputationGraph = KerasModelImport.importKerasModelAndWeights("D:\\workspace\\phModel\\dog_cat_TK_2.h5")
////        val model: MultiLayerNetwork = KerasModelImport.importKerasSequentialModelAndWeights("D:\\workspace\\phModel\\dog_cat_TK_2.h5")
//        //        ComputationGraph graph2 = KerasModelImport.importKerasModelAndWeights("D:\\workspace\\phModel\\cat_dog_classifier.h5");
////        val graph3 = KerasModelImport.importKerasSequentialModelAndWeights("D:\\workspace\\phModel\\cat_dog_classifier.h5")//TODO OK
//
//        // TODO save the h5 to dl4j format and try run on android
//        return null
//    }
    fun createModel(seed: Int, iterations: Int, numLabels: Int): MultiLayerNetwork {
//        val fullModel: String = ClassPathResource("D:\\workspace\\phModel\\flower_model.h5").getFile().getPath()
//        val model: ComputationGraph = KerasModelImport.importKerasModelAndWeights("D:\\workspace\\phModel\\dog_cat_TK_2.h5")
//        val model: MultiLayerNetwork = KerasModelImport.importKerasSequentialModelAndWeights("D:\\workspace\\phModel\\dog_cat_TK_2.h5")
        //        ComputationGraph graph2 = KerasModelImport.importKerasModelAndWeights("D:\\workspace\\phModel\\cat_dog_classifier.h5");
//        val graph3 = KerasModelImport.importKerasSequentialModelAndWeights("D:\\workspace\\phModel\\cat_dog_classifier.h5")

        val modelConf = NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(Updater.ADAM)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l1(1e-4)
                .regularization(true)
                .l2(5 * 1e-4)
                .list()
                .layer(0, ConvolutionLayer.Builder(intArrayOf(4, 4), intArrayOf(1, 1), intArrayOf(0, 0))
                        .name("cnn1")
                        .convolutionMode(ConvolutionMode.Same)
                        .nIn(3)
                        .nOut(32)
                        .weightInit(WeightInit.XAVIER_UNIFORM)
                        .activation(Activation.RELU)
                        .learningRate(1e-2)
                        .biasInit(1e-2)
                        .biasLearningRate(1e-2 * 2)
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
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(config.imageSize, config.imageSize, config.channels)) //TODO
                .build()

        return MultiLayerNetwork(modelConf).also { it.init() }
    }

    internal class CnnModelProperties {
        private val epochsNum = 512
        private val learningRate = 0.001
        private val optimizer = Updater.ADAM
    }

    fun createModel2(seed: Int, iterations: Int, numLabels: Int): MultiLayerNetwork {
        val inputType = InputType.convolutional(32, 32, 3)
        val modelConf = NeuralNetConfiguration.Builder()
                .seed(123L)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.001)
                .regularization(true)
                .updater(Updater.ADAM)
                .list()
                .layer(0, conv5x5())
                .layer(1, pooling2x2Stride2())
                .layer(2, conv3x3Stride1Padding2())
                .layer(3, pooling2x2Stride1())
                .layer(4, conv3x3Stride1Padding1())
                .layer(5, pooling2x2Stride1())
                .layer(6, dense())
                .pretrain(false)
                .backprop(true)
                .setInputType(inputType)
                .build();

        return MultiLayerNetwork(modelConf).also { it.init() }
    }
    private fun conv5x5(): ConvolutionLayer? {
        return ConvolutionLayer.Builder(5, 5)
                .nIn(3)
                .nOut(16)
                .stride(1, 1)
                .padding(1, 1)
                .weightInit(WeightInit.XAVIER_UNIFORM)
                .activation(Activation.RELU)
                .build()
    }
    private fun pooling2x2Stride2(): SubsamplingLayer? {
        return SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build()
    }
    private fun conv3x3Stride1Padding2(): ConvolutionLayer? {
        return ConvolutionLayer.Builder(3, 3)
                .nOut(32)
                .stride(1, 1)
                .padding(2, 2)
                .weightInit(WeightInit.XAVIER_UNIFORM)
                .activation(Activation.RELU)
                .build()
    }
    private fun pooling2x2Stride1(): SubsamplingLayer? {
        return SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(1, 1)
                .build()
    }
    private fun conv3x3Stride1Padding1(): ConvolutionLayer? {
        return ConvolutionLayer.Builder(3, 3)
                .nOut(64)
                .stride(1, 1)
                .padding(1, 1)
                .weightInit(WeightInit.XAVIER_UNIFORM)
                .activation(Activation.RELU)
                .build()
    }
    private fun dense(): OutputLayer? {
        return  OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER_UNIFORM)
//                .nOut(dataSetService.labels().size() - 1)
//                .nOut(CifarLoader.NUM_LABELS - 1)
                .nOut(CifarLoader.NUM_LABELS)
                .build()
    }



    fun train(model: MultiLayerNetwork, numSamples: Int, epochs: Int, scoreListener: IterationListener): MultiLayerNetwork {
        model.setListeners(scoreListener)
//        val statsStorage: StatsStorage = FileStatsStorage(File(System.getProperty("D:\\workspace\\phModel\\"), "ui-stats.dl4j"))
//        val uiServer = UIServer.getInstance()
//        uiServer.attach(statsStorage)
//        model.setListeners(scoreListener, StatsListener(statsStorage), ScoreIterationListener(50))

//        val cifar = CifarDataSetIterator(config.batchSize, numSamples,
//                intArrayOf(config.imageSize, config.imageSize, config.channels),
//                CifarLoader.NUM_LABELS,
//                null,
//                false,
//                true)

        val trainIterator = CifarDataSetIterator(16, 512, true) //TODO
        val testIterator = CifarDataSetIterator(8, 128, false)

        for (i in 0 until epochs) {
            println("Epoch=====================$i")
//            model.fit(cifar) // TODO
            model.fit(trainIterator) // TODO
        }

        //evaluate
        model.evaluate(testIterator)

//        val testIter = CifarDataSetIterator(config.batchSize, numSamples,
//                intArrayOf(config.imageSize, config.imageSize, config.channels),
//                CifarLoader.NUM_LABELS,
//                null,
//                false,
//                false)
//
////        val rrTest:RecordReader = CSVRecordReader()
////        val testIter:DataSetIterator = RecordReaderDataSetIterator(rrTest,config.batchSize,0,10)
//        val eval = Evaluation(10)
//        while (testIter.hasNext()) {
//            val t: DataSet = testIter.next()
//            val features: INDArray = t.getFeatures()
//            val labels: INDArray = t.getLabels()
//            val predicted = model.output(features, false)
//            eval.eval(labels, predicted)
//        }

        return model
    }

    fun eval(model: MultiLayerNetwork, numSamples: Int): Evaluation {
        val cifarEval = CifarDataSetIterator(config.batchSize, numSamples,
                intArrayOf(config.imageSize, config.imageSize, config.channels),
                CifarLoader.NUM_LABELS,
                null,
                false,
                false)

        println("=====eval model========")
        val eval = Evaluation(cifarEval.labels)
        while (cifarEval.hasNext()) {
            val testDS = cifarEval.next(config.batchSize) // FIXME: Exception in thread "main" java.lang.IllegalArgumentException: bound must be positive
            val output = model.output(testDS.featureMatrix)
            eval.eval(testDS.labels, output)
        }
        return eval
    }

    fun saveModel(model: MultiLayerNetwork, location: String) {
        ModelSerializer.writeModel(model, File(location), true)
    }
}