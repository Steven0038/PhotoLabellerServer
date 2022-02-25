import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * MIT License
 *
 * Copyright (c) 2019 Packt
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * image prediction API
 * <br></br>
 * ref. https://github.com/rahul-raj/Java-Deep-Learning-Cookbook
 */
public class ImageClassifierAPI {
    public static INDArray generateOutput(File inputFile, String modelFileLocation) throws IOException, InterruptedException {
        //retrieve the saved model
        final File modelFile = new File(modelFileLocation);
        final MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        final RecordReader imageRecordReader = generateReader(inputFile);
        final ImagePreProcessingScaler normalizerStandardize = ModelSerializer.restoreNormalizerFromFile(modelFile);
        final DataSetIterator dataSetIterator = new RecordReaderDataSetIterator.Builder(imageRecordReader, 1).build();
        normalizerStandardize.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(normalizerStandardize);
        return model.output(dataSetIterator);
    }

    private static RecordReader generateReader(File file) throws IOException, InterruptedException {
        final RecordReader recordReader = new ImageRecordReader(30, 30, 3);
        final InputSplit inputSplit = new FileSplit(file);
        recordReader.initialize(inputSplit);
        return recordReader;
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        final List<String> results = new ArrayList<>();
        String modelFilePath = "E:\\workspace\\phModelNew\\experiment_results\\weather_federated_beta3-1645674600781-DONE.zip";
        String predictImgPath = "E:\\dataset\\MultiClassWeatherDataset\\Shine\\shine30.jpg";

        final File file = new File(predictImgPath);
        final INDArray indArray = generateOutput(file, modelFilePath);
        System.out.println(indArray);

        DecimalFormat df2 = new DecimalFormat("#.####");
        for (int i = 0; i < indArray.rows(); i++) {
            StringBuilder result = new StringBuilder("Image " + i + "->>>>>");
            for (int j = 0; j < indArray.columns(); j++) {
                result.append("\n Category ").append(j).append(": ").append(df2.format(indArray.getDouble(i, j) * 100)).append("%,   ");
            }
            result.append("\n\n");
            results.add(result.toString());
        }

        System.out.println(results);
    }
}


