package io.github.stlabunifi.deepgreen.dl4j.expt.imported.vgg16;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.nd4j.common.io.ClassPathResource;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import io.github.stlabunifi.deepgreen.dl4j.core.dataloader.FashionMNISTDataloader;
import io.github.stlabunifi.deepgreen.dl4j.core.model.builder.ModelRebuilder;
import io.github.stlabunifi.deepgreen.dl4j.python.handler.PythonCommandHandler;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

public class Vgg16TrainFashionExpt {

	public final static String vgg16_py_filepath = "/models/sequential_vgg16.py";
	public final static String vgg16_fashion_h5_filename = "sequential_vgg16_fashion.h5";

	public final static int rngSeed = 123; 		// random number seed for reproducibility
	public final static int batchSize = 128; 	// batch size for each epoch
	public final static int numClasses = 10; 	// number of output classes
	public final static int numEpochs = 30; 	// number of epochs to perform
	public final static double lrAdam = 1e-4;	// learning rate used in Adam optimizer

	public static final int transformed_imgHeight = 32;
	public static final int transformed_imgWidth = 32;
	public static final int transformed_imgChannels = 3;

	public static final String fashion_downloader_py_filepath = "/dataset/download_convert_fashion.py"; // located in resources
	public static final String fashion_png_dirpath = "data/fashion_mnist_png";

	public static void main(String[] args) throws Exception {
		try {
			// Generate Keras model
			Path modelFilePath = Paths.get(vgg16_fashion_h5_filename);
			if (!Files.exists(modelFilePath) || !Files.isRegularFile(modelFilePath)) {
				System.out.println("Generating VGG-16 model in h5 format...");
				String pyScriptFullPath = new ClassPathResource(vgg16_py_filepath).getFile().getPath();
				PythonCommandHandler.runGenerateModelScript(pyScriptFullPath, vgg16_fashion_h5_filename, numClasses, lrAdam,
						transformed_imgHeight, transformed_imgWidth, transformed_imgChannels);
			}

			// Load Fashion MNIST
			Path datasetDir = Paths.get(fashion_png_dirpath);
			if (!Files.exists(datasetDir) || !Files.isDirectory(datasetDir)) {
				System.out.println("Getting Fashion MNIST as PNGs-dataset...");
				String scriptPath = new ClassPathResource(fashion_downloader_py_filepath).getFile().getPath();
				PythonCommandHandler.runDownloadDatasetScript(scriptPath, fashion_png_dirpath);
			}

			DataSetIterator fashionTrain = FashionMNISTDataloader.loadDataAndTransform(fashion_png_dirpath, batchSize, true, true,
					transformed_imgHeight, transformed_imgWidth, transformed_imgChannels);
			DataSetIterator fashionTest = FashionMNISTDataloader.loadDataAndTransform(fashion_png_dirpath, batchSize, false, false,
					transformed_imgHeight, transformed_imgWidth, transformed_imgChannels);

			// Subtract the mean RGB value
			fashionTrain.setPreProcessor(new VGG16ImagePreProcessor());
			fashionTest.setPreProcessor(new VGG16ImagePreProcessor());


			// Import Keras VGG-16 model with training config
			MultiLayerNetwork importedVgg16 = KerasModelImport.importKerasSequentialModelAndWeights(
					/* modelHdf5Stream = */vgg16_fashion_h5_filename,
					/* enforceTrainingConfig = */true);
			
			MultiLayerNetwork vgg16 = ModelRebuilder
					.rebuildSequentialModelWithOutputLayer(importedVgg16, rngSeed, 
							transformed_imgHeight, transformed_imgWidth, transformed_imgChannels);

			// Listener
			vgg16.setListeners(new ScoreIterationListener(10)); // print score every 10 batches
			
			// Training
			System.out.println("Starting training...");
			for (int i = 0; i < numEpochs; i++) {
				vgg16.fit(fashionTrain);
				System.out.println("Epoch " + (i + 1) + " completed.");
			}
			
			// Evaluation
			System.out.println("Starting evaluation...");
			var eval = vgg16.evaluate(fashionTest);
			System.out.println(eval.stats());
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
