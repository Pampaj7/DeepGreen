package io.github.stlabunifi.deepgreen.dl4j.expt.vgg16;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

import org.nd4j.common.io.ClassPathResource;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import io.github.stlabunifi.deepgreen.dl4j.python.handler.PythonCommandHandler;
import io.github.stlabunifi.dl4j.core.model.ModelRebuilder;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

public class Vgg16TrainTinyExpt {

	public final static String vgg16_py_filepath = "/models/vgg16.py";
	public final static String vgg16_tiny_h5_filename = "vgg16_tiny.h5";
	public final static int rngSeed = 1234; 	// random number seed for reproducibility
	public final static int batchSize = 64; 	// batch size for each epoch
	public final static int numClasses = 200; 	// number of output classes
	public final static int numEpochs = 1; 		// number of epochs to perform

	public static final int imgHeight = 64;
	public static final int imgWidth = 64;
	public static final int imgChannels = 3;
	
	public static final String tiny_downloader_py_filepath = "download_convert_tinyimage.py"; // located in reasources
	public static final String tiny_png_dir = "tiny_imagenet_png";

	public static void main(String[] args) throws Exception {
		// Generate Keras model
		Path modelFilePath = Paths.get(vgg16_tiny_h5_filename);
		if (!Files.exists(modelFilePath) || !Files.isRegularFile(modelFilePath)) {
			try {
				String pyScriptFullPath = new ClassPathResource(vgg16_py_filepath).getFile().getPath();
				PythonCommandHandler.runGenerateModelScript(pyScriptFullPath, vgg16_tiny_h5_filename, numClasses);
			} catch (Exception e) {
				e.printStackTrace();
				return;
			}
		}

		// Load Tiny ImageNet-200
		DataSetIterator tinyTrain = new TinyImageNetDataSetIterator(batchSize, DataSetType.TRAIN); // uses rngSeed = 123 by default
		DataSetIterator tinyTest = new TinyImageNetDataSetIterator(batchSize, DataSetType.TEST); // uses rngSeed = 123 by default

		// Normalize from (0-255) to (0-1)
		tinyTrain.setPreProcessor(new VGG16ImagePreProcessor());
		tinyTest.setPreProcessor(new VGG16ImagePreProcessor());
		
		DataSet ds = tinyTrain.next();
		System.out.println(Arrays.toString(ds.getFeatures().shape()));


		// Import Keras VGG-16 model with training config
		ComputationGraph importedVgg16 = KerasModelImport.importKerasModelAndWeights(
				/* modelHdf5Stream = */vgg16_tiny_h5_filename,
				/* enforceTrainingConfig = */true);
		
		ComputationGraph vgg16 = ModelRebuilder
				.rebuildModelWithInputShape(importedVgg16, rngSeed, imgHeight, imgWidth, imgChannels);

		// Listener
		vgg16.setListeners(new ScoreIterationListener(100)); // stampa score ogni 100 batch


		// Training
		System.out.println("Starting training...");
		for (int i = 0; i < numEpochs; i++) {
			vgg16.fit(tinyTrain);
			System.out.println("Epoch " + (i + 1) + " completed.");
		}

		// Evaluation
		System.out.println("Starting evaluation...");
		var eval = vgg16.evaluate(tinyTest);
		System.out.println(eval.stats());
	}

}
