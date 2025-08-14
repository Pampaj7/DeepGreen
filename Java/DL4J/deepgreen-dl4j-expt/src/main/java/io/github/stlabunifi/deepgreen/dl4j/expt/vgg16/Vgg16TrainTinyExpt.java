package io.github.stlabunifi.deepgreen.dl4j.expt.vgg16;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.nd4j.common.io.ClassPathResource;
import org.deeplearning4j.nn.graph.ComputationGraph;
//import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import io.github.stlabunifi.deepgreen.dl4j.core.dataloader.TinyImageNetDataloader;
import io.github.stlabunifi.deepgreen.dl4j.core.model.builder.Vgg16GraphBuilder;
import io.github.stlabunifi.deepgreen.dl4j.python.handler.PythonCommandHandler;
import io.github.stlabunifi.deepgreen.dl4j.python.handler.PythonTrackerHandler;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

public class Vgg16TrainTinyExpt {

	public final static String emission_output_dir = "emissions";
	public final static String emission_filename = "vgg16_tiny.csv";

	//public final static String vgg16_py_filepath = "/models/vgg16.py";
	//public final static String vgg16_tiny_h5_filename = "vgg16_tiny.h5";

	public final static int rngSeed = 1234; 	// random number seed for reproducibility
	public final static int batchSize = 128; 	// batch size for each epoch
	public final static int numClasses = 200; 	// number of output classes
	public final static int numEpochs = 30;		// number of epochs to perform
	public final static double lrAdam = 1e-5;	// learning rate used in Adam optimizer

	public static final int transformed_imgHeight = 32;
	public static final int transformed_imgWidth = 32;
	public static final int transformed_imgChannels = 3;
	
	public static final String tiny_downloader_py_filepath = "/dataset/download_convert_tinyimage.py"; // located in reasources
	public static final String tiny_png_dirpath = "data/tiny_imagenet_png";

	public static void main(String[] args) {
		try {
			Path emissionOutputDir = Paths.get(emission_output_dir).toAbsolutePath();
			// Remove existing emission file
			Path emissionFilePath = emissionOutputDir.resolve(emission_filename);
			if (Files.exists(emissionFilePath) && !Files.isDirectory(emissionFilePath))
				Files.delete(emissionFilePath);

			PythonTrackerHandler trackerHandler = new PythonTrackerHandler(emissionOutputDir.toString());

			// Generate Keras model
			//Path modelFilePath = Paths.get(vgg16_tiny_h5_filename);
			//if (!Files.exists(modelFilePath) || !Files.isRegularFile(modelFilePath)) {
			//	System.out.println("Generating VGG-16 model in h5 format...");
			//	String pyScriptFullPath = new ClassPathResource(vgg16_py_filepath).getFile().getPath();
			//	PythonCommandHandler.runGenerateModelScript(pyScriptFullPath, vgg16_tiny_h5_filename, numClasses, lrAdam);
			//}

			// Load Tiny ImageNet-200
			Path datasetDir = Paths.get(tiny_png_dirpath);
			if (!Files.exists(datasetDir) || !Files.isDirectory(datasetDir)) {
				System.out.println("Getting Tiny ImageNet-200 as PNGs-dataset...");
				String scriptPath = new ClassPathResource(tiny_downloader_py_filepath).getFile().getPath();
				PythonCommandHandler.runDownloadDatasetScript(scriptPath, tiny_png_dirpath);
			}


			DataSetIterator tinyTrain = TinyImageNetDataloader.loadDataAndTransform(tiny_png_dirpath, batchSize, true,
					transformed_imgHeight, transformed_imgWidth, transformed_imgChannels);
			DataSetIterator tinyTest = TinyImageNetDataloader.loadDataAndTransform(tiny_png_dirpath, batchSize, false,
					transformed_imgHeight, transformed_imgWidth, transformed_imgChannels);
	
			// Normalize from (0-255) to (0-1)
			tinyTrain.setPreProcessor(new VGG16ImagePreProcessor());
			tinyTest.setPreProcessor(new VGG16ImagePreProcessor());


			// Import Keras VGG-16 model with training config
			//ComputationGraph importedVgg16 = KerasModelImport.importKerasModelAndWeights(
			//		/* modelHdf5Stream = */vgg16_tiny_h5_filename,
			//		/* enforceTrainingConfig = */true);
			//
			//ComputationGraph vgg16 = ModelRebuilder
			//		.rebuildModelWithInputShape(importedVgg16, rngSeed, imgHeight, imgWidth, imgChannels);

			ComputationGraph vgg16 = Vgg16GraphBuilder.buildVGG16(numClasses, rngSeed, 
					transformed_imgChannels, transformed_imgHeight, transformed_imgWidth, lrAdam);

			// Listener
			vgg16.setListeners(new ScoreIterationListener(100)); // print score every 100 batches
			
			// Training
			System.out.println("Starting training...");
			for (int i = 0; i < numEpochs; i++) {
				trackerHandler.startTracker(emission_filename);
				vgg16.fit(tinyTrain);
				trackerHandler.stopTracker();
				System.out.println("Epoch " + (i + 1) + " completed.");
			}
			
			// Evaluation
			System.out.println("Starting evaluation...");
			trackerHandler.startTracker(emission_filename);
			var eval = vgg16.evaluate(tinyTest);
			trackerHandler.stopTracker();
			System.out.println(eval.stats());
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
