package io.github.stlabunifi.deepgreen.dl4j.expt.imported.vgg16;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.nd4j.common.io.ClassPathResource;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import io.github.stlabunifi.deepgreen.dl4j.dataloader.TinyImageNetDataloader;
import io.github.stlabunifi.deepgreen.dl4j.model.builder.ModelRebuilder;
import io.github.stlabunifi.deepgreen.dl4j.python.handler.PythonCommandHandler;


import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

public class Vgg16TrainTinyExpt {

	public final static String vgg16_py_filepath = "/models/vgg16.py";
	public final static String vgg16_tiny_h5_filename = "vgg16_tiny.h5";

	public final static int rngSeed = 1234; 	// random number seed for reproducibility
	public final static int batchSize = 64; 	// batch size for each epoch
	public final static int numClasses = 200; 	// number of output classes
	public final static int numEpochs = 30;		// number of epochs to perform
	public final static double lrAdam = 1e-5;	// learning rate used in Adam optimizer

	public static final int transformed_imgHeight = 32;
	public static final int transformed_imgWidth = 32;
	public static final int transformed_imgChannels = 3;
	
	public static final String tiny_downloader_py_filepath = "/dataset/download_convert_tinyimage.py"; // located in reasources
	public static final String tiny_png_dirpath = "data/tiny_imagenet_png";

	public static void main(String[] args) throws Exception {
		try {
			// Generate Keras model
			Path modelFilePath = Paths.get(vgg16_tiny_h5_filename);
			if (!Files.exists(modelFilePath) || !Files.isRegularFile(modelFilePath)) {
				System.out.println("Generating VGG-16 model in h5 format...");
				String pyScriptFullPath = new ClassPathResource(vgg16_py_filepath).getFile().getPath();
				PythonCommandHandler.runGenerateModelScript(pyScriptFullPath, vgg16_tiny_h5_filename, numClasses, lrAdam,
						transformed_imgHeight, transformed_imgWidth, transformed_imgChannels);
			}

			// Load Tiny ImageNet-200
			Path datasetDir = Paths.get(tiny_png_dirpath);
			if (!Files.exists(datasetDir) || !Files.isDirectory(datasetDir)) {
				System.out.println("Getting Tiny ImageNet-200 as PNGs-dataset...");
				String scriptPath = new ClassPathResource(tiny_downloader_py_filepath).getFile().getPath();
				PythonCommandHandler.runDownloadDatasetScript(scriptPath, tiny_png_dirpath);
			}


			DataSetIterator tinyTrain = TinyImageNetDataloader.loadDataAndTransform(tiny_png_dirpath, batchSize, true, true,
					transformed_imgHeight, transformed_imgWidth, transformed_imgChannels);
			DataSetIterator tinyTest = TinyImageNetDataloader.loadDataAndTransform(tiny_png_dirpath, batchSize, false, false,
					transformed_imgHeight, transformed_imgWidth, transformed_imgChannels);
	
			// Subtract the mean RGB value
			tinyTrain.setPreProcessor(new VGG16ImagePreProcessor());
			tinyTest.setPreProcessor(new VGG16ImagePreProcessor());


			// Import Keras VGG-16 model with training config
			ComputationGraph importedVgg16 = KerasModelImport.importKerasModelAndWeights(
					/* modelHdf5Stream = */vgg16_tiny_h5_filename,
					/* enforceTrainingConfig = */true);
			
			ComputationGraph vgg16 = ModelRebuilder
					.rebuildModelWithInputShape(importedVgg16, rngSeed, 
							transformed_imgHeight, transformed_imgHeight, transformed_imgChannels);

			// Listener
			vgg16.setListeners(new ScoreIterationListener(10)); // print score every 10 batches
			
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
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
