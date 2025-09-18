package io.github.stlabunifi.deepgreen.dl4j.expt.imported.resnet18;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.nd4j.common.io.ClassPathResource;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import io.github.stlabunifi.deepgreen.dl4j.core.dataloader.Cifar100Dataloader;
import io.github.stlabunifi.deepgreen.dl4j.core.model.builder.ModelRebuilder;
import io.github.stlabunifi.deepgreen.dl4j.python.handler.PythonCommandHandler;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

public class Resnet18TrainCifar100Expt {

	public final static String resnet18_py_filepath = "/models/resnet18.py";
	public final static String resnet18_cifar100_h5_filename = "resnet18_cifar100.h5";

	public final static int rngSeed = 123; 		// random number seed for reproducibility
	public final static int batchSize = 128; 	// batch size for each epoch
	public final static int numClasses = 100; 	// number of output classes
	public final static int numEpochs = 30; 	// number of epochs to perform
	public final static double lrAdam = 1e-3; 	// learning rate used in Adam optimizer

	public static final int imgHeight = 32;
	public static final int imgWidth = 32;
	public static final int imgChannels = 3;
	
	public static final String cifar100_downloader_py_filepath = "/dataset/download_convert_cifar100.py"; // located in resources
	public static final String cifar100_png_dirpath = "data/cifar100_png";

	public static void main(String[] args) throws Exception {
		try {
			// Generate Keras model
			Path modelFilePath = Paths.get(resnet18_cifar100_h5_filename);
			if (!Files.exists(modelFilePath) || !Files.isRegularFile(modelFilePath)) {
				System.out.println("Generating ResNet-18 model in h5 format...");
				String pyScriptFullPath = new ClassPathResource(resnet18_py_filepath).getFile().getPath();
				PythonCommandHandler.runGenerateModelScript(pyScriptFullPath, resnet18_cifar100_h5_filename, numClasses, lrAdam,
						imgHeight, imgWidth, imgChannels);
			}

			// Load CIFAR-100
			Path datasetDir = Paths.get(cifar100_png_dirpath);
			if (!Files.exists(datasetDir) || !Files.isDirectory(datasetDir)) {
				System.out.println("Getting CIFAR-100 as PNGs-dataset...");
				String scriptPath = new ClassPathResource(cifar100_downloader_py_filepath).getFile().getPath();
				PythonCommandHandler.runDownloadDatasetScript(scriptPath, cifar100_png_dirpath);
			}

			DataSetIterator cifar100Train = Cifar100Dataloader.loadData(cifar100_png_dirpath, batchSize, true, true);
			DataSetIterator cifar100Test = Cifar100Dataloader.loadData(cifar100_png_dirpath, batchSize, false, false);

			// Normalize from (0 - 255) to (-1 - 1)
			cifar100Train.setPreProcessor(new ImagePreProcessingScaler(-1, 1));
			cifar100Test.setPreProcessor(new ImagePreProcessingScaler(-1, 1));


			// Import Keras ResNet-18 model with training config
			ComputationGraph importedResnet18 = KerasModelImport.importKerasModelAndWeights(
					/* modelHdf5Stream = */resnet18_cifar100_h5_filename,
					/* enforceTrainingConfig = */true);
			
			ComputationGraph resnet18 = ModelRebuilder.rebuildModelWithInputShape(importedResnet18, rngSeed,
					imgHeight, imgWidth, imgChannels);

			// Listener
			resnet18.setListeners(new ScoreIterationListener(10)); // print score every 10 batches
			
			// Training
			System.out.println("Starting training...");
			for (int i = 0; i < numEpochs; i++) {
				resnet18.fit(cifar100Train);
				System.out.println("Epoch " + (i + 1) + " completed.");
			}
			
			// Evaluation
			System.out.println("Starting evaluation...");
			var eval = resnet18.evaluate(cifar100Test);
			System.out.println(eval.stats());
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}


