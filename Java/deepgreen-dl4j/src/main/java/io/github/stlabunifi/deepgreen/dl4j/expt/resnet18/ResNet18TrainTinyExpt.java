package io.github.stlabunifi.deepgreen.dl4j.expt.resnet18;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.nd4j.common.io.ClassPathResource;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import io.github.stlabunifi.deepgreen.dl4j.dataloader.TinyImageNetDataloader;
import io.github.stlabunifi.deepgreen.dl4j.model.builder.ResNet18GraphBuilder;
import io.github.stlabunifi.deepgreen.dl4j.python.handler.PythonCommandHandler;
import io.github.stlabunifi.deepgreen.dl4j.python.handler.PythonTrackerHandler;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

public class ResNet18TrainTinyExpt {

	public final static String emission_output_dir = "emissions";
	public final static String emission_filename = "resnet18_tiny";

	public final static int rngSeed = 123; 	// random number seed for reproducibility
	public final static int batchSize = 128; 	// batch size for each epoch
	public final static int numClasses = 200; 	// number of output classes
	public final static int numEpochs = 30; 	// number of epochs to perform
	public final static double lrAdam = 1e-4; 	// learning rate used in Adam optimizer

	public static final int transformed_imgHeight = 32;
	public static final int transformed_imgWidth = 32;
	public static final int transformed_imgChannels = 3;
	
	public static final String tiny_downloader_py_filepath = "/dataset/download_convert_tinyimage.py"; // located in reasources
	public static final String tiny_png_dirpath = "data/tiny_imagenet_png";

	public static void main(String[] args) {
		try {
			String moduleBaseDir = System.getProperty("module.basedir");
			Path emissionOutputDir;
			if (moduleBaseDir != null && !moduleBaseDir.isBlank()) {
				emissionOutputDir = Paths.get(moduleBaseDir, emission_output_dir);
			} else {
				emissionOutputDir = Paths.get(emission_output_dir).toAbsolutePath();
			}
			
			// Remove existing emission files
			String train_emission_filename = emission_filename + "_train.csv";
			Path trainEmissionFilePath = emissionOutputDir.resolve(train_emission_filename);
			if (Files.exists(trainEmissionFilePath) && !Files.isDirectory(trainEmissionFilePath))
				Files.delete(trainEmissionFilePath);
			String test_emission_filename = emission_filename +  "_test.csv";
			Path testEmissionFilePath = emissionOutputDir.resolve(test_emission_filename);
			if (Files.exists(testEmissionFilePath) && !Files.isDirectory(testEmissionFilePath))
				Files.delete(testEmissionFilePath);

			PythonTrackerHandler trackerHandler = new PythonTrackerHandler(emissionOutputDir.toString());


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

			// Normalize from (0 - 255) to (0 - 1)
			ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
			tinyTrain.setPreProcessor(scaler);
			tinyTest.setPreProcessor(scaler);


			ComputationGraph resnet18 = ResNet18GraphBuilder.buildResNet18(numClasses, rngSeed, 
					transformed_imgChannels, transformed_imgHeight, transformed_imgWidth, lrAdam);

			// Listener
			resnet18.setListeners(new ScoreIterationListener(10)); // print score every 10 batches
			
			// Training
			System.out.println("Starting training...");
			for (int i = 0; i < numEpochs; i++) {
				System.out.println("Epoch " + (i + 1) + "/" + numEpochs);
				
				trackerHandler.startTracker(train_emission_filename);
				resnet18.fit(tinyTrain);
				trackerHandler.stopTracker();
			
				trackerHandler.startTracker(test_emission_filename);
				var eval = resnet18.evaluate(tinyTest);
				trackerHandler.stopTracker();
				
				System.out.println(eval.stats());
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
