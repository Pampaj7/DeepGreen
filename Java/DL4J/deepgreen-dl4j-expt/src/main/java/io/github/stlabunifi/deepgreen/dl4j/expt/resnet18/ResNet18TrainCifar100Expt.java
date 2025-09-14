package io.github.stlabunifi.deepgreen.dl4j.expt.resnet18;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.nd4j.common.io.ClassPathResource;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import io.github.stlabunifi.deepgreen.dl4j.core.dataloader.Cifar100Dataloader;
import io.github.stlabunifi.deepgreen.dl4j.core.model.builder.ResNet18GraphBuilder;
import io.github.stlabunifi.deepgreen.dl4j.python.handler.PythonCommandHandler;
import io.github.stlabunifi.deepgreen.dl4j.python.handler.PythonTrackerHandler;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

public class ResNet18TrainCifar100Expt {

	public final static String emission_output_dir = "emissions";
	public final static String emission_filename = "resnet18_cifar100.csv";

	public final static int rngSeed = 123; 		// random number seed for reproducibility
	public final static int batchSize = 128; 	// batch size for each epoch
	public final static int numClasses = 100; 	// number of output classes
	public final static int numEpochs = 30; 	// number of epochs to perform
	public final static double lrAdam = 1e-4; 	// learning rate used in Adam optimizer

	public static final int imgHeight = 32;
	public static final int imgWidth = 32;
	public static final int imgChannels = 3;
	
	public static final String cifar100_downloader_py_filepath = "/dataset/download_convert_cifar100.py"; // located in resources
	public static final String cifar100_png_dirpath = "data/cifar100_png";

	public static void main(String[] args) {
		try {
			String moduleBaseDir = System.getProperty("module.basedir");
			Path emissionOutputDir;
			if (moduleBaseDir != null && !moduleBaseDir.isBlank()) {
				emissionOutputDir = Paths.get(moduleBaseDir, emission_output_dir);
			} else {
				emissionOutputDir = Paths.get(emission_output_dir).toAbsolutePath();
			}
			
			// Remove existing emission file
			Path emissionFilePath = emissionOutputDir.resolve(emission_filename);
			if (Files.exists(emissionFilePath) && !Files.isDirectory(emissionFilePath))
				Files.delete(emissionFilePath);

			PythonTrackerHandler trackerHandler = new PythonTrackerHandler(emissionOutputDir.toString());


			// Load CIFAR-100
			Path datasetDir = Paths.get(cifar100_png_dirpath);
			if (!Files.exists(datasetDir) || !Files.isDirectory(datasetDir)) {
				System.out.println("Getting CIFAR-100 as PNGs-dataset...");
				String scriptPath = new ClassPathResource(cifar100_downloader_py_filepath).getFile().getPath();
				PythonCommandHandler.runDownloadDatasetScript(scriptPath, cifar100_png_dirpath);
			}

			DataSetIterator cifar100Train = Cifar100Dataloader.loadData(cifar100_png_dirpath, batchSize, true, true);
			DataSetIterator cifar100Test = Cifar100Dataloader.loadData(cifar100_png_dirpath, batchSize, false, false);

			// Normalize from (0 - 255) to (0 - 1)
			ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
			cifar100Train.setPreProcessor(scaler);
			cifar100Test.setPreProcessor(scaler);


			ComputationGraph resnet18 = ResNet18GraphBuilder.buildResNet18(numClasses, rngSeed, 
					imgChannels, imgHeight, imgWidth, lrAdam);

			// Listener
			resnet18.setListeners(new ScoreIterationListener(10)); // print score every 10 batches
			
			// Training
			System.out.println("Starting training...");
			for (int i = 0; i < numEpochs; i++) {
				System.out.println("Epoch " + (i + 1) + "/" + numEpochs);
				
				trackerHandler.startTracker(emission_filename);
				resnet18.fit(cifar100Train);
				trackerHandler.stopTracker();
				
				// Evaluation
				trackerHandler.startTracker(emission_filename);
				var eval = resnet18.evaluate(cifar100Test);
				trackerHandler.stopTracker();
				
				System.out.println(eval.stats());
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}


