package io.github.stlabunifi.deepgreen.dl4j.expt.vgg16;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.nd4j.common.io.ClassPathResource;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import io.github.stlabunifi.deepgreen.dl4j.dataloader.FashionMNISTDataloader;
import io.github.stlabunifi.deepgreen.dl4j.model.Evaluator;
import io.github.stlabunifi.deepgreen.dl4j.model.builder.Vgg16GraphBuilder;
import io.github.stlabunifi.deepgreen.dl4j.python.handler.PythonCommandHandler;
import io.github.stlabunifi.deepgreen.dl4j.python.handler.DeepGreenTracker;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;

public class Vgg16TrainFashionExpt {

	public final static String emission_output_dir = "emissions";
	public final static String checkpoint_output_dir = "checkpoints";
	public final static String filename = "vgg16_fashion";

	public final static int rngSeed = (int) DeepGreenTracker.seed();
	public final static int batchSize = 128; 	// batch size for each epoch
	public final static int numClasses = 10; 	// number of output classes
	// Epochs, repetition and seed come from the shared run contract
	// (tools/deepgreen_tracker.py); the first campaign hard-coded 30 and had
	// no notion of an independent repetition.
	public final static int numEpochs = DeepGreenTracker.epochs();
	public final static double lrAdam = 1e-4;	// learning rate used in Adam optimizer

	public static final int transformed_imgHeight = 32;
	public static final int transformed_imgWidth = 32;
	public static final int transformed_imgChannels = 3;

	public static final String fashion_downloader_py_filepath = "/dataset/download_convert_fashion.py"; // located in resources
	public static final String fashion_png_dirpath = "data/fashion_mnist_png";

	public static void main(String[] args) {
		try {
			String moduleBaseDir = System.getProperty("module.basedir");
			Path emissionOutputDir;
			Path checkpointOutputDir;
			if (moduleBaseDir != null && !moduleBaseDir.isBlank()) {
				emissionOutputDir = Paths.get(moduleBaseDir, emission_output_dir);
				checkpointOutputDir = Paths.get(moduleBaseDir, checkpoint_output_dir);
			} else {
				emissionOutputDir = Paths.get(emission_output_dir).toAbsolutePath();
				checkpointOutputDir = Paths.get(checkpoint_output_dir).toAbsolutePath();
			}
			
			// Remove existing emission files
			String train_emission_filename = filename + "_train.csv";
			Path trainEmissionFilePath = emissionOutputDir.resolve(train_emission_filename);
			if (Files.exists(trainEmissionFilePath) && !Files.isDirectory(trainEmissionFilePath))
				Files.delete(trainEmissionFilePath);
			String test_emission_filename = filename +  "_test.csv";
			Path testEmissionFilePath = emissionOutputDir.resolve(test_emission_filename);
			if (Files.exists(testEmissionFilePath) && !Files.isDirectory(testEmissionFilePath))
				Files.delete(testEmissionFilePath);

			DeepGreenTracker tracker = DeepGreenTracker.start();


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

			// Normalize from (0 - 255) to (0 - 1)
			ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
			fashionTrain.setPreProcessor(scaler);
			fashionTest.setPreProcessor(scaler);


			ComputationGraph vgg16 = Vgg16GraphBuilder.buildVGG16(numClasses, rngSeed, 
					transformed_imgChannels, transformed_imgHeight, transformed_imgWidth,
					lrAdam);

			DeepGreenTracker.assertParameters(vgg16);

			// What this stack's loader actually produced, over the whole test split.
			// A batch is comparable across stacks only if it holds the same images,
			// and which images it holds depends on the loader's enumeration order.
			tracker.dataFingerprint(fashionTest);

			// Listener
			vgg16.setListeners(new ScoreIterationListener(10)); // print score every 10 batches
			
			// Training
			System.out.println("Starting training...");
			for (int i = 0; i < numEpochs; i++) {
				System.out.println("Epoch " + (i + 1) + "/" + numEpochs);
				
				tracker.startPhase("train", i + 1);
				vgg16.fit(fashionTrain);
				tracker.stopPhase();
				
				tracker.startPhase("eval", i + 1);
				double[] eval = Evaluator.lossAndAccuracy(vgg16, fashionTest);
				tracker.stopPhase();
				
				// Outside the tracked window: writing the metric must not be measured.
				// Test loss and accuracy come from one inference pass over the test set
				// (model/Evaluator.java); the training score is the model's last
				// minibatch score, as in the first campaign.
				tracker.metric(i + 1, vgg16.score(), eval[0], eval[1]);
				System.out.printf(java.util.Locale.ROOT,
						"test loss %.4f  accuracy %.2f%%%n", eval[0], eval[1]);
			}
			
			// Save the model
			String model_filename = filename + ".zip";
			Path modelFilePath = checkpointOutputDir.resolve(model_filename);
			// Create the directory first. Without this every Java run trained,
			// measured and reported correctly and then died on the last line with
			// FileNotFoundException, so the driver recorded the job as failed.
			Files.createDirectories(checkpointOutputDir);
			ModelSerializer.writeModel(vgg16, modelFilePath.toFile(), true);
			System.out.println("Model saved");
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
