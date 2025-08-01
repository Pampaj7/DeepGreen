package io.github.stlabunifi.deepgreen.dl4j.expt.vgg16;

import java.io.File;
import java.util.Arrays;

import org.nd4j.common.io.ClassPathResource;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

public class TrainTiny {

	public final static String model_vgg16_py_filename = "/models/model_vgg16_tiny.py";
	public final static String model_vgg16_h5_filename = "model_vgg16_tiny.h5";
	public final static int rngSeed = 1234; 	// random number seed for reproducibility
	public final static int batchSize = 64; 	// batch size for each epoch
	public final static int outputNum = 200; 	// number of output classes
	public final static int numEpochs = 1; 		// number of epochs to perform

	public static void main(String[] args) throws Exception {
		// Generate Keras model
		File f = new File(model_vgg16_h5_filename);
		if(!f.exists() || f.isDirectory()) {
			String pyScriptFullPath = new ClassPathResource(model_vgg16_py_filename).getFile().getPath();
			PythonHandler.runModelGenerationScript(pyScriptFullPath, model_vgg16_h5_filename);
		}

		// Load Tiny ImageNet-200
		DataSetIterator tinyTrain = new TinyImageNetDataSetIterator(batchSize, DataSetType.TRAIN); // uses rngSeed = 123 by default
		DataSetIterator tinyTest = new TinyImageNetDataSetIterator(batchSize, DataSetType.TEST); // uses rngSeed = 123 by default

		// Normalize from (0-255) to (0-1)
		tinyTrain.setPreProcessor(new VGG16ImagePreProcessor());
		tinyTest.setPreProcessor(new VGG16ImagePreProcessor());
		
		DataSet ds = tinyTrain.next();
		System.out.println(Arrays.toString(ds.getFeatures().shape()));

		// TODO: call Python code to create model to import

		// Import Keras VGG-16 model with training config
		ComputationGraph importedVgg16 = KerasModelImport.importKerasModelAndWeights(
				/* modelHdf5Stream = */model_vgg16_h5_filename,
				/* enforceTrainingConfig = */true);
		
		ComputationGraph vgg16 = VGG16TinyImageNetRebuilder
				.rebuildModelWithInputShape(importedVgg16, rngSeed);

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
