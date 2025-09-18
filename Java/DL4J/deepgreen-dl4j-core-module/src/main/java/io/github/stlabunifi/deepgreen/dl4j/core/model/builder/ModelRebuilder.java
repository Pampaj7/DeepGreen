package io.github.stlabunifi.deepgreen.dl4j.core.model.builder;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.apache.commons.lang3.SerializationUtils;

import io.github.stlabunifi.deepgreen.dl4j.core.model.ModelInspector;

public class ModelRebuilder {

	/**
	 * DL4J imports Keras models in .h5 format, but it doesn't recognize
	 * the input shape (even if specified with Keras). This function copy
	 * the whole model configuration and adds the necessary input shape.
	 */
	public static ComputationGraph rebuildModelWithInputShape(ComputationGraph importedModel, long seed,
			int imgHeight, int imgWidth, int imgChannels) {
		// Get imported model configuration
		ComputationGraphConfiguration importedConf = importedModel.getConfiguration();

		// Create same configuration with input shape
		ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.weightInit(WeightInit.RELU)
				.activation(Activation.IDENTITY)
				.convolutionMode(ConvolutionMode.Same)
				.graphBuilder()
					.addInputs(importedConf.getNetworkInputs().get(0))
					.setInputTypes(InputType.convolutional(
							imgHeight,
							imgWidth,
							imgChannels,
							CNN2DFormat.NCHW));
		
		for (String vertexName : importedConf.getVertices().keySet()) {
			List<String> inputs = importedConf.getVertexInputs().get(vertexName);
			
			String[] stringInputs = new String[inputs.size()];
			for (int i = 0; i < inputs.size(); i++) {
				stringInputs[i] = inputs.get(i);
				
			}
			graphBuilder.addVertex(
					vertexName, 
					importedConf.getVertices().get(vertexName),
					stringInputs);
		}

		ComputationGraphConfiguration newConf = graphBuilder
				.setOutputs(importedConf.getNetworkOutputs().get(0))
				.build();

		// Create a new model
		ComputationGraph newModel = new ComputationGraph(newConf);
		newModel.init();
		
		//newModel.setParams(importedModel.params());
		
		ModelInspector.printGraphDetails(newModel);

		return newModel;
	}


	/**
	 * FC layers defined through Keras are not correctly converted as Output Layer,
	 * necessary for classification purposes in DL4J. This function copies the model configuration
	 * and replaces the last two layers (DenseLayer and LossLayer) with just one OutputLayer
	 * with the same configuration.
	 */
	public static MultiLayerNetwork rebuildSequentialModelWithOutputLayer(MultiLayerNetwork importedModel,
			long seed, int imgHeight, int imgWidth, int imgChannels) {
		// Get current model configuration
		MultiLayerConfiguration oldConf = importedModel.getLayerWiseConfigurations();
		int numLayers = oldConf.getConfs().size();

		List<org.deeplearning4j.nn.conf.layers.Layer> newLayerList = new ArrayList<>();
		for (int i = 0; i < numLayers - 1; i++) {
			org.deeplearning4j.nn.conf.layers.Layer origLayer = oldConf.getConf(i).getLayer();
			
			if (i == numLayers - 2) { // Replace
				OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(((FeedForwardLayer) origLayer).getNIn())
						.nOut(((FeedForwardLayer) origLayer).getNOut())
						.activation(((BaseLayer) origLayer).getActivationFn())
						.updater(((BaseLayer) origLayer).getIUpdater())
						.build();

				newLayerList.add(outputLayer);
			} else { // Copy
				newLayerList.add(SerializationUtils.clone(origLayer));
			}
		}

		// Rebuild configuration
		NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.list();
		
		for (int i = 0; i < newLayerList.size(); i++) {
			listBuilder.layer(i, newLayerList.get(i));
		}
		
		MultiLayerConfiguration newConf = listBuilder
				.setInputType(InputType.convolutional(imgHeight, imgWidth, imgChannels, CNN2DFormat.NCHW))
				.build();
		
		// Create new model
		MultiLayerNetwork newModel = new MultiLayerNetwork(newConf);
		newModel.init();
		
		ModelInspector.printModelHierarchy(newModel);

		return newModel;
	}
}
