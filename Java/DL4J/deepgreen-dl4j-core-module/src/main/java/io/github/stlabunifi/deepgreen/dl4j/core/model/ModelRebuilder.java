package io.github.stlabunifi.deepgreen.dl4j.core.model;

import java.util.List;

import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.learning.config.Adam;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;

public class ModelRebuilder {

	public static ComputationGraph rebuildModelWithInputShape(ComputationGraph importedModel, long seed,
			int imgHeight, int imgWidth, int imgChannels) {
		// Get imported model configuration
		ComputationGraphConfiguration importedConf = importedModel.getConfiguration();

		// Create same configuration with input shape
		ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.updater(new Adam(1e-5))
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

}