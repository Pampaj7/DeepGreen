package io.github.stlabunifi.deepgreen.dl4j.model.builder;

//import java.util.Map;

//import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
//import org.deeplearning4j.nn.conf.graph.GraphVertex;
//import org.deeplearning4j.nn.conf.graph.LayerVertex;
//import org.deeplearning4j.nn.conf.layers.BaseLayer;
//import org.deeplearning4j.nn.weights.WeightInitXavierUniform;

import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;

//import io.github.stlabunifi.deepgreen.dl4j.model.ModelInspector;

public class Vgg16GraphBuilder {

	@SuppressWarnings("deprecation")
	public static ComputationGraph buildVGG16(int numClasses, int seed, 
			int imgChannels, int imgHeight, int imgWidth, double lr) {
		// Build the VGG-16 ZooModel
		ZooModel<?> vgg16Zoo = VGG16.builder()
				.numClasses(numClasses)
				.seed(seed)
				.inputShape(new int[] {imgChannels, imgHeight, imgWidth})
				.updater(new Adam(lr))
				.cacheMode(CacheMode.DEVICE) // Default: CacheMode.NONE
				.workspaceMode(WorkspaceMode.ENABLED) // Default value
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST) // Default value
				.build();
		
		// Temporarily initialization in order to obtain the original configuration
		ComputationGraph vgg16 = vgg16Zoo.init();


		// 1) Output layer uses NEGATIVELOGLIKELIHOOD, instead of MCXENT (same as TensorFlow's categorical_crossentropy)
		// Get last and second to last vertexes names
		String lastVertex = vgg16.getConfiguration().getNetworkOutputs().get(0); // "20" in VGG16
		String secondToLastVertex = vgg16.getConfiguration().getVertexInputs().get(lastVertex).get(0); // "19" in VGG16
		
		// Get the output size of the second to last layer
		long nOut = ((FeedForwardLayer) vgg16.getLayer(secondToLastVertex).conf().getLayer()).getNOut();
		
		// Substitute the last layer with a new one with MCXENT as loss function
		ComputationGraph vgg16WithCrossEntropyLoss = new TransferLearning.GraphBuilder(vgg16)
			.removeVertexAndConnections(lastVertex)
			.addLayer("output",
				new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
					.nIn(nOut)
					.nOut(numClasses)
					.activation(Activation.SOFTMAX)
					.updater(new Adam(lr))
					.build(),
				secondToLastVertex) // linked to second to last layer
			.setOutputs("output")
			.build();

/*
		// 2) Default initializer are XAVIER = gaussian distribution, instead of XAVIER_UNIFORM = uniform distribution (same as TensorFlow's GlorotUniform)
		// Deep copy of configuration (using JSON for compatibility)
		String confJson = vgg16WithCrossEntropyLoss.getConfiguration().toJson();
		ComputationGraphConfiguration conf = ComputationGraphConfiguration.fromJson(confJson);
		
		// Set XAVIER_UNIFORM for every BaseLayer (Conv, Dense, Output, etc)
		for (Map.Entry<String, GraphVertex> e : conf.getVertices().entrySet()) {
			GraphVertex gv = e.getValue();
			if (gv instanceof LayerVertex) {
				LayerVertex lv = (LayerVertex) gv;
				if (lv.getLayerConf() != null && lv.getLayerConf().getLayer() instanceof BaseLayer) {
					BaseLayer bl = (BaseLayer) lv.getLayerConf().getLayer();
					bl.setWeightInitFn(new WeightInitXavierUniform());
				}
			}
		}
		
		// Create a new ComputationGraph with the new initializer XAVIER_UNIFORM
		ComputationGraph vgg16WithLossAndWeights = new ComputationGraph(conf);
		vgg16WithLossAndWeights.init();
*/

		// DEBUG ONLY
//		ModelInspector.printWeightInitializer(vgg16WithCrossEntropyLoss);
//		ModelInspector.printGraphSummary(vgg16WithCrossEntropyLoss);
//		ModelInspector.printGraphDetails(vgg16WithCrossEntropyLoss);
		
		return vgg16WithCrossEntropyLoss;
	}

}
