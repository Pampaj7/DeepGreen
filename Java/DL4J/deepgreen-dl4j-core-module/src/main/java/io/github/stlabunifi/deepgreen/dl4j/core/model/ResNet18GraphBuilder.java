package io.github.stlabunifi.deepgreen.dl4j.core.model;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
//import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.learning.config.Adam;
//import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInit;
//import org.deeplearning4j.nn.weights.WeightInitDistribution;

public class ResNet18GraphBuilder {

	//static private IWeightInit weightInit = new WeightInitDistribution(new TruncatedNormalDistribution(0.0, 0.5)); //TODO: differes
	static private CacheMode cacheMode = CacheMode.NONE;
	static private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
	static private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

	private ResNet18GraphBuilder() {}

	public Class<? extends Model> modelType() {
		return ComputationGraph.class;
	}

	public static ComputationGraph buildResNet18(int numClasses, int seed, 
			int imgChannels, int imgHeight, int imgWidth, double lr) {
		
		ComputationGraphConfiguration.GraphBuilder graph = graphBuilder(numClasses, seed,
				imgChannels, imgHeight, imgWidth, lr);
		ComputationGraphConfiguration conf = graph.build();
		ComputationGraph model = new ComputationGraph(conf);
		model.init();
		
		ModelInspector.printGraphDetails(model);
		return model;
	}

	static public ComputationGraphConfiguration.GraphBuilder graphBuilder(int numClasses, int seed, 
			int imgChannels, int imgHeight, int imgWidth, double lr) {
		
		ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.activation(Activation.IDENTITY)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Adam(lr))
				.weightInit(WeightInit.RELU) // or weightInit var
				.l1(1e-7)
				.l2(5e-5)
				.miniBatch(true)
				.cacheMode(cacheMode)
				.trainingWorkspaceMode(workspaceMode)
				.inferenceWorkspaceMode(workspaceMode)
				.cudnnAlgoMode(cudnnAlgoMode)
				.convolutionMode(ConvolutionMode.Truncate)
				.graphBuilder();

		graph.addInputs("input")
				.setInputTypes(InputType.convolutional(
						imgHeight,
						imgWidth,
						imgChannels,
						CNN2DFormat.NCHW));

		// Initial layers
		graph
			.addLayer("zero", new ZeroPaddingLayer.Builder(3, 3).build(), "input")
			.addLayer("conv1", new ConvolutionLayer.Builder(7,7).stride(2,2).nOut(64).build(), "zero")
			.addLayer("bn1", new BatchNormalization.Builder().build(), "conv1")
			.addLayer("relu1", new ActivationLayer.Builder().activation(Activation.RELU).build(), "bn1")
			.addLayer("maxpool", new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(3,3).stride(2,2).build(), "relu1");

		String prev = "maxpool";

		// Add 4 stages of 2 residual blocks each
		prev = addResBlock(graph, prev, 64, false, "block1_1");
		prev = addResBlock(graph, prev, 64, false, "block1_2");

		prev = addResBlock(graph, prev, 128, true, "block2_1");
		prev = addResBlock(graph, prev, 128, false, "block2_2");

		prev = addResBlock(graph, prev, 256, true, "block3_1");
		prev = addResBlock(graph, prev, 256, false, "block3_2");

		prev = addResBlock(graph, prev, 512, true, "block4_1");
		prev = addResBlock(graph, prev, 512, false, "block4_2");

		// Global average pooling
		graph.addLayer("avgpool", new GlobalPoolingLayer.Builder()
				.poolingType(PoolingType.AVG)
				.build(), prev);

		// Output layer
		graph.addLayer("fc", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT) // Differs from ResNet50: NEGATIVELOGLIKELIHOOD
				.activation(Activation.SOFTMAX)
				.nOut(numClasses)
				.build(), "avgpool");

		graph.setOutputs("fc");

		return graph;
	}

	private static String addResBlock(ComputationGraphConfiguration.GraphBuilder graph,
			String input, int filters, boolean downsample, String namePrefix) {

		int stride = downsample ? 2 : 1;
		String conv1 = namePrefix + "_conv1";
		String bn1 = namePrefix + "_bn1";
		String relu1 = namePrefix + "_relu1";

		String conv2 = namePrefix + "_conv2";
		String bn2 = namePrefix + "_bn2";

		String add = namePrefix + "_add";
		String reluOut = namePrefix + "_relu_out";

		// Main branch
		graph
			// First conv
			.addLayer(conv1, 
					new ConvolutionLayer.Builder(3,3)
						.stride(stride,stride)
						.nOut(filters)
						.cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
						.convolutionMode(ConvolutionMode.Same) // Differs from resnet50
						.build(),
					input)
			.addLayer(bn1, new BatchNormalization.Builder().build(), conv1)
			.addLayer(relu1, new ActivationLayer.Builder().activation(Activation.RELU).build(), bn1)

			// Second conv
			.addLayer(conv2, 
					new ConvolutionLayer.Builder(3,3)
						.stride(1,1)
						.nOut(filters)
						.cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
						.convolutionMode(ConvolutionMode.Same)
						.build(),
					relu1)
			.addLayer(bn2, new BatchNormalization.Builder().build(), conv2);

		// Shortcut branch
		String shortcutOut;
		if (downsample) {
			String shortcutConv = namePrefix + "_shortcut_conv";
			String shortcutBN = namePrefix + "_shortcut_bn";

			graph
				.addLayer(shortcutConv,
						new ConvolutionLayer.Builder(1,1)
							.stride(stride,stride)
							.nOut(filters)
							.convolutionMode(ConvolutionMode.Same) // Differs from resnet50
							.build(),
						input)
				.addLayer(shortcutBN, new BatchNormalization.Builder().build(), shortcutConv);
			shortcutOut = shortcutBN;
		} else {
			shortcutOut = input;
		}

		// Add
		graph
			.addVertex(add, new ElementWiseVertex(ElementWiseVertex.Op.Add), bn2, shortcutOut)
			.addLayer(reluOut, new ActivationLayer.Builder().activation(Activation.RELU).build(), add);

		return reluOut;
	}
}