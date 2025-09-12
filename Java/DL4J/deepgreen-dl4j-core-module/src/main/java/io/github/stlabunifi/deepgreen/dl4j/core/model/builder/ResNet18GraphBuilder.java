package io.github.stlabunifi.deepgreen.dl4j.core.model.builder;

//import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution;
//import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
//import org.deeplearning4j.nn.weights.IWeightInit;
//import org.deeplearning4j.nn.weights.WeightInitDistribution;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.learning.config.Adam;

import io.github.stlabunifi.deepgreen.dl4j.core.model.ModelInspector;

public class ResNet18GraphBuilder {

	static private CacheMode cacheMode = CacheMode.DEVICE; // Default: CacheMode.NONE
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
		
		ModelInspector.printWeightInitializer(model);
		ModelInspector.printGraphSummary(model);
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
				.weightInit(WeightInit.RELU) 	// set weights via a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in)
												// where fan_in is the number of input units in the weight tensor.
												// This is similar to "kernel_initializer="he_normal"" in Python,
												// but PyTorch uses fan_out = kernel_size * out_channels.
												// Differs from ResNet50: "new WeightInitDistribution(new TruncatedNormalDistribution(0.0, 0.5))"
				//.l1(1e-7) // used in ResNet50
				//.l2(1e-4) // used in paper ResNet-18 (but with different training config); ResNet50 of DL4J uses 5e-5
				.miniBatch(true)
				.cacheMode(cacheMode)
				.trainingWorkspaceMode(workspaceMode)
				.inferenceWorkspaceMode(workspaceMode)
				.cudnnAlgoMode(cudnnAlgoMode)
				.convolutionMode(ConvolutionMode.Truncate) // This conf requires to use .convolutionMode(ConvolutionMode.Same) in each ConvolutionLayer
				.graphBuilder();

		graph.addInputs("input")
				.setInputTypes(InputType.convolutional(
						imgHeight,
						imgWidth,
						imgChannels,
						CNN2DFormat.NCHW));

		// Initial layers
		graph
			//.addLayer("zero", new ZeroPaddingLayer.Builder(3, 3).build(), "input") // Same to use .convolutionMode(ConvolutionMode.Same) in next ConvolutionLayer
			.addLayer("conv1", new ConvolutionLayer.Builder(7,7)
						.stride(2,2)
						.nOut(64)
						.convolutionMode(ConvolutionMode.Same)
						.build(),
					"input")
			.addLayer("bn1", new BatchNormalization.Builder()
						.eps(1e-5)
						.decay(0.9)
						.build(),
					"conv1")
			.addLayer("relu1", new ActivationLayer.Builder().activation(Activation.RELU).build(), "bn1")
			.addLayer("maxpool", new SubsamplingLayer.Builder(PoolingType.MAX)
						.kernelSize(3,3)
						.stride(2,2)
						.padding(1,1)
						.build(),
					"relu1");

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
						.convolutionMode(ConvolutionMode.Same) // padding 1x1
						.build(),
					input)
			.addLayer(bn1, new BatchNormalization.Builder()
						.eps(1e-5)
						.decay(0.9)
						.build(),
					conv1)
			.addLayer(relu1, new ActivationLayer.Builder().activation(Activation.RELU).build(), bn1)

			// Second conv
			.addLayer(conv2, 
					new ConvolutionLayer.Builder(3,3)
						.stride(1,1) // in DLJ is omitted because stride=1 is default value 
						.nOut(filters)
						.cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
						.convolutionMode(ConvolutionMode.Same) // padding 1x1
						.build(),
					relu1)
			.addLayer(bn2, new BatchNormalization.Builder()
						.eps(1e-5)
						.decay(0.9)
						.build(),
					conv2);

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
							// conv 1x1 doesn't require padding, i.e. .convolutionMode(ConvolutionMode.Same)
							.build(),
						input)
				.addLayer(shortcutBN, new BatchNormalization.Builder()
							.eps(1e-5)
							.decay(0.9)
							.build(),
						shortcutConv); 	// Although not present in DLJ (see https://github.com/deepjavalibrary/d2l-java/blob/master/chapter_convolutional-modern/resnet.ipynb),
										// both the official PyTorch (see https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18)
										// and Tensorflow (https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/resnet.py)
										// implementations use the bn layer here.
			shortcutOut = shortcutBN;
		} else {
			shortcutOut = input; // equivalent to identity block
		}

		// Add
		graph
			.addVertex(add, new ElementWiseVertex(ElementWiseVertex.Op.Add), bn2, shortcutOut)
			.addLayer(reluOut, new ActivationLayer.Builder().activation(Activation.RELU).build(), add);

		return reluOut;
	}
}