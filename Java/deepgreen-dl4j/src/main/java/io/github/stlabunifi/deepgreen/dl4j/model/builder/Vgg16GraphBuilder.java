package io.github.stlabunifi.deepgreen.dl4j.model.builder;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
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
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.BaseLayer;

/**
 * VGG-16 with the classifier every other stack trains.
 *
 * <p>The zoo model's head is the ImageNet one: at a 32x32 input the last pool
 * emits 1x1x512, and the head is then 512 -> 4096 -> 4096 -> classes, which is
 * 34,006,948 parameters for CIFAR-100. The LibTorch stacks were training
 * 134,670,244 and JAX 14,765,988 -- four different networks under one name,
 * spanning 9.1x, while the specification claimed parameter counts were checked
 * against the exported module. Nothing checked them anywhere in the tree, and
 * half the study's energy comparisons were comparing models.
 *
 * <p>The canonical head, defined once in scripts/export_torchscript_models.py:
 * global average pooling to 512, then 512 -> 512 with ReLU and dropout, then
 * the classifier. The convolutional trunk is untouched at 14,714,688
 * parameters, so the total is 15,028,644 for 100 classes -- what
 * models/MANIFEST.json records and what DEEPGREEN_EXPECTED_PARAMS carries into
 * every run.
 */
public class Vgg16GraphBuilder {

	/** Last pooling vertex of the zoo model's convolutional trunk. */
	private static final String TRUNK_OUTPUT = "17";
	/** Dense 512->4096, Dense 4096->4096, Output 4096->classes. */
	private static final String[] IMAGENET_HEAD = {"20", "19", "18"};

	/**
	 * Re-initialise every convolution the way torchvision does.
	 *
	 * <p>VGG16.builder() takes no IWeightInit, so the zoo model's Xavier-family
	 * initialiser has to be replaced after the graph is configured. Done on the
	 * configuration in memory rather than through the JSON round trip the
	 * abandoned attempt in this file's history used: a round trip cannot carry
	 * an initialiser type it does not know.
	 */
	private static ComputationGraph withTorchvisionConvInit(ComputationGraph graph) {
		ComputationGraphConfiguration conf = graph.getConfiguration();
		for (java.util.Map.Entry<String, GraphVertex> e : conf.getVertices().entrySet()) {
			if (!(e.getValue() instanceof LayerVertex)) {
				continue;
			}
			LayerVertex lv = (LayerVertex) e.getValue();
			if (lv.getLayerConf() != null
					&& lv.getLayerConf().getLayer() instanceof ConvolutionLayer) {
				((BaseLayer) lv.getLayerConf().getLayer())
						.setWeightInitFn(TorchWeightInit.convolution());
			}
		}
		ComputationGraph reinitialised = new ComputationGraph(conf);
		reinitialised.init();
		return reinitialised;
	}

	@SuppressWarnings("deprecation")
	public static ComputationGraph buildVGG16(int numClasses, int seed,
			int imgChannels, int imgHeight, int imgWidth, double lr) {
		// The zoo model's own initialiser is Xavier-family. torchvision uses
		// kaiming_normal_(fan_out, relu) for every convolution, and holding
		// everything else fixed the initialiser is what decides whether VGG-16
		// collapses to chance at this learning rate -- 0 of 6 runs under He, 2 of
		// 6 under Glorot, 4 of 6 under Xavier. This stack carried 5 of the
		// campaign's 12 collapses. See TorchWeightInit.
		ZooModel<?> vgg16Zoo = VGG16.builder()
				.numClasses(numClasses)
				.seed(seed)
				.inputShape(new int[] {imgChannels, imgHeight, imgWidth})
				.updater(new Adam(lr))
				.cacheMode(CacheMode.DEVICE) // Default: CacheMode.NONE
				.workspaceMode(WorkspaceMode.ENABLED) // Default value
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST) // Default value
				.build();

		ComputationGraph vgg16 = vgg16Zoo.init();

		// MCXENT with a softmax output, matching TensorFlow's
		// categorical_crossentropy and torch's CrossEntropyLoss over logits.
		TransferLearning.GraphBuilder builder = new TransferLearning.GraphBuilder(vgg16);
		for (String vertex : IMAGENET_HEAD) {
			builder = builder.removeVertexAndConnections(vertex);
		}

		ComputationGraph vgg16Canonical = builder
			.addLayer("gap",
				new GlobalPoolingLayer.Builder(PoolingType.AVG).build(),
				TRUNK_OUTPUT)
			.addLayer("fc1",
				new DenseLayer.Builder()
					.nIn(512).nOut(512)
					.activation(Activation.RELU)
					.weightInit(TorchWeightInit.dense())
					.updater(new Adam(lr))
					.build(),
				"gap")
			.addLayer("drop",
				new DropoutLayer.Builder(0.5).build(),
				"fc1")
			.addLayer("output",
				new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
					.nIn(512)
					.nOut(numClasses)
					.activation(Activation.SOFTMAX)
					.weightInit(TorchWeightInit.dense())
					.updater(new Adam(lr))
					.build(),
				"drop")
			.setOutputs("output")
			.build();

		return withTorchvisionConvInit(vgg16Canonical);
	}

}
