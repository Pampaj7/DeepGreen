package io.github.stlabunifi.deepgreen.dl4j.core.model.builder;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.learning.config.Adam;

import io.github.stlabunifi.deepgreen.dl4j.core.model.ModelInspector;

public class Vgg16GraphBuilder {

	@SuppressWarnings("deprecation")
	public static ComputationGraph buildVGG16(int numClasses, int seed, 
			int imgChannels, int imgHeight, int imgWidth, double lr) {
		ZooModel<?> vgg16Zoo = VGG16.builder()
				.numClasses(numClasses)
				.seed(seed)
				.inputShape(new int[] {imgChannels, imgHeight, imgWidth})
				.updater(new Adam(lr))
				.build();
		
		ComputationGraph vgg16 = vgg16Zoo.init();
		ModelInspector.printGraphDetails(vgg16);
		
		return vgg16;
	}

}
