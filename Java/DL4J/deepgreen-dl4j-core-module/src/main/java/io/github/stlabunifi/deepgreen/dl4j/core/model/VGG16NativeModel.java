package io.github.stlabunifi.deepgreen.dl4j.core.model;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.learning.config.Adam;

public class VGG16NativeModel {

	public static ComputationGraph buildVGG16(int numClasses, int seed, 
			int[] inputShape, double lr) {
		ZooModel<?> vgg16Zoo = VGG16.builder()
				.numClasses(numClasses)
				.seed(seed)
				.inputShape(inputShape)
				.updater(new Adam(lr))
				.build();
		
		ComputationGraph vgg16 = vgg16Zoo.init();
		ModelInspector.printGraphDetails(vgg16);
		
		return vgg16;
	}

}
