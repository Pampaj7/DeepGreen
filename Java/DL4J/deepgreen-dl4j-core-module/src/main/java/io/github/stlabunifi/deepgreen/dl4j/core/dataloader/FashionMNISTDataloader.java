package io.github.stlabunifi.deepgreen.dl4j.core.dataloader;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import java.io.File;

public class FashionMNISTDataloader {

	private static final int HEIGHT = 28;
	private static final int WIDTH = 28;
	private static final int CHANNELS = 1;
	private static final int NUM_CLASSES = 10;


	public static DataSetIterator loadData(String datasetPath, int batchSize, boolean isTrain, boolean shuffle) throws Exception {
		// Choose correct path
		String folder = isTrain ? "train" : "test";
		File dataDir = new File(datasetPath, folder);
		
		return PNGDataloader.loadPNGData(dataDir, batchSize, HEIGHT, WIDTH, CHANNELS, NUM_CLASSES, shuffle);
	}

	public static DataSetIterator loadDataAndTransform(String datasetPath, int batchSize, boolean isTrain, boolean shuffle,
			int transformedHeight, int transformedWidth, int transformedChannels) throws Exception {
		// Choose correct path
		String folder = isTrain ? "train" : "test";
		File dataDir = new File(datasetPath, folder);

		return PNGDataloader.loadPNGData(dataDir, batchSize, transformedHeight,
				transformedWidth, transformedChannels, NUM_CLASSES, shuffle);
	}
}