package io.github.stlabunifi.deepgreen.dl4j.core.dataloader;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import java.io.File;

public class TinyImageNetDataloader {

	private static final int HEIGHT = 64;
	private static final int WIDTH = 64;
	private static final int CHANNELS = 3;
	private static final int NUM_CLASSES = 200;
	private static final int RNG_SEED = 123;

	public static DataSetIterator loadData(String datasetPath, int batchSize, boolean isTrain) throws Exception {
		// Choose correct path
		String folder = isTrain ? "train" : "val";
		File dataDir = new File(datasetPath, folder);

		return PNGDataloader.loadPNGData(dataDir, batchSize, HEIGHT, WIDTH, CHANNELS, NUM_CLASSES, RNG_SEED);
	}
}

