package io.github.stlabunifi.deepgreen.dl4j.core.dataloader;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.Random;

public class FashionMNISTDataloader {

	private static final int HEIGHT = 28;
	private static final int WIDTH = 28;
	private static final int CHANNELS = 1;
	private static final int NUM_CLASSES = 10;
	private static final int RNG_SEED = 123;

	private static final int TRANSFORMED_HEIGHT = 32;
	private static final int TRANSFORMED_WIDTH = 32;
	private static final int TRANSFORMED_CHANNELS = 3;


	public static DataSetIterator loadData(String datasetPath, int batchSize, boolean isTrain) throws Exception {
		// Choose correct path
		String folder = isTrain ? "train" : "test";
		File dataDir = new File(datasetPath, folder);

		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		FileSplit fileSplit = new FileSplit(dataDir, NativeImageLoader.ALLOWED_FORMATS, new Random(RNG_SEED));
		RecordReader recordReader = new ImageRecordReader(
				TRANSFORMED_HEIGHT,
				TRANSFORMED_WIDTH,
				TRANSFORMED_CHANNELS,
				labelMaker);
		recordReader.initialize(fileSplit);

		// Create DataSetIterator
		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, NUM_CLASSES);

		return dataIter;
	}
}