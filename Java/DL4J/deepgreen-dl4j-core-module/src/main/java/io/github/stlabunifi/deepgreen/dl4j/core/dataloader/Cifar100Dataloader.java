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

public class Cifar100Dataloader {
	private static final int HEIGHT = 32;
	private static final int WIDTH = 32;
	private static final int CHANNELS = 3;
	private static final int NUM_CLASSES = 100;
	private static final int RNG_SEED = 123;

	public static DataSetIterator loadData(String datasetPath, int batchSize, boolean isTrain) throws Exception {
		// Choose correct path
		String folder = isTrain ? "train" : "test";
		File dataDir = new File(datasetPath, folder);

		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		FileSplit fileSplit = new FileSplit(dataDir, NativeImageLoader.ALLOWED_FORMATS, new Random(RNG_SEED));
		RecordReader recordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
		recordReader.initialize(fileSplit);

		// Create DataSetIterator
		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, NUM_CLASSES);

		return dataIter;
	}
}
