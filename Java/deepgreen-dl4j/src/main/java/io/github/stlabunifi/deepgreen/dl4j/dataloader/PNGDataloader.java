package io.github.stlabunifi.deepgreen.dl4j.dataloader;

import java.io.File;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class PNGDataloader {

	private static final int RNG_SEED = 123;

	static DataSetIterator loadPNGData(File dataDir, int batchSize,
			int height, int width, int channels, int numClasses, boolean shuffle) throws Exception {

		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		
		Random rng = shuffle ? new Random(RNG_SEED) : null;
		FileSplit fileSplit = new FileSplit(dataDir, NativeImageLoader.ALLOWED_FORMATS, rng);
		
		RecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
		recordReader.initialize(fileSplit);

		// Create DataSetIterator
		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
		DataSetIterator asyncIter = new AsyncDataSetIterator(dataIter, 2); // same as num_workers=2 in PyTorch
		
		return asyncIter;
	}

}
