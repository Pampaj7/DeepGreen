package io.github.stlabunifi.deepgreen.dl4j.dataloader;

import java.io.File;
import java.util.Random;

import io.github.stlabunifi.deepgreen.dl4j.python.handler.DeepGreenTracker;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class PNGDataloader {

	/**
	 * The data order comes from the run contract, not from this file.
	 *
	 * <p>It was {@code private static final int RNG_SEED = 123}, so all five
	 * repetitions of every Java configuration shuffled identically no matter
	 * what seed the campaign handed them -- while the conformance check
	 * reported "5 of 5 distinct seeds", having asked the campaign *planner*
	 * rather than any stack whether the seed was used. Java's five repetitions
	 * were therefore not five independent draws of the data order, which is one
	 * of the two things a repetition is for.
	 */
	private static long dataSeed() {
		return DeepGreenTracker.seed();
	}

	static DataSetIterator loadPNGData(File dataDir, int batchSize,
			int height, int width, int channels, int numClasses, boolean shuffle) throws Exception {

		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		
		Random rng = shuffle ? new Random(dataSeed()) : null;
		FileSplit fileSplit = new FileSplit(dataDir, NativeImageLoader.ALLOWED_FORMATS, rng);
		
		RecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
		recordReader.initialize(fileSplit);

		// Create DataSetIterator
		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
		DataSetIterator asyncIter = new AsyncDataSetIterator(dataIter, 2); // same as num_workers=2 in PyTorch
		
		return asyncIter;
	}

}
