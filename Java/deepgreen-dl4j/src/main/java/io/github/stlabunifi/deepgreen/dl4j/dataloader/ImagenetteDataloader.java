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

/**
 * Imagenette: ten ImageNet classes at ImageNet resolution.
 *
 * <p>The accelerator-saturation cell. Every other dataset in this campaign is
 * 32x32 on disk; this one is 224x224, pre-resized offline by
 * {@code dataloader/download_convert_imagenette.py} (shorter side to 224,
 * centre crop) so that nothing is resized at run time and the seven stacks
 * decode identical pixels.
 *
 * <p>Class directories are the human-readable names, lower case and with
 * spaces ("cassette player", "english springer"). {@link ImageRecordReader}
 * sorts the inferred labels with {@link String#compareTo}, which is code-point
 * order and therefore the same order Python's {@code sorted()}, C++'s
 * {@code std::sort} over the directory names and Rust's {@code sort_by_key}
 * over the OS strings produce:
 * {@code [cassette player, chain saw, church, english springer, french horn,
 * garbage truck, gas pump, golf ball, parachute, tench]}, indices 0..9. The
 * label index of a class is therefore the same here as in every other stack of
 * this cell. The class directories were lower-cased for exactly this reason: a
 * capitalised name sorts before every lower-case one in byte order but not in a
 * locale-aware collation, and a stack that disagreed would train on a
 * permutation of the labels without anything failing. Verified by printing
 * {@code ImageRecordReader.getLabels()} against the real directory, not
 * assumed; the spaces are handled by {@link FileSplit} without quoting.
 *
 * <p>This loader does not delegate to {@link PNGDataloader} for one reason:
 * that class pins the prefetch depth to the literal 2 that the conformance
 * check greps for, and the saturation cell has to be able to raise it
 * (DEEPGREEN_LOADER_THREADS) without touching the twelve runs of the frozen
 * campaign. Everything else -- the label generator, the seeded shuffle, the
 * record reader, the iterator -- is what {@link PNGDataloader} does.
 */
public class ImagenetteDataloader {

	private static final int HEIGHT = 224;
	private static final int WIDTH = 224;
	private static final int CHANNELS = 3;
	private static final int NUM_CLASSES = 10;

	/**
	 * Prefetch depth of the asynchronous iterator: this stack's equivalent of
	 * PyTorch's {@code num_workers}. Default 2, as in
	 * {@link PNGDataloader}; the driver may raise it for this cell, so it is a
	 * named constant read from the run contract rather than a literal inline.
	 */
	public static final int LOADER_THREADS = loaderThreads();

	private static int loaderThreads() {
		String v = System.getenv("DEEPGREEN_LOADER_THREADS");
		if (v == null || v.isBlank()) {
			return 2;
		}
		try {
			int n = Integer.parseInt(v.trim());
			return n > 0 ? n : 2;
		} catch (NumberFormatException e) {
			return 2;
		}
	}

	/**
	 * The data order comes from the run contract, not from this file.
	 * See {@link PNGDataloader} for why that is worth saying out loud.
	 */
	private static long dataSeed() {
		return DeepGreenTracker.seed();
	}

	public static DataSetIterator loadData(String datasetPath, int batchSize, boolean isTrain, boolean shuffle) throws Exception {
		// Choose correct path
		String folder = isTrain ? "train" : "test";
		File dataDir = new File(datasetPath, folder);

		return loadPNGData(dataDir, batchSize, HEIGHT, WIDTH, CHANNELS, NUM_CLASSES, shuffle);
	}

	public static DataSetIterator loadDataAndTransform(String datasetPath, int batchSize, boolean isTrain, boolean shuffle,
			int transformedHeight, int transformedWidth, int transformedChannels) throws Exception {
		// Choose correct path
		String folder = isTrain ? "train" : "test";
		File dataDir = new File(datasetPath, folder);

		return loadPNGData(dataDir, batchSize, transformedHeight,
				transformedWidth, transformedChannels, NUM_CLASSES, shuffle);
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
		DataSetIterator asyncIter = new AsyncDataSetIterator(dataIter, LOADER_THREADS); // same as num_workers=2 in PyTorch

		return asyncIter;
	}

}
