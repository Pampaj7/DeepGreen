package io.github.stlabunifi.deepgreen.dl4j.model;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Test loss and accuracy in one pass, as every other ecosystem computes them.
 *
 * <p>This stack recorded {@code Double.NaN} for test loss in all 900 of its
 * epoch rows, because DL4J's {@link Evaluation} does not expose a loss and the
 * training loop did not compute one. It was the single failing conformance
 * check, and the cost was not cosmetic: the discriminator that separates an
 * optimisation collapse from a data-pipeline defect uses test loss as one of
 * its two arms, so for the stack contributing five of the campaign's twelve
 * collapses it had one arm. Worse, the expression ANDs the two arms and any
 * comparison with NaN is false, so the conjunction could never fire -- the
 * manuscript described a graceful degradation to a single arm that the code did
 * not implement, and Java's sensitivity to pipeline defects was zero rather
 * than reduced.
 *
 * <p>One pass, not two: the other six stacks compute loss and accuracy over the
 * same forward pass inside the measured window, and evaluating twice here would
 * make Java's inference blocks measure twice the work.
 */
public final class Evaluator {

    private Evaluator() {
    }

    /** Mean test loss and accuracy in per cent, over the whole iterator. */
    public static double[] lossAndAccuracy(ComputationGraph graph, DataSetIterator test) {
        Evaluation evaluation = new Evaluation();
        double weightedLoss = 0.0;
        long examples = 0;

        test.reset();
        while (test.hasNext()) {
            DataSet batch = test.next();
            // score(..., false) is the inference-mode loss: dropout off, batch
            // normalisation on its running statistics, matching model.eval() and
            // no_grad in the torch stacks and Keras's evaluate().
            double batchLoss = graph.score(batch, false);
            long n = batch.getLabels().size(0);
            weightedLoss += batchLoss * n;
            examples += n;
            evaluation.eval(batch.getLabels(),
                            graph.outputSingle(false, batch.getFeatures()));
        }

        double loss = examples > 0 ? weightedLoss / examples : Double.NaN;
        return new double[] {loss, evaluation.accuracy() * 100.0};
    }
}
