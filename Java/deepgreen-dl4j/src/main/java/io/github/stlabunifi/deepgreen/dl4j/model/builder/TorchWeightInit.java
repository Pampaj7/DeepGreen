package io.github.stlabunifi.deepgreen.dl4j.model.builder;

import org.deeplearning4j.nn.weights.IWeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * The initialisers torchvision uses, so this stack starts where the others do.
 *
 * <p>This is the setting the campaign's most surprising result turned out to
 * rest on. Deeplearning4j was configured with {@code WeightInit.RELU}: a
 * truncated normal with standard deviation sqrt(2 / fan_in). torchvision uses
 * {@code kaiming_normal_(mode='fan_out', nonlinearity='relu')}, an untruncated
 * normal with sqrt(2 / fan_out). For ResNet-18's first convolution -- 7x7,
 * 3 in, 64 out -- those are 0.116 and 0.025: the Java stack's weights started
 * <b>4.6x wider</b> than every other stack's.
 *
 * <p>Why it matters beyond tidiness. Twelve of the campaign's 105 VGG-16 runs
 * converged to exactly chance accuracy and sat at a loss of ln(N) for all
 * thirty epochs, and the manuscript reported the collapse rate as a property of
 * the ecosystem. Holding framework, optimiser, learning rate and data order
 * fixed and varying only the initialiser gives 0 of 6 collapses under He, 2 of
 * 6 under Glorot and 4 of 6 under Xavier -- and Deeplearning4j, the one stack
 * with a hand-rolled initialiser, carried 5 of the 12. What was being read as
 * an ecosystem effect tracks the initialiser, and no built-in DL4J enum
 * expresses torchvision's choice: RELU is fan_in, VAR_SCALING_NORMAL_FAN_OUT
 * has scale 1 rather than 2, and XAVIER lands near the right number for the
 * wrong reason.
 */
public final class TorchWeightInit {

    private TorchWeightInit() {
    }

    /**
     * {@code kaiming_normal_(mode='fan_out', nonlinearity='relu')}: an
     * untruncated normal with standard deviation sqrt(2 / fan_out). What
     * torchvision applies to every convolution in ResNet-18 and VGG-16.
     */
    public static IWeightInit convolution() {
        return new KaimingNormalFanOut();
    }

    /** Named rather than a lambda: ModelSerializer writes the configuration at
     *  the end of every run, and a lambda has no stable type for Jackson. */
    public static class KaimingNormalFanOut implements IWeightInit {
        private static final long serialVersionUID = 1L;

        @Override
        public INDArray init(double fanIn, double fanOut, long[] shape, char order,
                             INDArray paramView) {
            // fan_out from the shape, not from the argument. DL4J divides a
            // convolution's fan_out by the stride product; torch's
            // _calculate_fan_in_and_fan_out does not. For ResNet-18's 7x7
            // stride-2 stem that is a factor of four in the variance, so the
            // stride-2 convolutions came out exactly 2x too wide while every
            // stride-1 convolution matched -- which is precisely the kind of
            // difference that hides until someone measures it.
            double effectiveFanOut = fanOut;
            if (shape.length == 4) {
                effectiveFanOut = (double) shape[0] * shape[2] * shape[3];
            }
            double stddev = Math.sqrt(2.0 / effectiveFanOut);
            INDArray w = Nd4j.randn(paramView.dataType(), paramView.length()).muli(stddev);
            paramView.assign(w);
            return paramView.reshape(order, shape);
        }
    }

    /**
     * PyTorch's {@code nn.Linear} default: {@code kaiming_uniform_(a=sqrt(5))},
     * which reduces to U(-1/sqrt(fan_in), +1/sqrt(fan_in)). The classifier of
     * the canonical head is built from fresh {@code nn.Linear} layers, so this
     * is what they get in the exported module.
     */
    public static IWeightInit dense() {
        return new LinearUniformFanIn();
    }

    /** See {@link KaimingNormalFanOut} on why this is not a lambda. */
    public static class LinearUniformFanIn implements IWeightInit {
        private static final long serialVersionUID = 1L;

        @Override
        public INDArray init(double fanIn, double fanOut, long[] shape, char order,
                             INDArray paramView) {
            double bound = 1.0 / Math.sqrt(fanIn);
            INDArray w = Nd4j.rand(paramView.dataType(), paramView.length())
                             .muli(2.0 * bound).subi(bound);
            paramView.assign(w);
            return paramView.reshape(order, shape);
        }
    }
}
