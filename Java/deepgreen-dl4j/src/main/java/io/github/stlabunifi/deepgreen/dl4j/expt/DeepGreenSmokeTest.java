package io.github.stlabunifi.deepgreen.dl4j.expt;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Smoke test for spec S5: this stack must be proven to run on the accelerator.
 *
 * A framework that cannot reach the GPU falls back to the CPU and says so only
 * in a log line; the run then completes normally and the CPU cost is attributed
 * to the ecosystem. Nothing in the first campaign recorded device placement, so
 * it cannot be checked after the fact.
 *
 *   mvn -q exec:java -Dexec.mainClass=io.github.stlabunifi.deepgreen.dl4j.expt.DeepGreenSmokeTest
 */
public class DeepGreenSmokeTest {

    public static void main(String[] args) {
        String backend = Nd4j.getBackend().getClass().getSimpleName();
        System.out.println("ND4J_BACKEND " + backend);
        System.out.println("ND4J_DEVICES " + Nd4j.getAffinityManager().getNumberOfDevices());

        INDArray a = Nd4j.rand(2048, 2048);
        INDArray c = a.mmul(a);
        System.out.println("MATMUL_OK " + (c.sumNumber().doubleValue() != 0.0));

        boolean cuda = backend.toLowerCase().contains("cuda")
                || backend.toLowerCase().contains("jcublas");
        System.out.println("GPU_VISIBLE " + cuda);
        if (!cuda) {
            System.err.println(
                "ND4J is on the CPU backend. Measuring now would attribute a CPU "
                + "fallback to the ecosystem.");
            System.exit(1);
        }
    }
}
