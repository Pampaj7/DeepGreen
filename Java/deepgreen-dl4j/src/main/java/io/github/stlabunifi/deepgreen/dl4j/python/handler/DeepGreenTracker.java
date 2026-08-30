package io.github.stlabunifi.deepgreen.dl4j.python.handler;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import java.util.Map;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Bridge to the shared measurement helper, tools/deepgreen_tracker.py.
 *
 * <p>Replaces PythonTrackerHandler + resources/tracker/tracker_control.py, which
 * built their own EmissionsTracker with their own defaults while the Rust, R,
 * C++ and MATLAB stacks each did the same with different settings: five copies,
 * four configurations. Every ecosystem now shares one bridge and one run
 * contract, so the instrument is identical by construction.
 *
 * <p>START and STOP are synchronous. The Rust stack fired the command into the
 * pipe and began computing immediately; its inference blocks recorded 0 J and
 * its training blocks were underestimated by 24% as a result.
 */
public final class DeepGreenTracker implements AutoCloseable {

    private final Process process;
    private final BufferedWriter in;
    private final BufferedReader out;

    private DeepGreenTracker(Process process) throws IOException {
        this.process = process;
        this.in = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
        this.out = new BufferedReader(new InputStreamReader(process.getInputStream()));
        await("tracker ready");
    }

    /** Launch the shared bridge with the interpreter named by the run contract. */
    public static DeepGreenTracker start() throws IOException {
        Path repoRoot = Paths.get("").toAbsolutePath();
        Path bridge = repoRoot.resolve("tools").resolve("deepgreen_tracker.py");
        if (!Files.exists(bridge)) {
            throw new IOException("shared measurement bridge not found at " + bridge);
        }
        String python = System.getenv("DEEPGREEN_PYTHON");
        if (python == null || python.isBlank()) {
            Path venv = repoRoot.resolve(".venv-deepgreen").resolve("bin").resolve("python");
            if (!Files.exists(venv)) {
                throw new IOException(
                    "no measurement interpreter: set DEEPGREEN_PYTHON. Refusing to fall back "
                    + "to PATH python, which would make the CodeCarbon version an ambient "
                    + "property of the shell.");
            }
            python = venv.toString();
        }
        ProcessBuilder pb = new ProcessBuilder(python, bridge.toString(), "--daemon");
        pb.redirectErrorStream(false);
        return new DeepGreenTracker(pb.start());
    }

    private void await(String token) throws IOException {
        String line;
        while ((line = out.readLine()) != null) {
            if (line.contains(token)) {
                return;
            }
        }
        throw new IOException("measurement bridge closed while waiting for '" + token + "'");
    }

    private void send(String command, String ack) throws IOException {
        in.write(command);
        in.write("\n");
        in.flush();
        await(ack);
    }

    public void startPhase(String phase, int epoch) throws IOException {
        send("START " + phase + " " + epoch, "START");
    }

    public void stopPhase() throws IOException {
        send("STOP", "STOP");
    }

    /**
     * Record per-epoch model quality. Every ecosystem computed accuracy in the
     * first campaign and every one of them only printed it, so energy could
     * never be normalised by the useful work produced.
     */
    public void metric(int epoch, double trainLoss, double testLoss, double testAcc)
            throws IOException {
        send(String.format(Locale.ROOT,
                "METRIC epoch=%d train_loss=%.6f test_loss=%.6f test_acc=%.4f",
                epoch, trainLoss, testLoss, testAcc), "METRIC");
    }

    @Override
    public void close() {
        try {
            in.write("EXIT\n");
            in.flush();
            process.waitFor();
        } catch (IOException | InterruptedException e) {
            process.destroy();
        }
    }

    // ---- the shared run contract -------------------------------------------

    private static long envLong(String key, long fallback) {
        String v = System.getenv(key);
        if (v == null || v.isBlank()) {
            return fallback;
        }
        try {
            return Long.parseLong(v.trim());
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    public static int repetition() {
        return (int) envLong("DEEPGREEN_REP", 0);
    }

    public static long seed() {
        return envLong("DEEPGREEN_SEED", 1000 + repetition());
    }

    public static int epochs() {
        return (int) envLong("DEEPGREEN_EPOCHS", 30);
    }

    public static String dataRoot() {
        String v = System.getenv("DEEPGREEN_DATA");
        return (v == null || v.isBlank()) ? "data" : v;
    }

    private static final java.util.Set<String> BATCHNORM_BUFFERS =
            java.util.Set.of("mean", "var", "log_var", "log10stdev");

    /**
     * Refuse to train a network that is not the one the study is comparing.
     *
     * <p>VGG-16 ran as four different networks across the seven stacks --
     * 134,670,244 parameters in the LibTorch lineage against 14,765,988 in JAX,
     * a 9.1x range -- while the specification claimed parameter counts were
     * checked against the exported module. Nothing checked them anywhere. The
     * campaign driver now carries models/MANIFEST.json's count in
     * DEEPGREEN_EXPECTED_PARAMS so every stack can check without a JSON parser
     * of its own.
     *
     * <p>Counted the way torch counts, which is not the way DL4J does: torch's
     * {@code model.parameters()} covers learnable weights, while
     * {@code numParams()} here also includes each BatchNormalization's running
     * {@code mean} and {@code log10stdev}. For ResNet-18 that is 9,600 values across
     * 4,800 normalised channels, and comparing the two conventions directly
     * makes an identical network look wrong by exactly that. Keras has the same
     * discrepancy, for the same reason.
     */
    public static void assertParameters(ComputationGraph graph) {
        String want = System.getenv("DEEPGREEN_EXPECTED_PARAMS");
        if (want == null || want.isBlank()) {
            return;
        }
        long got = 0;
        for (org.deeplearning4j.nn.api.Layer layer : graph.getLayers()) {
            Map<String, INDArray> params = layer.paramTable();
            if (params == null) {
                continue;
            }
            for (Map.Entry<String, INDArray> e : params.entrySet()) {
                if (BATCHNORM_BUFFERS.contains(e.getKey())) {
                    continue;   // buffers in torch, parameters here
                }
                got += e.getValue().length();
            }
        }
        long expected = Long.parseLong(want.trim());
        if (got == expected) {
            System.out.printf(Locale.ROOT,
                    "[deepgreen] %d parameters, as exported%n", got);
            return;
        }
        String msg = String.format(Locale.ROOT,
                "this stack has %d parameters; models/MANIFEST.json says %d "
                + "(difference %+d). Comparing its energy with the others would "
                + "compare models rather than ecosystems.", got, expected, got - expected);
        if ("1".equals(System.getenv("DEEPGREEN_ALLOW_MODEL_DRIFT"))) {
            System.out.println("[deepgreen] WARNING " + msg);
            return;
        }
        throw new IllegalStateException(
                msg + " Set DEEPGREEN_ALLOW_MODEL_DRIFT=1 to override.");
    }
}
