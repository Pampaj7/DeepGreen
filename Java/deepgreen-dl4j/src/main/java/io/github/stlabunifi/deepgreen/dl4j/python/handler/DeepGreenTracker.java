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
}
