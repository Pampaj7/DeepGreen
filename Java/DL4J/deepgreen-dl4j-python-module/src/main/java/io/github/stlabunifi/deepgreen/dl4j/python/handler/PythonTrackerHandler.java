package io.github.stlabunifi.deepgreen.dl4j.python.handler;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class PythonTrackerHandler {
	private static String defaultTracker = "src/main/resources/tracker/tracker_control.py";
	private static String readyWord = "READY"; // used as synchronization barrier

	private Process process;
	private ExecutorService executor;
	private String pythonPath;
	private String trackerPath;
	private String outputFolder;

	public PythonTrackerHandler() throws FileNotFoundException {
		this("");
	}

	public PythonTrackerHandler(String outputDir) throws FileNotFoundException {
		process = null;
		executor = null;
		pythonPath = "Linux".equals(System.getProperty("os.name")) ? "python3" : "python";
		
		Path filePath = Paths.get(defaultTracker);
		if (!Files.exists(filePath) || !Files.isRegularFile(filePath)) {
			throw new FileNotFoundException("There is no Python tracker file at: " + defaultTracker);
		}
		trackerPath = filePath.toAbsolutePath().toString();
		outputFolder = outputDir;
	}

	public void startTracker(String outputFile) throws IOException, InterruptedException {
		List<String> commands = List.of(pythonPath, trackerPath);
		if (!outputFolder.isEmpty())
			commands = List.of(pythonPath, trackerPath, outputFolder);
		
		ProcessBuilder pb = new ProcessBuilder(commands);
		this.process = pb.start();
		this.executor = Executors.newFixedThreadPool(2);

		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
		BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
		BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));

		CountDownLatch readySignal = new CountDownLatch(1);
		// Start reading output and error in separate threads
		executor.submit(() -> readStream(reader, "Python output", readySignal));
		executor.submit(() -> readStream(errorReader, "Python error", null));

		// Send the start command to the Python script
		writer.write("start " + outputFile + " " + readyWord + "\n");
		writer.flush();
		
		// Ensure the tracker starts
		readySignal.await();
	}

	public void stopTracker() throws IOException, InterruptedException {
		if (process != null) {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));

			// Send the stop command to the Python script
			writer.write("stop\n");
			writer.flush();

			// Send the exit command to the Python script
			writer.write("exit\n");
			writer.flush();

			stopPythonScript();
		}
	}

	private void readStream(BufferedReader reader, String streamName, CountDownLatch latch) {
		try {
			String line;
			while ((line = reader.readLine()) != null) {
				System.out.println(streamName + ": " + line);
				if (latch != null && readyWord.equals(line.trim()))
					latch.countDown();
			}
		} catch (IOException e) {
			System.err.println("Error reading " + streamName + ": " + e.getMessage());
		}
	}

	private void stopPythonScript() throws InterruptedException, IOException {
		if (process != null) {
			if (!process.waitFor(30, TimeUnit.SECONDS)) {
				System.err.println("Python process did not finish in time. Forcibly terminating.");
				process.destroyForcibly();
			}

			executor.shutdown();
			if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
				System.err.println("Stream reading did not complete in time.");
				executor.shutdownNow();
			}

			int exitCode = process.exitValue();
			System.out.println("Python process exited with code: " + exitCode);
		}
	}

	public void cleanup() {
		if (executor != null) {
			executor.shutdownNow(); // Ensure executor is shut down even if an exception occurs
		}
		if (process != null) {
			process.destroyForcibly(); // Ensure the process is terminated
		}
	}
}
