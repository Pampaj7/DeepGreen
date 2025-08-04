package io.github.stlabunifi.deepgreen.dl4j.python.handler;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import io.github.stlabunifi.deepgreen.dl4j.python.manager.VenvManager;

public class PythonCommandHandler {

	/**
	 * To be used inside a try-catch block, as follows:
	 * 	try {
	 * 			...
	 * 	} catch (Exception e) {
	 * 		e.printStackTrace();
	 * 	}
	 */
	public static void runGenerateModelScript(String scriptPath, String modelH5Filename, int numClasses) throws IOException, InterruptedException {
		String pythonPath = VenvManager.getTF2env();
		
		ProcessBuilder pb = new ProcessBuilder(pythonPath, scriptPath, modelH5Filename, String.valueOf(numClasses));
		pb.redirectErrorStream(true);
		Process process = pb.start();

		BufferedReader in = new BufferedReader(new InputStreamReader(process.getInputStream()));
		String line;
		while ((line = in.readLine()) != null) {
			System.out.println(line);
		}
		in.close();
		process.waitFor();
	}

	public static void runDownloadDatasetScript(String scriptPath, String outputDir) throws IOException, InterruptedException {
		String pythonPath = "python";
		if ("Linux".equals(System.getProperty("os.name"))) pythonPath = "python3";
		
		ProcessBuilder pb = new ProcessBuilder(pythonPath, scriptPath, outputDir);
		pb.redirectErrorStream(true);
		Process process = pb.start();

		BufferedReader in = new BufferedReader(new InputStreamReader(process.getInputStream()));
		String line;
		while ((line = in.readLine()) != null) {
			System.out.println(line);
		}
		in.close();
		process.waitFor();
	}

}
