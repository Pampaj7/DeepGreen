package io.github.stlabunifi.deepgreen.dl4j.expt.vgg16;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class PythonHandler {
	
	public final static String python_path = "C:\\Users\\marco_u3rv1hf\\anaconda3\\envs\\tf2env\\python.exe";
	
	/**
	 * Use with:
	 * try {
	 * 		...
	 * } catch (Exception e) {
	 * 		e.printStackTrace();
	 * }
	 * @param modelH5Filename 
	 */
	public static void runModelGenerationScript(String scriptPath, String modelH5Filename) throws IOException, InterruptedException {
		String pythonPath = VenvManager.getTF2env();
		
		ProcessBuilder pb = new ProcessBuilder(pythonPath, scriptPath, modelH5Filename);
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

class VenvManager {

	/**
	 * Important: require Conda to run.
	 * 
	 * Create the Python Virtual Environment, called tf2env, defined with 
	 * Python 3.9, TensorFlow (and Keras) 2.11.0, and Numpy 1.23.5, 
	 * i.e. the maximum versions compatible with DL4J importing system.
	 * 
	 * @return the Python path to tf2env virtual environment
	 */
	public static String getTF2env() {
		String unixVenvDir = "/home/marcopaglio/tools/Java/miniconda3/envs/tf2env/bin/python";
		String winVenvDir = "C:\\Users\\marco_u3rv1hf\\anaconda3\\envs\\tf2env\\python.exe"; // Percorso del venv

		try {
			// 1. Verifica se l'ambiente esiste
			if (!Files.exists(Paths.get(unixVenvDir)) &&
					!Files.exists(Paths.get(winVenvDir))) {

				System.out.println("TF2env virtual environment not found. Creating...");
				
				runCommand(List.of("conda", "create", "-y", "-n", "tf2env", "python=3.9"));
				runCommand(List.of("conda", "run", "-n", "tf2env", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"));
				runCommand(List.of("conda", "run", "-n", "tf2env", "pip", "install", "tensorflow==2.11", "numpy==1.23.5"));
				
			} else {
				System.out.println("Ambiente virtuale già esistente.");
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		String venvPython = Files.exists(Paths.get(unixVenvDir))
				? Paths.get(unixVenvDir).toString()
				: Paths.get(winVenvDir).toString();

		return venvPython;
	}

	private static void runCommand(List<String> command) throws IOException, InterruptedException {
		System.out.print(" $ ");
		for (String cmd : command) System.out.print(cmd + " ");
		System.out.println();
		
		ProcessBuilder builder = new ProcessBuilder(command);
		builder.inheritIO(); // optional: inherit output to terminal (show output)
		Process process = builder.start();
		int exitCode = process.waitFor();
		if (exitCode != 0) {
			throw new RuntimeException("Command failed: " + String.join(" ", command));
		}
	}
}
