package io.github.stlabunifi.deepgreen.dl4j.core.model;

import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.graph.ComputationGraph;

public class ModelInspector {

	public static void printGraphDetails(ComputationGraph graph) {
		ComputationGraphConfiguration conf = graph.getConfiguration();

		System.out.println("===== COMPUTATION GRAPH DETAILS =====\n");

		// INPUTS
		System.out.println("Inputs:");
		for (String input : conf.getNetworkInputs()) {
			System.out.println(" - " + input);
		}

		// OUTPUTS
		System.out.println("\nOutputs:");
		for (String output : conf.getNetworkOutputs()) {
			System.out.println(" - " + output);
		}

		// VERTICES
		System.out.println("\nVertices:");
		for (Map.Entry<String, GraphVertex> entry : conf.getVertices().entrySet()) {
			String name = entry.getKey();
			GraphVertex vertex = entry.getValue();
			System.out.println(" - Name: " + name);
			System.out.println("   Class: " + vertex.getClass().getSimpleName());
			System.out.println("   Vertex: " + vertex.toString());
			List<String> inputsToVertex = conf.getVertexInputs().get(name);
			if (inputsToVertex != null && inputsToVertex.size() > 0) {
				System.out.print("   Inputs: ");
				for (String in : inputsToVertex) {
					System.out.print(in + " ");
				}
				System.out.println();
			}
		}

		System.out.println("\n======================================");
	}
};
