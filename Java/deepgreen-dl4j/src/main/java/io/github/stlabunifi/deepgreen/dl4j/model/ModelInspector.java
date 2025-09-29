package io.github.stlabunifi.deepgreen.dl4j.model;

import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class ModelInspector {

	public static void printGraphSummary(ComputationGraph graph) {
		System.out.println(graph.summary());
	}

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

		System.out.println("Total number of parameters: " + graph.numParams());
		System.out.println("\n======================================");
	}

	public static void printWeightInitializer(ComputationGraph graph) {
		graph.getConfiguration().getVertices().forEach((name, vertex) -> {
			if (vertex instanceof LayerVertex) {
				org.deeplearning4j.nn.conf.layers.Layer lconf = ((LayerVertex) vertex).getLayerConf().getLayer();
				if (lconf instanceof BaseLayer) {
					BaseLayer bl = (BaseLayer) lconf;
					System.out.println(name + " -> " + bl.getWeightInitFn());
				} else {
					System.out.println(name + " -> layer without trainable weights");
				}
			} else {
				System.out.println(name + " -> not a layer");
			}
		});
	}


	public static void printModelHierarchy(MultiLayerNetwork model) {
		System.out.println("=== MultiLayerNetwork model hierarchy ===");
		System.out.println("Number of layers: " + model.getnLayers());

		MultiLayerConfiguration conf = model.getLayerWiseConfigurations();
		for (int i = 0; i < model.getnLayers(); i++) {
			Layer layer = model.getLayer(i);
			NeuralNetConfiguration layerConf = conf.getConf(i);

			System.out.println("Layer " + i + ":");
			System.out.println("  Runtime class: " + layer.getClass().getName());
			System.out.println("  Config class:  " + layerConf.getLayer().getClass().getName());
			System.out.println("  Layer config: " + layerConf.getLayer().toString());

			//System.out.println("  Trainable: " + ((Object) layer.conf()).isTrainable());
			System.out.println("------------------------------------------");
		}
	}
};
