package io.github.stlabunifi.deepgreen.dl4j.expt.test;

import org.nd4j.linalg.factory.Nd4j;

public class TestExpt {

	public static void main(String[] args)
	{
		System.out.println("Hello, World from " + System. getProperty("os.name"));
		System.out.println("Backend in uso: " + Nd4j.getBackend().getClass().getSimpleName());
		System.out.println("Info: " + Nd4j.getExecutioner().getClass().getSimpleName());
	}

}
