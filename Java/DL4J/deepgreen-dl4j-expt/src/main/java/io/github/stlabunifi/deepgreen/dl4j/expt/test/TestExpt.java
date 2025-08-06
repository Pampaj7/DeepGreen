package io.github.stlabunifi.deepgreen.dl4j.expt.test;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestExpt {

	public static void main(String[] args)
	{
		System.out.println("Hello, World from " + System. getProperty("os.name"));
		System.out.println("Backend in uso: " + Nd4j.getBackend().getClass().getSimpleName());
		System.out.println("Info: " + Nd4j.getExecutioner().getClass().getSimpleName());
		System.out.println("Device in use: " + Nd4j.getAffinityManager().getDeviceForCurrentThread()); // restituisce 0 se CPU
		INDArray a = Nd4j.create(new float[] { 1, 2, 3 });
		INDArray b = Nd4j.create(new float[] { 4, 5, 6 });
		INDArray c = a.add(b);
		System.out.println(c);
	}

}
