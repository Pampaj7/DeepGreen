package io.github.stlabunifi.deepgreen.dl4j.expt;

public class TimeToSleep {

	static final int idleTime = 30; // seconds

	public static void main(String[] args) {
		try {
			
			System.out.print("Take a break of " + idleTime + " seconds... ");
			Thread.sleep(idleTime * 1000);
			System.out.println("finished!");
			
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

	}

}
