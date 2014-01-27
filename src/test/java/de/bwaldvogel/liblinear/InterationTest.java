package de.bwaldvogel.liblinear;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import org.junit.Test;

public class InterationTest {

	@Test
    public void testIteration() throws IOException{
    	Problem prob = LinearTest.createRandomProblem(2);
    	
    	SolverType solver = SolverType.L1R_L2LOSS_SVC;
    	double C = 1.0;
    	double eps = 0.01;
    	
    	Parameter parameter = new Parameter(solver, C, eps);
    	Model model = Linear.train(prob, parameter);
    	
    	model.save(new File("first_round.lr"));
    	
    	Problem prob_col = Linear.transpose(prob);
    	
    	double[] w = new double[prob.n];
    	double[] b = new double[prob.l];
    	double Cp = 0.;
    	double Cn = 0.;
    	
    	Linear.solve_l1r_l2_svc_iteration(prob_col, w, b, eps, Cp, Cn);
    	
    	System.out.println(Arrays.toString(w));
    } 
}
