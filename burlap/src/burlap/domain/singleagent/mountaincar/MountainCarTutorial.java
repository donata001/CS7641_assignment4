package burlap.domain.singleagent.mountaincar;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.vfa.rbf.DistanceMetric;
import burlap.behavior.singleagent.vfa.rbf.functions.GaussianRBF;
import burlap.behavior.singleagent.vfa.rbf.RBFFeatureDatabase;
import burlap.behavior.singleagent.learning.GoalBasedRF;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldStateParser;
import burlap.domain.singleagent.mountaincar.MountainCar;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.domain.singleagent.mountaincar.MCRandomStateGenerator;
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.auxiliary.StateParser;
import burlap.behavior.singleagent.vfa.common.ConcatenatedObjectFeatureVectorGenerator;
import burlap.behavior.singleagent.vfa.fourier.FourierBasis;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.singleagent.planning.QComputablePlanner;
import burlap.behavior.singleagent.planning.StateConditionTest;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.mountaincar.MountainCarVisualizer;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.core.State;
import burlap.oomdp.singleagent.common.VisualActionObserver;
import burlap.oomdp.visualizer.Visualizer;
import burlap.behavior.singleagent.auxiliary.StateGridder;

import java.util.List;

import burlap.behavior.singleagent.vfa.rbf.metrics.EuclideanDistance;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.behavior.statehashing.DiscretizingStateHashFactory;

public class MountainCarTutorial {
	
	GridWorldDomain 			gwdg;
	Domain						domain;
	StateParser 				sp;
	RewardFunction 				rf;
	TerminalFunction			tf;
	StateConditionTest			goalCondition;
	State 						initialState;
	DiscretizingStateHashFactory	hashingFactory;
	MountainCar                 mcGen;
	SARSData                    dataset;


	public static void main(String [] args){
		
		MountainCarTutorial example = new MountainCarTutorial();
		
		String outputPath = "output2/"; 
		example.PolicyMCLSPIFB(outputPath);
		//example.PolicyMCLSPIRBF(outputPath);
		//example.ValueIterationExample(outputPath);
		//example.PolicyIterationExample(outputPath);
		
	}
	
	
	public MountainCarTutorial(){
		
		mcGen = new MountainCar();
		domain = mcGen.generateDomain();
		tf = mcGen.new ClassicMCTF();
		rf = new GoalBasedRF(tf, 100);
		sp = new MountainCarStateParser(domain); 
		
		
		StateGenerator rStateGen = new MCRandomStateGenerator(domain);
		SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain);
		dataset = collector.collectNInstances(rStateGen, rf, 1000, 20, tf, null);

		
		//set up the initial state of the task
		initialState = mcGen.getCleanState(domain);

			
		//set up the state hashing system
		hashingFactory = new DiscretizingStateHashFactory();
		
	}	

	public void PolicyMCLSPIFB(String outputPath){
		// MCLSPIFB
		System.out.println("Begin MCLSPIFB...");
		
						
		ConcatenatedObjectFeatureVectorGenerator featureVectorGenerator = 
	               new ConcatenatedObjectFeatureVectorGenerator(true, MountainCar.CLASSAGENT);
		
		FourierBasis fb = new FourierBasis(featureVectorGenerator, 4);
		LSPI lspi = new LSPI(domain, rf, tf, 0.99, fb);
		lspi.setDataset(dataset);

		lspi.runPolicyIteration(30, 1e-7);
		
		GreedyQPolicy p = new GreedyQPolicy(lspi);

		Visualizer v = MountainCarVisualizer.getVisualizer(mcGen);
		VisualActionObserver vexp = new VisualActionObserver(domain, v);
		vexp.initGUI();
		((SADomain)domain).addActionObserverForAllAction(vexp);

		State s = mcGen.getCleanState(domain);
		for(int i = 0; i < 5; i++){
			p.evaluateBehavior(s, rf, tf);
		}
		System.out.println("Finished MCLSPIFB.");
	}
	

		
	public void PolicyMCLSPIRBF(String outputPath){
		// MCLSPIRBF
		System.out.println("Begin MCLSPIRBF...");


		//set up RBF feature database
		RBFFeatureDatabase rbf = new RBFFeatureDatabase(true);
		StateGridder gridder = new StateGridder();
		gridder.gridEntireDomainSpace(domain, 5);
		List<State> griddedStates = gridder.gridInputState(initialState);

		DistanceMetric metric = new EuclideanDistance(
		          new ConcatenatedObjectFeatureVectorGenerator(true, MountainCar.CLASSAGENT));
		for(State g : griddedStates){
			rbf.addRBF(new GaussianRBF(g, metric, .2));
		}

		//notice we pass LSPI our RBF features this time
		LSPI lspi2 = new LSPI(domain, rf, tf, 0.99, rbf);
		lspi2.setDataset(dataset);

		lspi2.runPolicyIteration(30, 1e-7);

		GreedyQPolicy p2 = new GreedyQPolicy(lspi2);

		Visualizer v2 = MountainCarVisualizer.getVisualizer(mcGen);
		VisualActionObserver vexp2 = new VisualActionObserver(domain, v2);
		vexp2.initGUI();
		((SADomain)domain).addActionObserverForAllAction(vexp2);


		for(int j = 0; j < 5; j++){
			p2.evaluateBehavior(initialState, rf, tf);
		
		
		}
				System.out.println("Finished MCLSPIRBF.");
				
	}
	
	
public void ValueIterationExample(String outputPath){
		
		if(!outputPath.endsWith("/")){
			outputPath = outputPath + "/";
		}
		
		
		
		OOMDPPlanner planner = new ValueIteration(domain, rf, tf, 0.99, hashingFactory,
								0.00001, 30);
		planner.planFromState(initialState);
		
		//create a Q-greedy policy from the planner
		Policy p = new GreedyQPolicy((QComputablePlanner)planner);
		
		//record the plan results to a file
		p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "planResult" + "VIEexample", sp);
		
		
	}
	
	
	public void PolicyIterationExample(String outputPath) {

		if(!outputPath.endsWith("/")){
			outputPath = outputPath + "/";
		}
		
		OOMDPPlanner planner = new PolicyIteration(domain, rf, tf, 0.99, hashingFactory,
				0.00001, 20, 50);

		planner.planFromState(initialState);

		Policy p = new GreedyQPolicy((QComputablePlanner)planner);
		
		//record the plan results to a file
		p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "planResult" + "PIEexample", sp);
		
	}
	
						
				
}


