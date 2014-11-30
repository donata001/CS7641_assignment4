package burlap.behavior.singleagent;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import burlap.oomdp.auxiliary.StateParser;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.State;
import burlap.oomdp.singleagent.Action;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.common.NullAction;

import javax.swing.*;


/**
 * This class is used to keep track of all events that occur in an episode. This class should be created by either calling the constructor with the initial state of the episode,
 * or by calling the default constructor and then calling the {@link #initializeEpisideWithInitialState(State)} method to set the initil state of the episode, before recording
 * any transitions. It is then advised that transitions are recorded with the {@link #recordTransitionTo(GroundedAction, State, double)} method, which takes as input
 * the next state to which the agent transitioned, the action applied in the last recorded state, and the reward received fro the transition.
 * <p/>
 * When querying about the state, action, and reward sequences, use the {@link #getState(int)}, {@link #getAction(int)}, and {@link #getReward(int)} methods.
 * These methods take as input the time step of the element you want. Note that t = 0 refers to the initial state step so calling getState(0) and getAction(0)
 * will return the initial state and the action taken in the initial state, respectively. However, rewards are always received in the next time step
 * from the state and action that produced them. Therefore, getReward(0) is undefined. Instead, the first reward received will be at time step 1: getReward(1).
 * 
 * 
 * @author James MacGlashan
 *
 */
public class EpisodeAnalysis {

	/**
	 * The sequence of states observed
	 */
	public List<State>									stateSequence;
	
	/**
	 * The sequence of actions taken
	 */
	public List<GroundedAction>							actionSequence;
	
	/**
	 * The sequence of rewards received. Note the reward stored at index i is the reward received at time step i+1.
	 */
	public List<Double>									rewardSequence;
	
	
	/**
	 * Creates a new EpisodeAnalysis object. Before recording transitions, the {@link #initializeEpisideWithInitialState(State)} method
	 * should be called to set the initial state of the episode.
	 */
	public EpisodeAnalysis(){
		this.initializeDatastructures();
	}
	
	
	/**
	 * Initializes a new EpisodeAnalysis object with the initial state in which the episode started.
	 * @param initialState the initial state of the episode
	 */
	public EpisodeAnalysis(State initialState){
		this.initializeEpisideWithInitialState(initialState);
	}
	
	/**
	 * Initializes this object with the initial state in which the episode started.
	 * @param initialState the initial state of the episode
	 */
	public void initializeEpisideWithInitialState(State initialState){
		this.initializeDatastructures();
		this.stateSequence.add(initialState);
	}
	
	protected void initializeDatastructures(){
		stateSequence = new ArrayList<State>();
		actionSequence = new ArrayList<GroundedAction>();
		rewardSequence = new ArrayList<Double>();
	}
	
	
	/**
	 * Adds a state to the state sequence. In general, it is recommended that {@link #initializeEpisideWithInitialState(State)} method
	 * along with subsequent calls to the {@link #recordTransitionTo(GroundedAction, State, double)} method is used instead, but this
	 * method can be used to manually add a state.
	 * @param s the state to add
	 */
	public void addState(State s){
		stateSequence.add(s);
	}
	
	/**
	 * Adds a GroundedAction to the action sequence. In general, it is recommended that {@link #initializeEpisideWithInitialState(State)} method
	 * along with subsequent calls to the {@link #recordTransitionTo(GroundedAction, State, double)} method is used instead, but this
	 * method can be used to manually add a GroundedAction.
	 * @param ga the GroundedAction to add
	 */
	public void addAction(GroundedAction ga){
		actionSequence.add(ga);
	}
	
	/**
	 * Adds a reward to the reward sequence. In general, it is recommended that {@link #initializeEpisideWithInitialState(State)} method
	 * along with subsequent calls to the {@link #recordTransitionTo(GroundedAction, State, double)} method is used instead, but this
	 * method can be used to manually add a reward.
	 * @param r the reward to add
	 */
	public void addReward(double r){
		rewardSequence.add(r);
	}
	
	
	/**
	 * Records an transition event where the agent applied the usingAction argument in the last
	 * state in this object's state sequence, received reward r, and transitioned to state next. 
	 * @param next the next state to which the agent transitioned
	 * @param usingAction the action the agent used that caused the transition
	 * @param r the reward the agent received for this transition.
	 */
	@Deprecated
	public void recordTransitionTo(State next, GroundedAction usingAction, double r){
		stateSequence.add(next);
		actionSequence.add(usingAction);
		rewardSequence.add(r);
	}
	
	
	/**
	 * Records an transition event where the agent applied the usingAction action in the last
	 * state in this object's state sequence, transitioned to state nextState, and received reward r,. 
	 * @param usingAction the action the agent used that caused the transition
	 * @param nextState the next state to which the agent transitioned
	 * @param r the reward the agent received for this transition.
	 */
	public void recordTransitionTo(GroundedAction usingAction, State nextState, double r){
		stateSequence.add(nextState);
		actionSequence.add(usingAction);
		rewardSequence.add(r);
	}
	
	
	
	/**
	 * Returns the state observed at time step t. t=0 refers to the initial state.
	 * @param t the time step of the episode
	 * @return the state at time step t
	 */
	public State getState(int t){
		return stateSequence.get(t);
	}
	
	/**
	 * Returns the action taken in the state at time step t. t=0 refers to the action taken in the initial state.
	 * @param t the time step of the episode
	 * @return the action taken at time step t
	 */
	public GroundedAction getAction(int t){
		return actionSequence.get(t);
	}
	
	/**
	 * Returns the reward received at timestep t. Note that the fist received reward will be at time step 1, which is the reward received
	 * after taking the first action in the initial state.
	 * @param t the time step of the episode
	 * @return the ith reward received in this episode
	 */
	public double getReward(int t){
		if(t == 0){
			throw new RuntimeException("Cannot return the reward received at time step 0; the first received reward occurs after the initial state at time step 1");
		}
		if(t > rewardSequence.size()){
			throw new RuntimeException("There are only " + this.rewardSequence.size() + " rewards recorded; cannot return the reward for time step " + t);
		}
		return rewardSequence.get(t-1);
	}
	
	/**
	 * Returns the number of time steps in this episode, which is equivalent to the number of states.
	 * @return the number of time steps in this episode
	 */
	public int numTimeSteps(){
		return stateSequence.size(); //state sequence will always have the most because of initial state and terminal state
	}
	
	
	/**
	 * Returns the maximimum time step index in this episode which is the {@link #numTimeSteps()}-1.
	 * @return the maximum time step index in this episode
	 */
	public int maxTimeStep(){
		return this.stateSequence.size()-1;
	}
	
	
	/**
	 * Will return the discounted return received from the first state in the episode to the last state in the episode.
	 * @param discountFactor the discount factor to compute the discounted return; should be on [0, 1]
	 * @return the discounted return of the episode
	 */
	public double getDiscountedReturn(double discountFactor){
		double discount = 1.;
		double sum = 0.;
		for(double r : rewardSequence){
			sum += discount*r;
			discount *= discountFactor;
		}
		return sum;
	}
	
	
	/**
	 * This method will append execution results in e to this object's results. Note that it is assumed that the initial state in e
	 * is the last state recorded in this object. This method is useful for appending the results of an option's execution
	 * to a episode.
	 * @param e the execution results to append to this episode.
	 */
	public void appendAndMergeEpisodeAnalysis(EpisodeAnalysis e){
		for(int i = 0; i < e.numTimeSteps()-1; i++){
			this.recordTransitionTo(e.getAction(i), e.getState(i+1), e.getReward(i+1));
		}
	}
	
	
	/**
	 * Returns a string representing the actions taken in this episode. Actions are separated
	 * by ';' characters.
	 * @return a string representing the actions taken in this episode
	 */
	public String getActionSequenceString(){
		return this.getActionSequenceString("; ");
	}
	
	
	/**
	 * Returns a string representing the actions taken in this episode. Actions are separated
	 * by the provided delimiter string.
	 * @param delimiter the delimiter to separate actions in the string.
	 * @return a string representing the actions taken in this episode
	 */
	public String getActionSequenceString(String delimiter){
		StringBuffer buf = new StringBuffer();
		boolean first = true;
		for(GroundedAction ga : actionSequence){
			if(!first){
				buf.append(delimiter);
			}
			buf.append(ga.toString());
			first = false;
		}
		
		return buf.toString();
	}


	/**
	 * Takes a {@link java.util.List} of {@link burlap.behavior.singleagent.EpisodeAnalysis} objects and writes them to a directory.
	 * The format of the file names will be "baseFileName{index}.episode" where {index} represents the index of the
	 * episode in the list. Furthermore, zeroPaddedDigits specifies the minimum number of digits in the {index} string
	 * (using leading zeros to force the size). For example,
	 * if baseName = "ep" and zeroPaddedDigits = 3, then the episode at index 2 will have the name: "ep002.episode"
	 * If you set zeroPaddedDigits = 0, then the number of digits will be forced to fit the maximum number of digits
	 * necessary to represent the last episode index in the list. For example, if the list has 22 episodes,
	 * then zeroPaddedDigits = 0 will behave like zeroPaddedDigits = 2.
	 * @param episodes the list of episodes to write to disk
	 * @param directoryPath the directory path in which the episodes will be written
	 * @param baseFileName the base file name to use for the episode files
	 * @param zeroPaddedDigits the minimum number of digits
	 * @param sp the state parse used to convert states objects t ostring representations
	 */
	public static void writeEpisodesToDisk(List<EpisodeAnalysis> episodes, String directoryPath, String baseFileName, int zeroPaddedDigits, StateParser sp){

		String format = "%0" + zeroPaddedDigits + "d";
		if(zeroPaddedDigits <= 0){
			zeroPaddedDigits = String.format("%d", episodes.size()-1).length();
			format = "%0" + zeroPaddedDigits + "d";
		}

		if(!directoryPath.endsWith("/")){
			directoryPath += "/";
		}

		for(int i = 0; i < episodes.size(); i++){
			EpisodeAnalysis ea = episodes.get(i);
			ea.writeToFile(directoryPath + baseFileName + String.format(format, i), sp);
		}

	}

	/**
	 * Writes this episode to a file. If the the directory for the specified file path do not exist, then they will be created.
	 * If the file extension is not ".episode" will automatically be added.
	 * @param path the path to the file in which to write this episode.
	 * @param sp the state parser to use to convert state objects to string representations.
	 */
	public void writeToFile(String path, StateParser sp){
		
		if(!path.endsWith(".episode")){
			path = path + ".episode";
		}
		
		File f = (new File(path)).getParentFile();
		if(f != null){
			f.mkdirs();
		}
		
		
		try{
			
			String str = this.parseIntoString(sp);
			BufferedWriter out = new BufferedWriter(new FileWriter(path));
			out.write(str);
			out.close();
			
			
		}catch(Exception e){
			System.out.println(e);
		}
		
	}
	
	
	/**
	 * Converts this episode into a string representation.
	 * @param sp the state parser to use to convert state objects to string representations.
	 * @return a string representation of this episode.
	 */
	public String parseIntoString(StateParser sp){
		
		StringBuffer sbuf = new StringBuffer(256);
		
		for(int i = 0; i < stateSequence.size(); i++){
			
			sbuf.append("#EL#\n").append(sp.stateToString(stateSequence.get(i))).append("\n#ES#\n");
			if(i < stateSequence.size()-1){
				sbuf.append(getSpaceDelimGAString(actionSequence.get(i))).append("\n").append(rewardSequence.get(i)).append("\n");
			}
			
		}
		
		
		return sbuf.toString();
		
	}


	/**
	 * Takes a path to a directory containing .episode files and reads them all into a {@link java.util.List}
	 * of {@link burlap.behavior.singleagent.EpisodeAnalysis} objects.
	 * @param directoryPath the path to the directory containing the episode files
	 * @param d the domain to which the episode states and actions belong
	 * @param sp a state parser that can parse the state string representation in each file
	 * @return a {@link java.util.List} of {@link burlap.behavior.singleagent.EpisodeAnalysis} objects.
	 */
	public static List<EpisodeAnalysis> parseFilesIntoEAList(String directoryPath, Domain d, StateParser sp){

		if(!directoryPath.endsWith("/")){
			directoryPath = directoryPath + "/";
		}

		File dir = new File(directoryPath);
		final String ext = ".episode";

		FilenameFilter filter = new FilenameFilter() {
			public boolean accept(File dir, String name) {
				if(name.endsWith(ext)){
					return true;
				}
				return false;
			}
		};
		String[] children = dir.list(filter);

		List<EpisodeAnalysis> eas = new ArrayList<EpisodeAnalysis>(children.length);

		for(int i = 0; i < children.length; i++){
			String episodeFile = directoryPath + children[i];
			EpisodeAnalysis ea = parseFileIntoEA(episodeFile, d, sp);
			eas.add(ea);
		}

		return eas;
	}

	
	/**
	 * Reads an episode that was written to a file and turns into an EpisodeAnalysis object.
	 * @param path the path to the episode file.
	 * @param d the domain to which the states and actions belong
	 * @param sp a state parser that can parse the state string representation in the file
	 * @return an EpisodeAnalysis object.
	 */
	public static EpisodeAnalysis parseFileIntoEA(String path, Domain d, StateParser sp){
		
		//read whole file into string first
		String fcont = null;
		try{
			fcont = new Scanner(new File(path)).useDelimiter("\\Z").next();
		}catch(Exception E){
			System.out.println(E);
		}
		
		return parseStringIntoEA(fcont, d, sp);
	}
	
	
	/**
	 * Parses a string representation of an episode into an EpisodeAnalysis object.
	 * @param str a string represenation of the episode.
	 * @param d the domain to which the states and actions belong
	 * @param sp a state parser that can parse the state string representation in the file
	 * @return an EpisodeAnalysis object.
	 */
	public static EpisodeAnalysis parseStringIntoEA(String str, Domain d, StateParser sp){
		
		EpisodeAnalysis ea = new EpisodeAnalysis();
		
		String [] elComps = str.split("#EL#\n");
		
		//System.out.println("--" + elComps[0] + "--");
		
		for(int i = 1; i < elComps.length; i++){
			
			String spToken = "\n#ES#";
			if(!elComps[i].endsWith(spToken)){
				spToken += "\n";
			}
			
			String [] parts = elComps[i].split(spToken);
			
			State s = sp.stringToState(parts[0]);
			if(i < elComps.length-1){
				String [] ars = parts[1].split("\n");
				ea.recordTransitionTo(getGAFromSpaceDelimGASTring(d, ars[0]), s, Double.parseDouble(ars[1]));
			}
			else{
				ea.addState(s);
			}
		}
		
		
		return ea;
	}
	
	
	private static GroundedAction getGAFromSpaceDelimGASTring(Domain d, String str){
		
		String [] scomps = str.split(" ");
		Action a = d.getAction(scomps[0]);
		if(a == null){
			//the domain does not have a reference, so create a null action in its place
			a = new NullAction(scomps[0]);
		}
		String [] params = new String[scomps.length-1];
		for(int i = 1; i < scomps.length; i++){
			params[i-1] = scomps[i];
		}
		
		return new GroundedAction(a, params);
	}
	
	private static String getSpaceDelimGAString(GroundedAction ga){
		StringBuffer sbuf = new StringBuffer(30);
		sbuf.append(ga.action.getName());
		for(int i = 0; i < ga.params.length; i++){
			sbuf.append(" ").append(ga.params[i]);
		}
		
		return sbuf.toString();
	}
	
	
	
}
