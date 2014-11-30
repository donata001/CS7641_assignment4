package burlap.behavior.stochasticgame.mavaluefunction.backupOperators;

import java.util.List;
import java.util.Map;

import burlap.behavior.stochasticgame.mavaluefunction.AgentQSourceMap;
import burlap.behavior.stochasticgame.mavaluefunction.QSourceForSingleAgent;
import burlap.behavior.stochasticgame.mavaluefunction.SGBackupOperator;
import burlap.behavior.stochasticgame.solvers.GeneralBimatrixSolverTools;
import burlap.behavior.stochasticgame.solvers.MinMaxSolver;
import burlap.oomdp.core.State;
import burlap.oomdp.stochasticgames.AgentType;
import burlap.oomdp.stochasticgames.GroundedSingleAction;
import burlap.oomdp.stochasticgames.JointAction;
import burlap.oomdp.stochasticgames.SingleAction;


/**
 * A minmax operator. This operator is useful for zero sum two player games. If there are more than two players in the game, a runtime exception will be thrown.
 * Before solving the minmax strategy, the Q-values are transformed into a minmax game. Then the resulting minmax strategy is used to compute
 * the expected "payoff" using the true Q-values of the query agent, which is then returned as the new Q-value.
 * @author James MacGlashan
 *
 */
public class MinMaxQ implements SGBackupOperator {

	@Override
	public double performBackup(State s, String forAgent,
			Map<String, AgentType> agentDefinitions, AgentQSourceMap qSourceMap) {
		
		if(agentDefinitions.size() != 2){
			throw new RuntimeException("MinMax only defined for two agents.");
		}
		
		String otherAgentName = null;
		for(String aname : agentDefinitions.keySet()){
			if(!aname.equals(forAgent)){
				otherAgentName = aname;
				break;
			}
		}
		
		QSourceForSingleAgent forAgentQSource = qSourceMap.agentQSource(forAgent);
		QSourceForSingleAgent otherAgentQSource = qSourceMap.agentQSource(otherAgentName);
		
		List<GroundedSingleAction> forAgentGSAs = SingleAction.getAllPossibleGroundedSingleActions(s, forAgent, agentDefinitions.get(forAgent).actions);
		List<GroundedSingleAction> otherAgentGSAs = SingleAction.getAllPossibleGroundedSingleActions(s, otherAgentName, agentDefinitions.get(otherAgentName).actions);
		
		double [][] payout1 = new double[forAgentGSAs.size()][otherAgentGSAs.size()];
		
		double [][] truePayout1 = new double[forAgentGSAs.size()][otherAgentGSAs.size()];
		
		for(int i = 0; i < forAgentGSAs.size(); i++){
			for(int j = 0; j < otherAgentGSAs.size(); j++){
				JointAction ja = new JointAction();
				ja.addAction(forAgentGSAs.get(i));
				ja.addAction(otherAgentGSAs.get(j));
				
				double q1 = forAgentQSource.getQValueFor(s, ja).q;
				double q2 = otherAgentQSource.getQValueFor(s, ja).q;
				
				truePayout1[i][j] = q1;
				payout1[i][j] = (q1-q2)/2.;
				
				
			}
		}
		
		
		double [] forAgentStrat = MinMaxSolver.getRowPlayersStrategy(payout1);
		double [] otherAgentStrat = MinMaxSolver.getColPlayersStrategy(GeneralBimatrixSolverTools.getNegatedMatrix(payout1));
		
		//we can use true payoff for player 1 for both players, because we're ignoring the payout for the second player.
		double expectedpayoffforPlayer1 = GeneralBimatrixSolverTools.expectedPayoffs(truePayout1, truePayout1, forAgentStrat, otherAgentStrat)[0];
		
		
		return expectedpayoffforPlayer1;
	}

}
