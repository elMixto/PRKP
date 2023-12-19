from pathlib import Path
from time import time
import pickle
from src.Gurobi import gurobi,SolverConfig,VAR_TYPE
from src.data_structures import Instance,Solution
from .functions_ml import prepare_set,fix_variables

def solve(instance: Instance,time_limit=600,fixed_percentage = 1-0.85):
    start = time()
    model_file = Path(__file__).resolve().parent / "model_data/finalized_model_rTrees.sav"
    n_features = 6
    clf = pickle.load(open(model_file, 'rb'))
    sol_cont = gurobi(instance, SolverConfig(VAR_TYPE.CONTINOUS, False, []))
    X = prepare_set(n_features, instance, sol_cont.sol)
    y_mlProba = clf.predict_proba(X)
    y_ml = fix_variables(instance.n_items, y_mlProba, fixed_percentage)
    discrete_config = SolverConfig(VAR_TYPE.BINARY, True, y_ml,time_limit=time_limit)
    final_gurobi = gurobi(instance, discrete_config)
    return Solution(final_gurobi.o, final_gurobi.sol, time() - start)