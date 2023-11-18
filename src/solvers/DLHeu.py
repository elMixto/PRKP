from src.data_structures.instance import Instance
from fastai.tabular.all import RandomSplitter,TabularPandas,CategoryBlock,accuracy,slide,valley,tabular_learner,TabularLearner
from fastai.tabular.all import Learner, DataLoader
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from functools import lru_cache
from src.data_structures.features import ItemBatchFeature,ItemSingleFeature
import time

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


class DLHeu:
    def __init__(self,
                 data_features:      list[ItemSingleFeature | ItemBatchFeature],
                 objective_features: list[ItemSingleFeature | ItemBatchFeature],
                 layers: list[int],
                 ) -> None:
    
        self.data_features = data_features
        self.objective_features = objective_features
        self.leaner: Learner | None = None
        self.training_dl: DataLoader | None = None
        self.layers = layers

    def create_data(self,instances: list[Instance],features: list[ItemSingleFeature | ItemBatchFeature] )->pd.DataFrame:
        """Toma las instancias, calcula todas las features y las empaqueta en un dataframe"""

        features_df = None
        for instance in instances:
            data = []
            for feature in features:
                if issubclass(feature,ItemBatchFeature):
                    data.append(feature.batch_evaluate(instance))
                elif issubclass(feature,ItemSingleFeature):
                    data.append([feature.evaluate(instance,x) for x in range(instance.n_items)])
                else:
                    raise Exception("Feature is not subclass of ItemBatchFeature or ItemSingleFeature")
            data = np.vstack(data).T
            data = pd.DataFrame(data,columns=[f.name for f in features])
            if features_df is None:
                features_df = data
            else:
                features_df = pd.concat([features_df,data])
        return features_df
    
    def create_training_data(self,instances):
        features: list[ItemBatchFeature | ItemSingleFeature] = self.data_features + self.objective_features
        return self.create_data(instances,features)
    
    def create_test_data(self,instances):
        features: list[ItemBatchFeature | ItemSingleFeature] = self.data_features
        return self.create_data(instances,features)

    def create_model(self,training_data: pd.DataFrame) -> TabularLearner:
        """
        Creates the learner object, that then i can run
            a = learner.lr_find(suggest_funcs=(slide, valley),stop_div=False)
            learner.fit(5,lr=(a[1]+a[0])/2)
        On

        """
        splits = RandomSplitter(seed=42)(training_data)
        dls = TabularPandas(
                training_data,splits = splits,
                procs = [],
                cat_names = [],
                cont_names = [f.name for f in self.data_features],
                y_names = [f.name for f in self.objective_features] ,
                y_block = CategoryBlock()).dataloaders(path=".",bs=64)

        learner = tabular_learner(dls,metrics=accuracy,layers=self.layers)
        return learner
    
    def train_learner(self,learner):
        from fastai.tabular.all import slide, valley
        a = learner.lr_find(suggest_funcs=(slide, valley),stop_div=False)
        learner.fit(5,lr=(a[1]+a[0])/2)
        return learner


    def solve(self,instance: Instance,learner: Learner)->"Solution":
        from src.solvers.MLHeu.functions_ml import fix_variables
        from src.solvers.collection import SolverCollection,SolverConfig,VAR_TYPE,Solution
        start = time.time()
        test_data = self.create_test_data([instance])
        test_dl = learner.dls.test_dl(test_data)
        preds, _ = learner.get_preds(dl=test_dl,reorder=True)
        y_ml = fix_variables(instance.n_items, preds, 0.85)
        discrete_config = SolverConfig(VAR_TYPE.BINARY,True,y_ml)
        final_solution  = SolverCollection.gurobi(instance,discrete_config)
        return Solution(final_solution.o,final_solution.sol,time.time()-start)

    def load_model(self)->TabularLearner:
        training_data = pd.read_csv("./models/final_dl.csv")
        model = self.create_model(training_data)
        model.load("final_dl")
        return model

