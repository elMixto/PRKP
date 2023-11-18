from flask import Flask,request
from asgiref.wsgi import WsgiToAsgi
from flask import jsonify
from src.solvers.gurobi import solve_polynomial_knapsack,VAR_TYPE,SolverConfig
from src.data_structures import instance
import json
app = Flask(__name__)

@app.route("/",methods = ["POST"])
def solve():
    data = request.json
    instance = instance.from_dict(json.loads(data['instance']))
    solver_config = SolverConfig.from_json(data['solver_config'])
    return jsonify(solve_polynomial_knapsack(instance,solver_config))

asgiapp = WsgiToAsgi(app)
