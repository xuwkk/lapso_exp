from collections.abc import Iterable
import numpy as np
import cvxpy as cp
import abc
from pypower.api import makeBdc, ext2int, makePTDF

class PS_Basic:
    def __init__(self, grid_xlsx: dict, operation_cfg: dict):
        """Construct the basic DC power grid and functions
        
        Args:
            grid_xlsx: Grid data in xlsx format
            operation_cfg: Operation configuration
        """
        # Read grid data from Excel
        bus = grid_xlsx["bus"]
        gen = grid_xlsx["gen"] 
        branch = grid_xlsx["branch"]
        gencost = grid_xlsx["gencost"]
        solar = grid_xlsx.get("solar", None)
        wind = grid_xlsx.get("wind", None)
        self.op_cfg = operation_cfg

        # System dimensions
        self.no_bus = len(bus)
        self.no_gen = len(gen)
        self.no_branch = len(branch)
        self.load_idx = np.where(bus["PD"].values > 0)[0]
        self.no_load = len(self.load_idx)
        self.no_solar = len(solar) if solar is not None else 0
        self.no_wind = len(wind) if wind is not None else 0
        
        # System parameters
        self.baseMVA = operation_cfg["baseMVA"]
        slack_idx = bus[bus["BUS_TYPE"] == 3]["BUS_I"].values[0]
        self.slack_idx = PS_Basic._to_python_idx(slack_idx)
        self.non_slack_idx = [i for i in range(self.no_bus) if i != self.slack_idx]
        self.non_slack_gen_idx = np.where(gen["GEN_BUS"].values != slack_idx)[0]

        self.slack_theta = bus.iloc[self.slack_idx]["VA"]
            
        # Load parameters
        self.Cl = np.zeros((self.no_bus, self.no_load))
        for i, idx in enumerate(self.load_idx): # Load bus incidence matrix
            self.Cl[idx, i] = 1
        self.load_default = bus["PD"].values[self.load_idx] / self.baseMVA  # p.u.
        self.cls = bus["LOAD_SHED"].values[self.load_idx] * self.baseMVA    # main the actual cost
        
        # Generator parameters
        gen_cols = gen.columns
        self.gen_idx = PS_Basic._to_python_idx(gen["GEN_BUS"].values)
        self.gen_bus_idx = np.array(list(set(self.gen_idx))) # More than one generator can be connected to the same bus

        for col_name in gen_cols:
            if col_name == "GEN_BUS":
                # GenIdx to BusIdx incidence matrix
                self.Cg = np.zeros((self.no_bus, self.no_gen))
                # GenIdx to GenIdx incidence matrix
                self.Cg_to_gen_bus = np.zeros((len(self.gen_bus_idx), self.no_gen))
                for i in range(self.no_gen):
                    idx = PS_Basic._to_python_idx(gen["GEN_BUS"][i])
                    self.Cg[idx, i] = 1
                    self.Cg_to_gen_bus[np.where(self.gen_bus_idx == idx)[0], i] = 1
            else:
                # constraints related to power: convert to p.u.
                setattr(self, col_name.lower(), gen[col_name].values / self.baseMVA)
    
        # Branch parameters
        self.branch = branch
        # incidence matrix
        Cf = np.zeros((self.no_branch, self.no_bus))
        Ct = np.zeros((self.no_branch, self.no_bus))
        for i in range(self.no_branch):
            fbus = PS_Basic._to_python_idx(branch["F_BUS"][i])
            tbus = PS_Basic._to_python_idx(branch["T_BUS"][i])
            Cf[i, fbus] = 1
            Ct[i, tbus] = 1
        self.A = Cf - Ct # bus-to-branch incidence matrix
        
        self.Cf = Cf
        self.Ct = Ct

        # DC power flow matrices
        # change tap to 1 if it is 0
        tap = branch["TAP"].values
        tap[np.where(tap == 0)] = 1
        Bff = 1/(branch["BR_X"].values * tap)
        self.Bf = np.diag(Bff) @ self.A # branch susceptance matrix
        self.Bbus = self.A.T @ self.Bf  # bus susceptance matrix
        self.Pfshift = -branch["SHIFT"].values / 180 * np.pi * Bff
        self.Pbusshift = self.A.T @ self.Pfshift
        self.Gsh = bus['GS'].values / self.baseMVA
        
        # branch flow
        self.pfmax = branch["RATE_A"].values / self.baseMVA
        
        # PTDF matrix
        self.ptdf = self.Bf[:, self.non_slack_idx] @ np.linalg.inv(self.Bbus[self.non_slack_idx, :][:, self.non_slack_idx])
        identity_remove_slack = np.delete(np.eye(self.no_bus), self.slack_idx, axis=0)
        self.ptdf = self.ptdf @ identity_remove_slack
        
        # Generator costs
        # Convert generator cost parameters to cope with the actual values
        # This will also make the status cost reasonable
        for col in gencost.columns:
            value = gencost[col].values
            if col == "SECOND":
                value = value * self.baseMVA**2  # Quadratic term needs square
            elif col in ["FIRST", "STORAGE", "RD_UP", "RD_DOWN"]:
                value = value * self.baseMVA  # Linear terms need single multiplier
            setattr(self, col.lower(), value)
        
        # Renewable parameters
        self.ren_idx = []
        ## solar
        if solar is not None:
            self._init_solar(solar)
        if wind is not None:
            self._init_wind(wind)
        
        # Passive bus
        self.not_gen_ren_idx = [i for i in range(self.no_bus) if i not in self.ren_idx and i not in self.gen_idx]
        
        # Optimization parameters
        self.T = operation_cfg.T
        self._init_optimization_params(operation_cfg)
        
        self.branch = branch
        self.bus = bus
        
        # Set constraint flags with defaults
        self.with_pf_constraint = getattr(operation_cfg, 'with_pf_constraint', True)
        self.with_reserve_constraint = getattr(operation_cfg, 'with_reserve_constraint', True)
        self.flexible_pg_init = getattr(operation_cfg, 'flexible_pg_init', False)
        self.renewable_min = getattr(operation_cfg, 'renewable_min', 0.0) / self.baseMVA
        self.with_min_on_off_time = getattr(operation_cfg, 'with_min_on_off_time', True)
        
    def _init_solar(self, solar):
        """Initialize solar parameters"""
        self.solar_idx = PS_Basic._to_python_idx(solar["INDEX"].values)
        self.solar_default = solar["CAPACITY"].values / self.baseMVA
        self.csc = solar["CURTAIL"].values * self.baseMVA
        self.Cs = np.zeros((self.no_bus, self.no_solar))
        self.ren_idx += self.solar_idx
        for i in range(self.no_solar):
            idx = PS_Basic._to_python_idx(solar["INDEX"][i])
            self.Cs[idx, i] = 1

    def _init_wind(self, wind):
        """Initialize wind parameters"""
        self.wind_idx = PS_Basic._to_python_idx(wind["INDEX"].values)
        self.wind_default = wind["CAPACITY"].values / self.baseMVA
        self.cwc = wind["CURTAIL"].values * self.baseMVA
        self.Cw = np.zeros((self.no_bus, self.no_wind))
        self.ren_idx += self.wind_idx
        for i in range(self.no_wind):
            idx = PS_Basic._to_python_idx(wind["INDEX"][i])
            self.Cw[idx, i] = 1

    def _init_optimization_params(self, operation_cfg):
        """Initialize optimization parameters"""
        # Initial commitment status
        ug_init = getattr(operation_cfg, 'ug_init', [1.0])
        if len(ug_init) == 1:
            self.ug_init = np.ones(self.no_gen) * ug_init[0]
        elif len(ug_init) == self.no_gen:
            self.ug_init = ug_init
        else:
            raise ValueError("Invalid initial commitment status length")

        # Initial power generation ratio 
        pg_init_ratio = getattr(operation_cfg, 'pg_init_ratio', [0.5])
        if len(pg_init_ratio) == 1:
            self.pg_init = pg_init_ratio[0] * self.pmax
        elif len(pg_init_ratio) == self.no_gen:
            self.pg_init = np.array(pg_init_ratio) * self.pmax
        else:
            raise ValueError("Invalid initial pg ratio length")

    @staticmethod
    def _to_python_idx(idx):
        """Convert to 0-based index"""
        if isinstance(idx, Iterable):
            return [int(i) - 1 for i in idx]
        return int(idx) - 1

    def _gen_constraint(self, constraints, pg, ug, delta_pg=None):
        """Add generation constraints"""
        for t in range(self.T):
            if delta_pg is not None:
                constraints += [
                    pg[t] + delta_pg[t] <= cp.multiply(self.pmax, ug[t]),
                    pg[t] + delta_pg[t] >= cp.multiply(self.pmin, ug[t])
                ]
            else:
                constraints += [
                    pg[t] <= cp.multiply(self.pmax, ug[t]),
                    pg[t] >= cp.multiply(self.pmin, ug[t])
                ]
        return constraints

    def _minimum_on_off_time_constraint(self, constraints, ug, yg, zg):
        """Add minimum on/off time constraints"""
        min_on_time = int(self.min_on_time[0])
        min_off_time = int(self.min_off_time[0])
        for t in range(self.T - min_on_time + 1):
            constraints += [cp.sum(yg[t:t+min_on_time]) <= ug[t+min_on_time-1]]
        
        for t in range(self.T - min_off_time + 1):
            constraints += [cp.sum(zg[t:t+min_off_time]) <= 1 - ug[t+min_off_time-1]]
        return constraints
    
    def _flow_balance_constraint(self, constraints, power_flow, pg_all, load_all, solar_all=None, wind_all=None):
        """Add power balance and flow constraints"""
        for t in range(self.T):
            
            power_inj = self.Cg @ pg_all[t] - self.Cl @ load_all[t]
            if solar_all is not None:
                power_inj += self.Cs @ solar_all[t]
            if wind_all is not None:
                power_inj += self.Cw @ wind_all[t]
            
            constraints += [
                cp.sum(power_inj) == 0, # Power balance constraint
            ]
            if self.with_pf_constraint:
                # Power flow constraint
                if power_flow is not None:
                    constraints += [
                        power_flow[t] == self.ptdf @ (power_inj - self.Pbusshift) + self.Pfshift,
                        power_flow[t] <= self.pfmax,
                        power_flow[t] >= -self.pfmax
                    ]
                else:
                    constraints[t] += [
                        self.ptdf @ (power_inj - self.Pbusshift) + self.Pfshift <= self.pfmax,
                        self.ptdf @ (power_inj - self.Pbusshift) + self.Pfshift >= -self.pfmax
                    ]
        return constraints

    def _variable_constraint(self, constraints, load, ls, pg=None, es=None, solar=None, solarc=None, wind=None, windc=None):
        """Add variable bounds"""
        for t in range(self.T):
            constraints += [ls[t] >= 0, ls[t] <= load[t]]
            if es is not None:
                constraints += [es[t] >= 0, es[t] <= pg[t]]
            if solar is not None:
                constraints += [solarc[t] >= 0, solarc[t] <= solar[t] - self.renewable_min] 
            if wind is not None:
                constraints += [windc[t] >= 0, windc[t] <= wind[t] - self.renewable_min]
        return constraints
    def _reserve_constraint(self, constraints, ug):
        for t in range(self.T):
            constraints += [cp.sum(ug[t]) >= 1]
        return constraints
    
    # def _reserve_constraint(self, constraints, sg):
    #     """Add reserve constraints"""
    #     for t in range(self.T):
    #         constraints += [
    #             cp.sum(sg[t]) >= np.max(self.pmax),
    #             sg[t] >= 0,
    #             sg[t] <= self.reserve_max
    #         ]
    #     return constraints

    @abc.abstractmethod
    def formulate(self):
        """Initialize the optimization problem"""
        self.prob_cvxpy = cp.Problem()
        raise NotImplementedError("formulate method must be implemented")

    def solve(self, parameters: dict, verbose: bool = False, solver: str = 'gurobi', warm_start = True, **solver_options):
        """Solve the optimization problem
        return solution dictionary,
        by default, the gurobi solver is used"""
        for param in self.prob_cvxpy.parameters():
            try:
                param.value = parameters[param.name()]
            except:
                raise ValueError(f'Parameter {param.name()} not found or dimension mismatch')
        
        if solver == 'gurobi' or solver == 'GUROBI':
            self.prob_cvxpy.solve(solver=getattr(cp, solver.upper()),
                        warm_start=warm_start,
                        verbose=verbose,
                        **solver_options)
        elif solver == 'mosek' or solver == 'MOSEK':
            self.prob_cvxpy.solve(solver=getattr(cp, solver.upper()),
                                warm_start=warm_start,
                                verbose=verbose,
                                mosek_params=solver_options)
        else:
            raise ValueError(f'Solver {solver} not supported')
        return self.get_sol()

    def get_sol(self):
        """Get solution as dictionary"""
        sol = {var.name(): var.value for var in self.prob_cvxpy.variables()}
        sol['status'] = self.prob_cvxpy.status
        sol['cost'] = self.prob_cvxpy.value
        return sol

    # def get_pf(self, pg_all, load_all, solar_all=None, wind_all=None):
    #     """Compute power flow"""
    #     power_inj = pg_all.reshape(self.T, -1) @ self.Cg.T - load_all.reshape(self.T, -1) @ self.Cl.T
    #     if self.no_solar > 0:
    #         power_inj += solar_all.reshape(self.T, -1) @ self.Cs.T
    #     if self.no_wind > 0:
    #         power_inj += wind_all.reshape(self.T, -1) @ self.Cw.T
        
    #     power_flow = (power_inj - self.Pbusshift.reshape(1,-1)) @ self.ptdf.T + self.Pfshift.reshape(1, -1)
    #     return power_flow

    @abc.abstractmethod
    def analysis(self):
        """Analyze solution"""
        raise NotImplementedError("analysis method must be implemented")

    def system_summary(self):
        """Print system summary"""
        print("========== System Summary: ==========")
        print(f"Bus: {self.no_bus}, Gen: {self.no_gen}, Branch: {self.no_branch}, Load: {self.no_load}, Solar: {self.no_solar}, Wind: {self.no_wind}")
        print(f"Slack bus: {self.slack_idx}")
        print(f"Non slack bus: {self.non_slack_idx}")
        
        gen_cap = np.sum(self.pmax)
        total_cap = gen_cap
        default_load = np.sum(self.load_default)
        
        solar_cap = np.sum(self.solar_default) if self.no_solar > 0 else 0
        wind_cap = np.sum(self.wind_default) if self.no_wind > 0 else 0
        total_cap += solar_cap + wind_cap

        print(f"Total capacity: {round(total_cap,3)}")
        print(f"Generator capacity: {round(gen_cap,3)}, Solar capacity: {round(solar_cap,3)}, Wind capacity: {round(wind_cap,3)}")
        print(f"Renewable/Load: {round((solar_cap + wind_cap) / default_load,3)}")
        print(f"Renewable/Total Gen: {round((solar_cap + wind_cap) / total_cap,3)}")
        print(f"Load/Total Gen: {round(default_load / total_cap,3)}")

    def optimization_summary(self):
        """Print optimization problem summary"""
        print("========== Optimization Info: ==========")
        print(f"Number of time steps: {self.T}")
        
        print('Variables:')
        no_var = sum(np.prod(var.shape) for var in self.prob_cvxpy.variables())
        no_binary = 0
        for var in self.prob_cvxpy.variables():
            print(var.name(), var.shape)
            if var.attributes['boolean']:
                no_binary += np.prod(var.shape)
        print(f'Total variables: {no_var}')
        print(f'Total binary variables: {no_binary}')
        
        print('Constraints:')
        no_cons = sum(int(np.prod(cons.shape)) for cons in self.prob_cvxpy.constraints)
        print(f'Total constraints: {no_cons}')
        
        print('Parameters:')
        param_dim = sum(np.prod(param.shape) for param in self.prob_cvxpy.parameters())
        for param in self.prob_cvxpy.parameters():
            print(param.name(), param.shape)
        print(f'Total parameters: {param_dim}')