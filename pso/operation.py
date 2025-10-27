"""
formulate, interface, solve DC OPF problems
1. NCUC without binary variables
2. NCUC with binary variables
3. ED
"""

import numpy as np
import cvxpy as cp
from pso.operation_basic import PS_Basic

class UC_DISCRETE(PS_Basic):
    """Formulate the DC network-constrained unit commitment problem with binary variables"""
    
    def formulate(self, fix_binary=False):
        """Formulate the optimization problem
        
        Args:
            fix_binary: If True, binary variables are treated as fixed parameters
                        This is used for differentiation
        """
        # Parameters and variables
        load = cp.Parameter((self.T, self.no_load), name='load')
        ls = cp.Variable((self.T, self.no_load), name='ls')
        pg = cp.Variable((self.T, self.no_gen), name='pg')
        if self.with_pf_constraint:
            power_flow = cp.Variable((self.T, self.no_branch), name='power_flow')
        else:
            power_flow = None
        
        # Initial generator status: only for T > 1
        if self.T > 1:
            if self.flexible_pg_init:
                # Treat pg_init as a variable
                pg_init = cp.Variable(self.no_gen, name='pg_init')
            else:
                pg_init = self.pg_init
        
        # Renewable variables
        if self.no_solar > 0:
            solar = cp.Parameter((self.T, self.no_solar), name='solar')
            solarc = cp.Variable((self.T, self.no_solar), name='solarc')
        
        if self.no_wind > 0:
            wind = cp.Parameter((self.T, self.no_wind), name='wind')
            windc = cp.Variable((self.T, self.no_wind), name='windc')

        # Binary variables
        if fix_binary:
            # binary variables == binary parameters
            ug = cp.Variable((self.T, self.no_gen), name='ug', nonneg=True)
            yg = cp.Variable((self.T, self.no_gen), name='yg', nonneg=True)  
            zg = cp.Variable((self.T, self.no_gen), name='zg', nonneg=True)
            ug_parameter = cp.Parameter((self.T, self.no_gen), name='ug_parameter')
            yg_parameter = cp.Parameter((self.T, self.no_gen), name='yg_parameter')
            zg_parameter = cp.Parameter((self.T, self.no_gen), name='zg_parameter')
        else:
            ug = cp.Variable((self.T, self.no_gen), boolean=True, name='ug')
            yg = cp.Variable((self.T, self.no_gen), boolean=True, name='yg')
            zg = cp.Variable((self.T, self.no_gen), boolean=True, name='zg')

        # Objective function
        obj = 0
        for t in range(self.T):
            # The objective is in actual scale, e.g. $/MWh
            obj += cp.scalar_product(self.zero, ug[t])
            obj += cp.scalar_product(self.startup, yg[t])
            obj += cp.scalar_product(self.shutdown, zg[t])
            obj += cp.scalar_product(self.first, pg[t])
            obj += cp.scalar_product(self.cls, ls[t])
            
            if np.sum(self.second) > 0:
                raise NotImplementedError("Second order cost not implemented")
            
            # Curtailment cost
            if self.no_solar > 0:
                obj += cp.scalar_product(self.csc, solarc[t])
            if self.no_wind > 0:
                obj += cp.scalar_product(self.cwc, windc[t])

        # Constraints
        constraints = []
        
        if fix_binary:
            constraints += [
                # binary variables == binary parameters
                ug == ug_parameter,
                yg == yg_parameter,
                zg == zg_parameter
            ]

        # Initial generator status: only for T > 1
        if self.T > 1:
            constraints += [
                # Initial generator status
                yg[0] - zg[0] == ug[0] - self.ug_init,
                pg[0] - pg_init <= cp.multiply(self.ramp_up, self.ug_init) + cp.multiply(self.ramp_startup, yg[0]),
                pg_init - pg[0] <= cp.multiply(self.ramp_down, ug[0]) + cp.multiply(self.ramp_shutdown, zg[0])
            ]
            if self.flexible_pg_init:
                constraints += [
                    pg_init <= self.pmax * self.ug_init,
                    pg_init >= self.pmin * self.ug_init
                ]

        # Follow-up generator status
        for t in range(1, self.T):
            constraints += [
                yg[t] - zg[t] == ug[t] - ug[t-1],
                pg[t] - pg[t-1] <= cp.multiply(self.ramp_up, ug[t-1]) + cp.multiply(self.ramp_startup, yg[t]),
                pg[t-1] - pg[t] <= cp.multiply(self.ramp_down, ug[t]) + cp.multiply(self.ramp_shutdown, zg[t])
            ]
        
        # Binary constraints
        for t in range(self.T):
            constraints += [yg[t] + zg[t] <= 1]
        
        # Reserve constraints
        if self.with_reserve_constraint:
            constraints = self._reserve_constraint(constraints, ug)
        
        constraints = self._gen_constraint(constraints, pg, ug)

        # Minimum on/off time
        if self.with_min_on_off_time:
            constraints = self._minimum_on_off_time_constraint(constraints, ug, yg, zg)
        
        # Power injection and flow constraints
        constraints = self._flow_balance_constraint(
            constraints=constraints,
            power_flow=power_flow,
            pg_all=pg,
            load_all=load - ls,
            solar_all=solar - solarc if self.no_solar > 0 else None,
            wind_all=wind - windc if self.no_wind > 0 else None
        )
        
        # Variable constraints
        constraints = self._variable_constraint(
            constraints=constraints,
            load=load,
            ls=ls,
            solar=solar if self.no_solar > 0 else None,
            solarc=solarc if self.no_solar > 0 else None,
            wind=wind if self.no_wind > 0 else None,
            windc=windc if self.no_wind > 0 else None
        )
        
        # Create and store the problem
        self.prob_cvxpy = cp.Problem(cp.Minimize(obj / self.baseMVA), constraints)

    def compute_cost(self, result):
        """Compute the total cost for the unit commitment solution"""
        cost = 0
        for t in range(self.T):
            cost += np.dot(self.first, result['pg'][t])
            cost += np.dot(self.zero, result['ug'][t])
            cost += np.dot(self.startup, result['yg'][t])
            cost += np.dot(self.shutdown, result['zg'][t])
            cost += np.dot(self.cls, result['ls'][t])
            
            if self.no_solar > 0:
                cost += np.dot(self.csc, result['solarc'][t])
            if self.no_wind > 0:
                cost += np.dot(self.cwc, result['windc'][t])
        
        return cost


class UC_CONTINUOUS(PS_Basic):
    """Formulate the continuous unit commitment problem"""
    
    def formulate(self):
        """Formulate the optimization problem"""
        # Parameters and variables
        load = cp.Parameter((self.T, self.no_load), name='load')
        ls = cp.Variable((self.T, self.no_load), name='ls')
        pg = cp.Variable((self.T, self.no_gen), name='pg')
        if self.with_pf_constraint:
            power_flow = cp.Variable((self.T, self.no_branch), name='power_flow')
        else:
            power_flow = None
        
        # Renewable variables
        if self.no_solar > 0:
            solar = cp.Parameter((self.T, self.no_solar), name='solar')
            solarc = cp.Variable((self.T, self.no_solar), name='solarc')
        
        if self.no_wind > 0:
            wind = cp.Parameter((self.T, self.no_wind), name='wind')
            windc = cp.Variable((self.T, self.no_wind), name='windc')

        # Generator status parameter
        ug = np.tile(self.ug_init, (self.T, 1)) # repeat the initial generator status for the entire time horizon
        
        # Initial generator status: only for T > 1
        if self.T > 1:
            if self.flexible_pg_init:
                pg_init = cp.Variable(self.no_gen, name='pg_init')
            else:
                pg_init = self.pg_init
        
        # Objective function
        obj = 0
        for t in range(self.T):
            obj += cp.scalar_product(self.first, pg[t])
            obj += cp.scalar_product(self.cls, ls[t])
            
            if np.sum(self.second) > 0:
                raise NotImplementedError("Second order cost not implemented")
            
            if self.no_solar > 0:
                obj += cp.scalar_product(self.csc, solarc[t])
            if self.no_wind > 0:
                obj += cp.scalar_product(self.cwc, windc[t])

        # Constraints
        constraints = []
        
        # Initial generator status: only for T > 1
        if self.T > 1:
            constraints += [
                # Only ramps without startup and shutdown constraints
                pg[0] - pg_init <= cp.multiply(self.ramp_up, self.ug_init),
                pg_init - pg[0] <= cp.multiply(self.ramp_down, ug[0])
            ]
            if self.flexible_pg_init:
                constraints += [
                    pg_init <= self.pmax * self.ug_init,
                    pg_init >= self.pmin * self.ug_init
                ]

        # Follow-up generator status
        for t in range(1, self.T):
            constraints += [
                pg[t] - pg[t-1] <= cp.multiply(self.ramp_up, ug[t-1]),
                pg[t-1] - pg[t] <= cp.multiply(self.ramp_down, ug[t])
            ]

        constraints = self._gen_constraint(constraints, pg, ug)

        # Flow balance
        constraints = self._flow_balance_constraint(
            constraints=constraints,
            power_flow=power_flow,
            pg_all=pg,
            load_all=load - ls,
            solar_all=solar - solarc if self.no_solar > 0 else None,
            wind_all=wind - windc if self.no_wind > 0 else None
        )
        
        # Variable constraints
        constraints = self._variable_constraint(
            constraints=constraints,
            load=load,
            ls=ls,
            solar=solar if self.no_solar > 0 else None,
            solarc=solarc if self.no_solar > 0 else None,
            wind=wind if self.no_wind > 0 else None,
            windc=windc if self.no_wind > 0 else None
        )
        
        # Create and store the problem: divided by baseMVA for scaling
        # NOTE: You need to multiply baseMVA back analyzing the results
        self.prob_cvxpy = cp.Problem(cp.Minimize(obj / self.baseMVA), constraints)
    
    def compute_cost(self, result):
        """Compute the total cost for the unit commitment solution"""
        cost = 0
        for t in range(self.T):
            cost += np.dot(self.first, result['pg'][t])
            cost += np.dot(self.cls, result['ls'][t])
            
            if self.no_solar > 0:
                cost += np.dot(self.csc, result['solarc'][t])
            if self.no_wind > 0:
                cost += np.dot(self.cwc, result['windc'][t])
        
        return cost / self.baseMVA
    
class RD(PS_Basic):
    """Redispatch optimization for both discrete and continuous UC cases"""
    
    def formulate(self, discrete_uc=False):
        """Formulate the redispatch optimization problem
        
        Args:
            discrete_uc: If True, use commitment status from discrete UC as parameter
        """
        # Parameters
        load = cp.Parameter((self.T, self.no_load), name='load')
        pg = cp.Parameter((self.T, self.no_gen), name='pg_parameter')
        
        # Variables 
        delta_pg = cp.Variable((self.T, self.no_gen), name='delta_pg')   # redispatch amount
        ls = cp.Variable((self.T, self.no_load), name='ls')             # load shedding
        es = cp.Variable((self.T, self.no_gen), name='es')             # energy storage
        rd_cost = cp.Variable((self.T, self.no_gen), name='rd_cost')   # redispatch cost
        if self.with_pf_constraint:
            power_flow = cp.Variable((self.T, self.no_branch), name='power_flow')
        else:
            power_flow = None

        # Unit commitment status
        if discrete_uc:
            ug = cp.Parameter((self.T, self.no_gen), name='ug_parameter')
        else:
            # repeat the initial generator status for the entire time horizon
            ug = np.tile(self.ug_init, (self.T, 1)) 
            
        # Renewable variables
        if self.no_solar > 0:
            solar = cp.Parameter((self.T, self.no_solar), name='solar')
            solarc = cp.Variable((self.T, self.no_solar), name='solarc')
        if self.no_wind > 0:
            wind = cp.Parameter((self.T, self.no_wind), name='wind')
            windc = cp.Variable((self.T, self.no_wind), name='windc')

        # Objective function
        obj = 0
        for t in range(self.T):
            obj += cp.scalar_product(self.first, delta_pg[t])
            obj += cp.scalar_product(self.cls, ls[t])
            obj += cp.sum(rd_cost[t])
            obj += cp.scalar_product(self.storage, es[t])
            if self.no_solar > 0:
                obj += cp.scalar_product(self.csc, solarc[t])
            if self.no_wind > 0:
                obj += cp.scalar_product(self.cwc, windc[t])

        # Constraints
        constraints = []
        constraints = self._gen_constraint(constraints=constraints, pg=pg, ug=ug, delta_pg=delta_pg)
        
        for t in range(self.T):
            # Ramping constraints
            constraints += [
                delta_pg[t] <= self.ramp_up_rd,
                delta_pg[t] >= -self.ramp_down_rd
            ]
            
            # Redispatch cost constraints
            constraints += [
                rd_cost[t] >= cp.multiply(self.rd_up, delta_pg[t]),
                rd_cost[t] >= -cp.multiply(self.rd_down, delta_pg[t])
            ]
            
        # Flow balance and variable constraints
        constraints = self._flow_balance_constraint(
            constraints=constraints,
            power_flow=power_flow,
            pg_all=pg + delta_pg - es,
            load_all=load - ls,
            solar_all=solar - solarc if self.no_solar > 0 else None,
            wind_all=wind - windc if self.no_wind > 0 else None
        )
        
        constraints = self._variable_constraint(
            constraints=constraints,
            load=load, ls=ls,
            pg=pg + delta_pg, es=es,
            solar=solar if self.no_solar > 0 else None,
            solarc=solarc if self.no_solar > 0 else None,
            wind=wind if self.no_wind > 0 else None,
            windc=windc if self.no_wind > 0 else None
        )

        self.prob_cvxpy = cp.Problem(cp.Minimize(obj / self.baseMVA), constraints)

    def compute_total_cost(self, uc_result, rd_result):
        """Compute total cost including UC and redispatch costs"""
        cost = 0
        for t in range(self.T):
            # Generation and redispatch costs
            cost += np.dot(self.first, uc_result['pg'][t] + rd_result['delta_pg'][t])
            cost += np.dot(self.cls, rd_result['ls'][t])
            cost += np.dot(self.storage, rd_result['es'][t])
            cost += np.sum(rd_result['rd_cost'][t])
            
            # Binary UC costs if applicable
            if 'ug' in uc_result:
                cost += np.dot(self.zero, uc_result['ug'][t])
            if 'yg' in uc_result:
                cost += np.dot(self.startup, uc_result['yg'][t])
            if 'zg' in uc_result:
                cost += np.dot(self.shutdown, uc_result['zg'][t])
                
            # Renewable curtailment costs
            if self.no_solar > 0:
                cost += np.dot(self.csc, rd_result['solarc'][t])
            if self.no_wind > 0:
                cost += np.dot(self.cwc, rd_result['windc'][t])
        
        return cost / self.baseMVA
        