# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simulate cost-optimal electrical grid construction under different policies.

Code contains GridElements: Power Sources, Demands and Storage.  Grid
Elements are placed in different grid regions.  Grid regions are
separated from each other so only sources with grid_region_id == x can
power Demands with grid_region_id == x

The costs of constructing GridElements are based upon:
  nameplate_unit_cost: The cost to build one unit (e.g. Megawatt) of power.
  variable_unit_cost: The cost to provide one unit of power over time.
    (e.g. Megawatt-Hour)

The code simulates the grid over multiple time-slices.  e.g.  Hourly
over a one year period which would map to 24 * 365 = 8760 time-slices.

The code is based upon a linear-program which contains:

  - An objective which is to minimize costs.
  - Constraints which must be met before the solution can converge.
    - conserve_power_constraint: Ensure that sum(power[t]) >=
      demand[t] for all t in each grid-region

This code will work with any set of consistent units.  For the
purposes of documentation, the units chosen are:

  Power: Megawatts
  Time: Hours
  (Derived) Energy = Power * Time => Megawatt-Hours
  Cost: Dollars ($)
  CO2 Emissions: Tonnes

  (Derived) CO2 Emitted per Energy => Tonnes / Megawatt-Hours
  Carbon Tax: $ / Tonnes

"""

import logging
import numpy as np

from ortools.linear_solver import pywraplp


class GridSimError(RuntimeError):
  pass


class DemandNotSatisfiedError(GridSimError):
  pass


class RpsExceedsDemandError(GridSimError):
  pass


class RpsCreditExceedsSourcesError(GridSimError):
  pass


class StorageExceedsDemandError(GridSimError):
  pass


class RpsPercentNotMetError(GridSimError):
  pass


class Constraint(object):
  """Holds an LP Constraint object with extra debugging information.

  Attributes:
     constraint: underlying pywraplp.Constraint object
     name: name of constraint
     formula: hashtable that maps names of variables to coefficients

  pywraplp.Constraint doesn't surface a list of variables/coefficients, so
  we have to keep track ourselves.
  """

  def __init__(self, lp, lower_bound, upper_bound, name=None, debug=False):
    """Initializes Constraint.

    Args:
      lp: LinearProgramContainer that wraps the LP solver which
        creates the constraint.
      lower_bound: (float) Lower bound on product between coeffs and variables.
      upper_bound: (float) Upper bound on product between coeffs and variables.
      name: Optional human readable string.
      debug: Boolean which if set, logs constraint info.
    """

    self.constraint = lp.solver.Constraint(lower_bound, upper_bound)
    self.name = name
    self.formula = {}
    self.debug = debug

    if self.debug:
      logging.debug('CONSTRAINT: %f <= %s <= %f',
                    lower_bound, name, upper_bound)

  def set_coefficient(self, variable, coefficient):
    """Adds variable * coefficient to LP Coefficient.

    Wraps pywrap.SetCoefficient(variable, coefficient) method and
    saves variable, coefficient to formula dict.

    After calling this method, Objective += variable * coefficient

    Args:
      variable: (Lp Variable) The Variable multiplicand.
      coefficient: (float) The coefficient multiplicand.

    """

    self.constraint.SetCoefficient(variable, coefficient)
    self.formula[variable.name()] = coefficient

    if self.debug:
      logging.debug('%s += %s * %f', self.name, variable.name(), coefficient)


class Objective(object):
  """Holds an LP Objective object with extra debugging information.

  Attributes:
    objective: Underlying pywraplp.Objective object.
  """

  def __init__(self, lp, minimize=True):
    """Initializes Objective.

    Args:
       lp: LinearProgramContainer that wraps the LP solver which
         creates the Objective.
       minimize: boolean, True if objective should be minimized
         otherwise objective is maximizied.
    """
    self.objective = lp.solver.Objective()
    self.formula = {}
    if minimize:
      self.objective.SetMinimization()
    else:
      self.objective.SetMaximization()

  def set_coefficient(self, variable, coefficient):
    """Adds variable * coefficient to LP Objective.

    Wraps pywrap.SetCoefficient(variable, coefficient) method and
    saves variable, coefficient to formula dict.

    After calling this method, Objective += variable * coefficient

    Args:
      variable: (Lp Variable) The Variable multiplicand.
      coefficient: (float) The coefficient multiplicand.

    """

    self.objective.SetCoefficient(variable, coefficient)
    self.formula[variable.name()] = coefficient

  def value(self):
    return self.objective.Value()


class GridDemand(object):
  """Simple place-holder object which represents load on the grid."""

  def __init__(self,
               name,
               grid_region_id=0):
    """Initializes GridDemand object.

    Args:
      name: name of the demand object

      grid_region_id: An int specifying the grid region of the demand.
        Only sources with the same grid_region_id can power this demand.

    """

    self.name = name
    self.grid_region_id = grid_region_id


class GridSource(object):
  """Denotes Costs, co2, region, power and energy limitations of a power source.

    Grid Sources may either be dispatchable or non-dispatchable.
      - Dispatchable sources may power at any time, e.g. fossil fuel plants.
      - Non-dispatchable sources are dependent on the environment to
          generate power. e.g. Solar or Wind plants.

    If there is a time-slice power profile indexed by the same name as
    this source in LinearProgramContainer.profiles.  The source is
    considered Non-dispatchable.  Otherwise, it is considered dispatchable.

    Attributes:
      name: (str) name of the object.
      nameplate_unit_cost: (float) Cost to build a unit of
        dispatchable power.  ($ / Megawatt of capacity)

      variable_unit_cost: (float) Cost to supply a unit of dispatchable power
        per time. ($ / Megawatt-Hour)

      grid_region_id: An int specifying the grid region of the source.
        Only demands with the same grid_region_id can sink the power
        from this source.

      max_power: (float) Optional Maximum power which object can supply.
        (Megawatt). Set < 0 if there is no limit.

      max_energy: (float) Optional maximum energy which object can
        supply. (Megawatt-Hours) Set < 0 if there is no limit.

      co2_per_electrical_energy: (float) (Tonnes of CO2 / Megawatt Hour).

      power_coefficient: (float) ratio of how much power is supplied by
        object vs. how much power gets on the grid.  0 <
        power_coefficient < 1.  Nominally 1.0.

      is_rps_source: Boolean which denotes if the source is included
        in the Renewable Portfolio Standard.

      solver: Either a _GridSourceDispatchableSolver or
        _GridSourceNonDispatchableSolver.  Used to setup LP
        Constraints, Objectives and variables for the source and to
        report results.

      timeslice_variables: An array of LP variables, one per time-slice
        of simulation.  Array is mapped so that variable for
        time-slice t is at index t.
        e.g.
          Variable for first time-slice is timeslice_variable[0].
          Variable for last time-slice is timeslice_variable[-1].
          Variable for time-slice at time t is timeslice_variable[t].
        Only gets declared if GridSource is a DispatchableSource.

      nameplate_variable: LP variable representing the nameplate or
        maximum power the GridSource can output at any given
        time.

  """

  def __init__(
      self,
      name,
      nameplate_unit_cost,
      variable_unit_cost,
      grid_region_id=0,
      max_power=-1.0,
      max_energy=-1.0,
      co2_per_electrical_energy=0,
      power_coefficient=1.0,
      is_rps_source=False
  ):
    """Sets characteristics of a GridSource object.

    Args:
      name: (str) name of the object.
      nameplate_unit_cost: (float) Cost to build a unit of
        dispatchable power.  ($ / Megawatt of capacity)

      variable_unit_cost: (float) Cost to supply a unit of dispatchable power
        per time. ($ / Megawatt-Hour)

      grid_region_id: An int specifying the grid region of the demand.
        Only demands with the same grid_region_id can sink the power
        from this source.

      max_power: (float) Maximum power which object can supply. (Megawatt)
      max_energy: (float) Maximum energy which object can
        supply. (Megawatt-Hours)

      co2_per_electrical_energy: (float) (Tonnes of CO2 / Megawatt Hour).
      power_coefficient: (float) ratio of how much power is supplied by
        object vs. how much power gets on the grid.  0 <
        power_coefficient < 1.  Nominally 1.0.
      is_rps_source: Boolean which denotes if the source is included
        in the Renewable Portfolio Standard.
    """

    self.name = name
    self.nameplate_unit_cost = nameplate_unit_cost
    self.variable_unit_cost = variable_unit_cost
    self.max_energy = max_energy
    self.max_power = max_power
    self.grid_region_id = grid_region_id
    self.co2_per_electrical_energy = co2_per_electrical_energy
    self.power_coefficient = power_coefficient
    self.is_rps_source = is_rps_source

    self.solver = None

    self.timeslice_variables = None
    self.nameplate_variable = None

  def configure_lp_variables_and_constraints(self, lp):
    """Declare lp variables, and set constraints.

    Args:
      lp: The LinearProgramContainer.

    Defers to self.solver which properly configures variables and
    constraints in this object.

    See Also:
      _GridSourceDispatchableSolver, _GridSourceNonDispatchableSolver

    """
    self.solver.configure_lp_variables_and_constraints(lp)

  def post_process(self, lp):
    """Update lp post_processing result variables.

    This is done post lp.solve() so that sanity data checks can be done
    on RPS before returning results.

    Args:
      lp: The LinearProgramContainer where the post processing variables reside.
    """

    if lp.rps_percent > 0.0 and self.is_rps_source:
      lp.rps_total[self.grid_region_id] += self.get_solution_values()
    else:
      lp.non_rps_total[self.grid_region_id] += self.get_solution_values()

  def get_solution_values(self):
    """Gets the linear program solver results.

    Must be called after lp.solve() to ensure solver has properly
    converged and has generated results.

    Returns:
      np.array of solutions for each timeslice variable.

    """

    return self.solver.get_solution_values()

  def get_nameplate_solution_value(self):
    """Gets the linear program solver results for nameplate.

    Must be called after lp.solve() to ensure solver has properly
    converged and has generated results.

    Raises:
      RuntimeError: If called before LinearProgramContainer.solve().

    Returns:
      Float value representing solved nameplate value.

    """
    nameplate_variable = self.nameplate_variable

    if nameplate_variable is None:
      raise RuntimeError('Get_nameplate_solution_value called before solve().')

    return nameplate_variable.solution_value()


class _GridSourceDispatchableSolver(object):
  """Power Source which can provide power at any time.

    Attributes:
      source: GridSource object where self generates LP variables
  """

  def __init__(self, source):
    self.source = source

  def configure_lp_variables_and_constraints(self, lp):
    """Declare lp variables, and set constraints in grid_source.

    Args:
      lp: The LinearProgramContainer.

    Variables Declared include:
      - timeslice variables: represent how much power the source is
          outputting at each time-slice.
      - nameplate variable: represents the maximum power sourced.

    The values of these variables are solved by the linear program to
    optimize costs subject to some constraints.

    The overall objective is to minimize cost.  Herein, the overall
    cost is increased by:
      - nameplate cost: nameplate_unit_cost * nameplate variable
      - variable cost: variable_unit_cost * sum(timeslice_variables)
      - carbon cost: lp.carbon_tax * sum(timeslice_variables) *
          co2_per_electrical_energy

    Since variable and carbon costs accrue on a periodic basis, we
      multiply them by lp.cost_of_money to make periodic and
      one-time costs comparable.

    Constraints created / modified here include:
      - Maximum Energy: Ensure sum timeslice-variables < max_energy if
          self.max_energy >= 0.

          This constraint is only for sources where there are limits
          to the total amount of generation which can be built.
          E.g. There are only a limited number of places where one can
          build hydropower.

      - Maximum Power: Ensure no timeslice-variables > max_power if
          self.max_power is >= 0.

          This constraint is only for sources where there are limits
          to the maximum amount of power which can be built.
          E.g. hydropower which can only discharge at a maximum rate.

      - Conserve Power: Ensure that sum(power) > demand for all
          time-slices.  Colloquially called "Keeping the Lights on."

      - Ensure nameplate variable > power(t) for all t.  We must make
          sure that we've priced out a plant which can supply the
          requested power.

    """

    source = self.source

    # setup LP variables.
    source.timeslice_variables = lp.declare_timeslice_variables(
        source.name,
        source.grid_region_id)

    source.nameplate_variable = lp.declare_nameplate_variable(
        source.name,
        source.grid_region_id)

    solver = lp.solver

    # Configure maximum energy if it is >= 0.  Otherwise do not
    # create a constraint.
    max_energy_constraint = (lp.constraint(0.0, source.max_energy)
                             if source.max_energy >= 0 else None)

    # Configure maximum nameplate if it is >= 0.  Otherwise do not
    # create a constraint.
    max_power = source.max_power
    if max_power >= 0:
      lp.constraint(0.0, max_power).set_coefficient(
          source.nameplate_variable, 1.0
      )

    # Total_cost includes nameplate cost.
    cost_objective = lp.minimize_costs_objective
    cost_objective.set_coefficient(source.nameplate_variable,
                                   source.nameplate_unit_cost)

    # Add timeslice variables to coefficients.
    for t, var in enumerate(source.timeslice_variables):

      # Total_cost also includes variable and carbon cost.
      variable_coef = ((source.variable_unit_cost +
                        source.co2_per_electrical_energy * lp.carbon_tax) *
                       lp.cost_of_money)
      cost_objective.set_coefficient(var, variable_coef)

      # Keep the lights on at all times.  Power_coefficient is usually
      # 1.0, but is -1.0 for GridStorage.sink and discharge_efficiency
      # for GridStorage.source.
      lp.conserve_power_constraint[source.grid_region_id][t].set_coefficient(
          var, source.power_coefficient)

      # Constrain rps_credit if needed.
      if source.is_rps_source:
        lp.rps_source_constraints[source.grid_region_id][t].set_coefficient(
            var, source.power_coefficient)

      # Ensure total energy is less than source.max_energy.
      if max_energy_constraint is not None:
        max_energy_constraint.set_coefficient(var, 1.0)

      # Ensure power doesn't exceed source.max_power.
      if max_power >= 0:
        lp.constraint(0.0, max_power).set_coefficient(var, 1.0)

      # Nameplate must be bigger than largest power.
      # If nameplate_unit_cost > 0, Cost Optimization will push
      # Nameplate near max(timeslice_variables).

      nameplate_constraint = lp.constraint(0.0, solver.infinity())
      nameplate_constraint.set_coefficient(var, -1.0)
      nameplate_constraint.set_coefficient(source.nameplate_variable, 1.0)

    # Constrain maximum nameplate if max_power is set.
    if source.max_power >= 0:
      lp.constraint(0.0, source.max_power).set_coefficient(
          source.nameplate_variable,
          1.0)

  def get_solution_values(self):
    """Gets the linear program solver results.

    Must be called after lp.solve() to ensure solver has properly
    converged and has generated results.

    Raises:
      RuntimeError: If called before LinearProgramContainer.solve().

    Returns:
      np.array of solutions for each timeslice variable.

    """

    timeslice_variables = self.source.timeslice_variables

    if timeslice_variables is None:
      raise RuntimeError('get_solution_values called before solve.')

    return np.array([v.solution_value()
                     for v in timeslice_variables])


class _GridSourceNonDispatchableSolver(object):
  """Power Source which can provide nameplate multiple of its profile.

    Attributes:
      source: GridSource object where self generates LP variables
      profile: pandas Series which represents what fraction of the
        nameplate the source can provide at any given time.

  """

  def __init__(self, source, profile):
    self.source = source

    # check profile isn't all zeros
    if (profile.values == 0.0).all():
      raise ValueError('%s profile may not be all zero.' %source.name)

    self.profile = source.power_coefficient * profile / max(profile)

  def configure_lp_variables_and_constraints(self, lp):
    """Declare lp variables, and set constraints in grid_source.

    Args:
      lp: The LinearProgramContainer.

    Variables Declared include:
      - nameplate variable: represents the maximum power sourced.

    The values of these variables are solved by the linear program to
    optimize costs subject to some constraints.

    The overall objective is to minimize cost.  Herein, the overall
    cost is increased by:
      - nameplate cost: nameplate_unit_cost * nameplate variable
      - variable cost: variable_unit_cost * nameplate variable * sum(profile)
      - carbon cost: lp.carbon_tax * nameplate variable * sum(profile)

    Since variable and carbon costs accrue on a yearly basis, we
      multiply them by lp.cost_of_money to make yearly and
      one-time costs comparable.

    Constraints created / modified here include:
      - Maximum Energy: Ensure nameplate * sum(profile) < max_energy if
          self.max_energy >= 0.

          This constraint is only for sources where there are limits
          to the total amount of generation which can be built.
          E.g. There are only a limited number of places where one can
          build hydropower.

      - Maximum Power: Ensure nameplate <= max_power if
          self.max_power >= 0.

          This constraint is only for sources where there are limits
          to the maximum amount of power which can be built.
          E.g. hydropower which can only discharge at a maximum rate.

      - Conserve Power: Ensure that sum(power) > demand for all
          time-slices.  Colloquially called "Keeping the Lights on."

    """

    source = self.source

    # setup LP variables.
    source.nameplate_variable = lp.declare_nameplate_variable(
        source.name,
        source.grid_region_id)

    sum_profile = sum(self.profile)

    # Configure maximum energy if it is >= 0.  Otherwise do not
    # create a constraint.
    if source.max_energy >= 0:
      lp.constraint(0.0, source.max_energy).set_coefficient(
          source.nameplate_variable, sum_profile)

    # Configure maximum energy if it is >= 0.  Otherwise do not
    # create a constraint.
    max_power = source.max_power
    if max_power >= 0:
      lp.constraint(0.0, max_power).set_coefficient(source.nameplate_variable,
                                                    1.0)

    # Total_cost includes nameplate cost.
    cost_objective = lp.minimize_costs_objective

    cost_coefficient = source.nameplate_unit_cost + lp.cost_of_money * (
        source.variable_unit_cost * sum_profile +
        source.co2_per_electrical_energy * sum_profile * lp.carbon_tax)

    cost_objective.set_coefficient(source.nameplate_variable,
                                   cost_coefficient)

    # Add timeslice variables to coefficients.
    for t, profile_t in enumerate(self.profile):

      # Keep the lights on at all times.
      try:
        constraint = lp.conserve_power_constraint[source.grid_region_id]
      except KeyError:
        raise KeyError('No Demand declared in grid_region %d.' % (
            source.grid_region_id))

      constraint[t].set_coefficient(source.nameplate_variable, profile_t)

      # Constrain rps_credit if needed.
      if source.is_rps_source:
        lp.rps_source_constraints[source.grid_region_id][t].set_coefficient(
            source.nameplate_variable, profile_t)

  def get_solution_values(self):
    """Gets the linear program solver results.

    Must be called after lp.solve() to ensure solver has properly
    converged and has generated results.

    Raises:
      RuntimeError: If called before LinearProgramContainer.solve().

    Returns:
      np.array of solutions for each timeslice variable.

    """

    nameplate_variable = self.source.nameplate_variable
    if nameplate_variable is None:
      raise RuntimeError('get_solution_values called before solve.')

    return nameplate_variable.solution_value() * self.profile.values


class GridStorage(object):
  """Stores energy from the grid and returns it when needed subject to losses.

  Attributes:
    name: A string which is the name of the object.
    storage_nameplate_cost: A float which is the cost per nameplate of
      energy storage.  E.g. The cost of batteries.
    charge_nameplate_cost: A float which is the cost per nameplate
      power to charge the storage.  E.g. The rectifier cost to convert
      an AC grid to DC storage.
    discharge_nameplate_cost: A float which is the cost per nameplate
      power to recharge the grid.  E.g. The cost of a power inverter to
      convert DC storage back to AC
    grid_region_id: An int specifying the grid region of the storage.
      The storage can only store energy generated by sources with the
      same grid_region_id.  Only demands with the same grid_region_id
      can sink power from this.
    charge_efficiency: A float ranging from 0.0 - 1.0 which describes
      the energy loss between the grid and the storage element.  0.0
      means complete loss, 1.0 means no loss.
    storage_efficiency: A float ranging from 0.0 - 1.0 which describes
      how much stored energy remains from previous stored energy after
      one time-cycle.  1.0 means no loss. 0.0 means all stored energy
      is lost.
    discharge_efficiency: A float ranging from 0.0 - 1.0 which describes
      the energy loss between storage and grid when recharging the grid.
      0.0 means complete loss, 1.0 means no loss.
    max_charge_power: A float which represents the maximum power that
      can charge storage (calculated before any efficiency losses.).
      A value < 0 means there is no charge power limit.
    max_discharge_power: A float which represents the maximum power
      that can discharge storage (calculated before any efficiency
      losses.).  A value < 0 means there is no discharge power limit.
    max_storage: An optional float which represents the maximum energy
      that can be stored.  A value < 0 means there is no maximum
      storage limit.
    is_rps: Boolean; if true, keeps track of rps_credit as storage is
      charged / discharged.  Amount charging[t] is subtracted from
      rps_credit[t] from rps_credit[t].  Amount discharging[t] is
      added to rps_credit[t].  If false, no rps_credits are adjusted.
  """

  def __init__(
      self,
      name,

      storage_nameplate_cost,
      charge_nameplate_cost=0.0,
      discharge_nameplate_cost=0.0,

      grid_region_id=0,

      charge_efficiency=1.0,
      storage_efficiency=1.0,
      discharge_efficiency=1.0,

      max_charge_power=-1,
      max_discharge_power=-1,
      max_storage=-1,
      is_rps=False
  ):

    self.name = name
    self.storage_nameplate_cost = storage_nameplate_cost
    self.charge_nameplate_cost = charge_nameplate_cost
    self.discharge_nameplate_cost = discharge_nameplate_cost

    self.grid_region_id = grid_region_id
    self.charge_efficiency = charge_efficiency
    self.storage_efficiency = storage_efficiency
    self.discharge_efficiency = discharge_efficiency

    self.max_charge_power = max_charge_power
    self.max_discharge_power = max_discharge_power
    self.max_storage = max_storage
    self.is_rps = is_rps

    # Sink is a power element which sinks from the grid into storage.
    # Source is a power element which sources to the grid from storage.
    # Both are constructed in configure_lp_variables_and_constraints

    self.sink = None
    self.source = None

  def configure_lp_variables_and_constraints(self, lp):
    """Declare lp variables, and set constraints.

    Args:
      lp: LinearProgramContainer, contains lp solver and constraints.
    """

    # Set up LP variables.
    self.energy_variables = lp.declare_timeslice_variables(
        self.name,
        self.grid_region_id
    )

    if self.storage_nameplate_cost:
      self.energy_nameplate = lp.declare_nameplate_variable(
          self.name,
          self.grid_region_id
      )

    # Set up source and configure LP variables.
    self.source = GridSource(
        name=self.name + ' source',
        nameplate_unit_cost=self.discharge_nameplate_cost,
        variable_unit_cost=0.0,
        grid_region_id=self.grid_region_id,
        max_power=self.max_discharge_power,
        co2_per_electrical_energy=0.0,
        power_coefficient=self.discharge_efficiency,
        is_rps_source=self.is_rps
    )
    self.source.solver = _GridSourceDispatchableSolver(self.source)
    self.source.configure_lp_variables_and_constraints(lp)

    # Set up sink and configure LP variables.
    self.sink = GridSource(
        name=self.name + ' sink',
        nameplate_unit_cost=self.discharge_nameplate_cost,
        variable_unit_cost=0.0,
        grid_region_id=self.grid_region_id,
        max_power=self.max_charge_power,
        co2_per_electrical_energy=0.0,
        power_coefficient=-1.0,
        is_rps_source=self.is_rps
    )
    self.sink.solver = _GridSourceDispatchableSolver(self.sink)
    self.sink.configure_lp_variables_and_constraints(lp)

    # Add energy nameplate costs to the objective.  Other costs are
    # added by source/sink.configure_lp_variables_and_constraints.
    if self.storage_nameplate_cost:
      nameplate = self.energy_nameplate
      lp.minimize_costs_objective.set_coefficient(nameplate,
                                                  self.storage_nameplate_cost)

    # Constrain Energy Storage to be Energy Last time plus sink minus source.
    # Storage is circular so variables at t=0 depend on variables at t=-1
    # which is equivalent to last value in python indexing scheme.
    variables = self.energy_variables
    for t in lp.time_index_iterable:
      # Ce = charge_efficiency,
      # Se = storage_efficiency.
      # Stored[i] = se * Stored[i-1] + ce * sink[i-1] - source[i-1]
      # 0 = -Stored[i] + se * Stored[i-1] + ce * sink[i-1] - source[i-1]
      c = lp.constraint(0.0, 0.0)
      c.set_coefficient(variables[t], -1.0)   # -Stored[i]
      c.set_coefficient(variables[t - 1], self.storage_efficiency)

      # Source and sink are relative to the grid, so opposite here:
      # Sink adds to storage, source subtracts from storage.
      c.set_coefficient(self.source.timeslice_variables[t - 1], -1.0)
      c.set_coefficient(self.sink.timeslice_variables[t - 1],
                        self.charge_efficiency)

      # Ensure nameplate is larger than stored_value.
      if self.storage_nameplate_cost:
        nameplate_constraint = lp.constraint(0.0, lp.solver.infinity())
        nameplate_constraint.set_coefficient(nameplate, 1.0)
        nameplate_constraint.set_coefficient(variables[t], -1.0)

      # Constrain maximum storage if max_storage >= 0
      if self.max_storage >= 0.0:
        max_storage_constraint = lp.constraint(0.0, self.max_storage)
        max_storage_constraint.set_coefficient(variables[t], 1.0)

  def post_process(self, lp):
    """Update lp post_processing result variables.

    This is done post lp.solve() so that sanity data checks can be done
    on RPS before returning results.

    Args:
      lp: The LinearProgramContainer where the post processing variables reside.
    """

    sink_vals = self.sink.get_solution_values()
    source_vals = (self.source.get_solution_values() *
                   self.discharge_efficiency)

    if self.is_rps:
      lp.rps_total[self.grid_region_id] += source_vals - sink_vals
    else:
      lp.non_rps_total[self.grid_region_id] += source_vals - sink_vals

  def get_nameplate_solution_value(self):
    """Gets the linear program solver results for nameplate.

    Must be called after lp.solve() to ensure solver has properly
    converged and has generated results.

    Raises:
      RuntimeError: If called before LinearProgramContainer.solve().

    Returns:
      Float value representing solved nameplate value.

    """
    if self.storage_nameplate_cost:
      nameplate_variable = self.energy_nameplate

      if nameplate_variable is None:
        raise RuntimeError(
            'Get_nameplate_solution_value called before solve().')

      return nameplate_variable.solution_value()
    else:
      return max(self.get_solution_values())

  def get_solution_values(self):
    """Gets the linear program solver results.

    Must be called after lp.solve() to ensure solver has properly
    converged and has generated results.

    Raises:
      RuntimeError: If called before LinearProgramContainer.solve().

    Returns:
      np.array of solutions for each timeslice variable.

    """

    timeslice_variables = self.energy_variables

    if timeslice_variables is None:
      raise RuntimeError('get_solution_values called before solve.')

    return np.array([v.solution_value()
                     for v in timeslice_variables])


class GridRecStorage(object):
  """Stores energy from the grid and returns it when needed subject to losses.

  This is a wrapper around two GridStorage objects, one which stores
  "clean" energy (is_rps) and one which stores "dirty" energy (not
  is_rps).  There is a need for both types of storage to keep track of
  renewable energy credits.

  Attributes:
    name: A string which is the name of the object.
    storage_nameplate_cost: A float which is the cost per nameplate of
      energy storage.  E.g. The cost of batteries.
    charge_nameplate_cost: A float which is the cost per nameplate
      power to charge the storage.  E.g. The rectifier cost to convert
      an AC grid to DC storage.
    discharge_nameplate_cost: A float which is the cost per nameplate
      power to recharge the grid.  E.g. The cost of a power inverter to
      convert DC storage back to AC
    grid_region_id: An int specifying the grid region of the storage.
      The storage can only store energy generated by sources with the
      same grid_region_id.  Only demands with the same grid_region_id
      can sink power from this.
    charge_efficiency: A float ranging from 0.0 - 1.0 which describes
      the energy loss between the grid and the storage element.  0.0
      means complete loss, 1.0 means no loss.
    storage_efficiency: A float ranging from 0.0 - 1.0 which describes
      how much stored energy remains from previous stored energy after
      one time-cycle.  1.0 means no loss. 0.0 means all stored energy
      is lost.
    discharge_efficiency: A float ranging from 0.0 - 1.0 which describes
      the energy loss between storage and grid when recharging the grid.
      0.0 means complete loss, 1.0 means no loss.
    max_charge_power: A float which represents the maximum power that
      can charge storage (calculated before any efficiency losses.).
      A value < 0 means there is no charge power limit.
    max_discharge_power: A float which represents the maximum power
      that can discharge storage (calculated before any efficiency
      losses.).  A value < 0 means there is no discharge power limit.
    max_storage: An optional float which represents the maximum energy
      that can be stored.  A value < 0 means there is no maximum
      storage limit.

    rec_storage: GridStorage object which stores "clean" energy.
    no_rec_storage: GridStorage object which stores "dirty" energy.
  """

  def __init__(
      self,
      name,

      storage_nameplate_cost,
      charge_nameplate_cost=0.0,
      discharge_nameplate_cost=0.0,

      grid_region_id=0,

      charge_efficiency=1.0,
      storage_efficiency=1.0,
      discharge_efficiency=1.0,

      max_charge_power=-1,
      max_discharge_power=-1,
      max_storage=-1,
  ):

    self.name = name
    self.storage_nameplate_cost = storage_nameplate_cost
    self.charge_nameplate_cost = charge_nameplate_cost
    self.discharge_nameplate_cost = discharge_nameplate_cost

    self.grid_region_id = grid_region_id
    self.charge_efficiency = charge_efficiency
    self.storage_efficiency = storage_efficiency
    self.discharge_efficiency = discharge_efficiency

    self.max_charge_power = max_charge_power
    self.max_discharge_power = max_discharge_power
    self.max_storage = max_storage

    self.rec_storage = None
    self.no_rec_storage = None

  def configure_lp_variables_and_constraints(self, lp):
    """Declare lp variables, and set constraints."""

    # For rec_storage and no_rec_storage storage, set all costs to 0
    # and with no limits.  Calculate costs and limits after
    # declaration.

    self.rec_storage = GridStorage(
        name=self.name+' REC_STORAGE',
        storage_nameplate_cost=0,
        grid_region_id=self.grid_region_id,
        charge_efficiency=self.charge_efficiency,
        discharge_efficiency=self.discharge_efficiency,
        storage_efficiency=self.storage_efficiency,
        is_rps=True)

    self.no_rec_storage = GridStorage(
        name=self.name+' NO_REC_STORAGE',
        storage_nameplate_cost=0,
        grid_region_id=self.grid_region_id,
        charge_efficiency=self.charge_efficiency,
        discharge_efficiency=self.discharge_efficiency,
        storage_efficiency=self.storage_efficiency,
        is_rps=False)

    self.rec_storage.configure_lp_variables_and_constraints(lp)
    self.no_rec_storage.configure_lp_variables_and_constraints(lp)

    # Calculate costs and limits based on the sum of both rec_storage
    # and no_rec_storage.

    # Set up LP variables.
    self.energy_variables = lp.declare_timeslice_variables(
        self.name,
        self.grid_region_id
    )

    self.energy_nameplate = lp.declare_nameplate_variable(
        self.name,
        self.grid_region_id
    )

    self.charge_nameplate = lp.declare_nameplate_variable(
        self.name + ' charge nameplate',
        self.grid_region_id
    )

    self.discharge_nameplate = lp.declare_nameplate_variable(
        self.name + ' discharge nameplate',
        self.grid_region_id
    )

    # Set limits if needed.
    if self.max_storage >= 0:
      lp.constraint(0.0, self.max_storage).set_coefficient(
          self.energy_nameplate, 1.0)

    if self.max_charge_power >= 0:
      lp.constraint(0.0, self.max_charge_power).set_coefficient(
          self.charge_nameplate, 1.0)

    if self.max_discharge_power >= 0:
      lp.constraint(0.0, self.max_discharge_power).set_coefficient(
          self.discharge_nameplate, 1.0)

    # Add energy nameplate costs to the objective.
    lp.minimize_costs_objective.set_coefficient(self.energy_nameplate,
                                                self.storage_nameplate_cost)
    lp.minimize_costs_objective.set_coefficient(self.charge_nameplate,
                                                self.charge_nameplate_cost)
    lp.minimize_costs_objective.set_coefficient(self.discharge_nameplate,
                                                self.discharge_nameplate_cost)

    rec_storage_energy_variables = self.rec_storage.energy_variables
    no_rec_storage_energy_variables = self.no_rec_storage.energy_variables

    for t in lp.time_index_iterable:
      # Ensure nameplate is >= sum(stored_values)[t].
      nameplate_constraint = lp.constraint(0.0, lp.solver.infinity())
      nameplate_constraint.set_coefficient(self.energy_nameplate, 1.0)
      nameplate_constraint.set_coefficient(rec_storage_energy_variables[t],
                                           -1.0)
      nameplate_constraint.set_coefficient(no_rec_storage_energy_variables[t],
                                           -1.0)

      rec_storage_charge_variables = (
          self.rec_storage.sink.timeslice_variables)
      no_rec_storage_charge_variables = (
          self.no_rec_storage.sink.timeslice_variables)
      rec_storage_discharge_variables = (
          self.rec_storage.source.timeslice_variables)
      no_rec_storage_discharge_variables = (
          self.no_rec_storage.source.timeslice_variables)

      max_charge_constraint = lp.constraint(0.0, lp.solver.infinity())
      max_charge_constraint.set_coefficient(self.charge_nameplate, 1.0)
      max_charge_constraint.set_coefficient(
          rec_storage_charge_variables[t], -1.0)
      max_charge_constraint.set_coefficient(
          no_rec_storage_charge_variables[t], -1.0)
      max_charge_constraint.set_coefficient(
          rec_storage_discharge_variables[t], 1.0)
      max_charge_constraint.set_coefficient(
          no_rec_storage_discharge_variables[t], 1.0)

      max_discharge_constraint = lp.constraint(0.0, lp.solver.infinity())
      max_discharge_constraint.set_coefficient(self.discharge_nameplate, 1.0)
      max_discharge_constraint.set_coefficient(
          rec_storage_charge_variables[t], 1.0)
      max_discharge_constraint.set_coefficient(
          no_rec_storage_charge_variables[t], 1.0)
      max_discharge_constraint.set_coefficient(
          rec_storage_discharge_variables[t], -1.0)
      max_discharge_constraint.set_coefficient(
          no_rec_storage_discharge_variables[t], -1.0)

  def get_solution_values(self):
    return (self.rec_storage.get_solution_values() +
            self.no_rec_storage.get_solution_values())

  def get_source_solution_values(self):
    return (self.rec_storage.source.get_solution_values() +
            self.no_rec_storage.source.get_solution_values() -
            self.rec_storage.sink.get_solution_values() -
            self.no_rec_storage.sink.get_solution_values())

  def get_sink_solution_values(self):
    return -self.get_source_solution_values()

  def get_nameplate_solution_value(self):
    """Gets the linear program solver results for nameplate.

    Must be called after lp.solve() to ensure solver has properly
    converged and has generated results.

    Raises:
      RuntimeError: If called before LinearProgramContainer.solve().

    Returns:
      Float value representing solved nameplate value.

    """
    if self.storage_nameplate_cost:
      nameplate_variable = self.energy_nameplate

      if nameplate_variable is None:
        raise RuntimeError(
            'Get_nameplate_solution_value called before solve().')

      return nameplate_variable.solution_value()
    else:
      return max(self.get_solution_values())

  def post_process(self, lp):
    self.rec_storage.post_process(lp)
    self.no_rec_storage.post_process(lp)


class _GridTransmission(GridSource):
  """Shuttles power from one time-zone to another."""

  def __init__(
      self,
      name,
      nameplate_unit_cost,
      source_grid_region_id=0,
      sink_grid_region_id=1,
      max_power=-1.0,
      efficiency=1.0):
    """Init function.

    Args:
      name: String name of the object.
      nameplate_unit_cost: (float) Cost to build a unit of
        transmission capacity.  ($ / Megawatt of capacity)
      source_grid_region_id: An int specifying which grid_region
        power gets power added.
      sink_grid_region_id: An int specifying which grid_region
        power gets power subtracted.
      max_power: (float) Optional Maximum power which can be transmitted.
        (Megawatt). Set < 0 if there is no limit.
      efficiency: (float) ratio of how much power gets moved one
        grid_region to the other grid_region. Acceptable values are
        0. < efficiency < 1.
    """

    super(_GridTransmission, self).__init__(
        name,
        nameplate_unit_cost=nameplate_unit_cost,
        variable_unit_cost=0,
        grid_region_id=source_grid_region_id,
        max_power=max_power,
        max_energy=-1,
        co2_per_electrical_energy=0,
        power_coefficient=efficiency
    )

    self.sink_grid_region_id = sink_grid_region_id

    self.solver = _GridSourceDispatchableSolver(self)

  def configure_lp_variables_and_constraints(self, lp):
    """Declare lp variables, and set constraints.

    Args:
      lp: LinearProgramContainer, contains lp solver and constraints.
    """

    super(_GridTransmission, self).configure_lp_variables_and_constraints(lp)

    # Handle Constraints.
    for t, var in enumerate(self.timeslice_variables):

      sink_id = self.sink_grid_region_id
      source_id = self.grid_region_id

      # Whatever the super-class is sourcing in source_grid_region_id,
      # sink it from sink_grid_region_id.
      lp.conserve_power_constraint[sink_id][t].set_coefficient(var, -1.0)
      if self.is_rps_source:
        lp.rps_source_constraints[sink_id][t].set_coefficient(var, -1.0)

  def post_process(self, lp):
    """Update lp post_processing result variables.

    This is done so that sanity data checks can be done on RPS before
    returning results.

    Args:
      lp: The LinearProgramContainer where the post processing variables reside.
    """
    # Normal source post_process
    super(_GridTransmission, self).post_process(lp)

    # Sink post_process
    sink_id = self.sink_grid_region_id
    if lp.rps_percent > 0.0 and self.is_rps_source:
      lp.rps_total[sink_id] -= self.get_solution_values()
    else:
      lp.non_rps_total[sink_id] -= self.get_solution_values()


class GridTransmission(object):
  """Transmits power bidirectionally between two grid_regions.

    At interface level, transmitting from region-m to region-n is
    identical to transmitting from region-n to region-m.

    Attributes:
      name: (str) name of the object.
      nameplate_unit_cost: (float) Cost to build a unit of
        transmission capacity.  ($ / Megawatt of capacity)
      grid_region_id_a: An int specifying one grid_region transmission
        terminus
      grid_region_id_b: An int specifying a different grid_region
        transmission terminus
      max_power: (float) Optional Maximum power which can be transmitted.
        (Megawatt). Set < 0 if there is no limit.
      efficiency: (float) ratio of how much power gets moved one
        grid_region to the other grid_region. Acceptable values are
        0. < efficiency < 1.
      a_to_b: _GridTransmission object which moves dirty power from
        grid_region_a to grid_region_b
      b_to_a: _GridTransmission object which moves dirty power from
        grid_region_b to grid_region_a
      rec_a_to_b: _GridTransmission object which moves clean power
        from grid_region_a to grid_region_b
      rec_b_to_a: _GridTransmission object which moves clean power
        from grid_region_b to grid_region_a
  """

  def __init__(
      self,
      name,
      nameplate_unit_cost,
      grid_region_id_a,
      grid_region_id_b,
      efficiency=1.0,
      max_power=-1.0,
  ):

    self.name = name
    self.nameplate_unit_cost = nameplate_unit_cost
    self.grid_region_id_a = grid_region_id_a
    self.grid_region_id_b = grid_region_id_b
    self.efficiency = efficiency
    self.max_power = max_power

    self.a_to_b = None
    self.b_to_a = None
    self.rec_a_to_b = None
    self.rec_b_to_a = None

  def configure_lp_variables_and_constraints(self, lp):
    """Declare lp variables, and set constraints.

    Args:
      lp: LinearProgramContainer, contains lp solver and constraints.
    """

    self.a_to_b = _GridTransmission(self.name + ' a_to_b',
                                    0,
                                    self.grid_region_id_b,
                                    self.grid_region_id_a,
                                    self.max_power,
                                    self.efficiency)

    self.b_to_a = _GridTransmission(self.name + ' b_to_a',
                                    0,
                                    self.grid_region_id_a,
                                    self.grid_region_id_b,
                                    self.max_power,
                                    self.efficiency)

    self.rec_a_to_b = _GridTransmission(self.name + ' rec a_to_b',
                                        0,
                                        self.grid_region_id_b,
                                        self.grid_region_id_a,
                                        self.max_power,
                                        self.efficiency,
                                        is_rps=True)

    self.rec_b_to_a = _GridTransmission(self.name + ' rec b_to_a',
                                        0,
                                        self.grid_region_id_a,
                                        self.grid_region_id_b,
                                        self.max_power,
                                        self.efficiency,
                                        is_rps=True)

    self.a_to_b.configure_lp_variables_and_constraints(lp)
    self.b_to_a.configure_lp_variables_and_constraints(lp)
    self.rec_a_to_b.configure_lp_variables_and_constraints(lp)
    self.rec_b_to_a.configure_lp_variables_and_constraints(lp)

    # Make sure nameplate >= sum(a_to_b) and nameplate >= sum(b_to_a)

    self.nameplate_variable = lp.declare_nameplate_variable(
        self.name, '%d_%d' % (self.grid_region_id_a, self.grid_region_id_b))

    lp.minimize_costs_objective.set_coefficient(self.nameplate_variable,
                                                self.nameplate_unit_cost)

    for t in lp.time_index_iterable:

      # nameplate >= a_to_b[t] + rec_a_to_b[t] - b_to_a[t] - rec_b_to_a[t]
      a_to_b_constraint = lp.constraint(0.0, lp.solver.infinity())
      a_to_b_constraint.set_coefficient(self.nameplate_variable, 1.0)
      a_to_b_constraint.set_coefficient(
          self.a_to_b.timeslice_variables[t], -1.0)
      a_to_b_constraint.set_coefficient(
          self.rec_a_to_b.timeslice_variables[t], -1.0)
      a_to_b_constraint.set_coefficient(
          self.b_to_a.timeslice_variables[t], 1.0)
      a_to_b_constraint.set_coefficient(
          self.rec_b_to_a.timeslice_variables[t], 1.0)

      # nameplate >= b_to_a[t] + rec_b_to_a[t] - a_to_b[t] - rec_a_to_b[t]
      b_to_a_constraint = lp.constraint(0.0, lp.solver.infinity())
      b_to_a_constraint.set_coefficient(self.nameplate_variable, 1.0)
      b_to_a_constraint.set_coefficient(
          self.b_to_a.timeslice_variables[t], -1.0)
      b_to_a_constraint.set_coefficient(
          self.rec_b_to_a.timeslice_variables[t], -1.0)
      b_to_a_constraint.set_coefficient(
          self.a_to_b.timeslice_variables[t], 1.0)
      b_to_a_constraint.set_coefficient(
          self.rec_a_to_b.timeslice_variables[t], 1.0)

  def post_process(self, lp):
    """Update lp post_processing result variables.

    This is done so that sanity data checks can be done on RPS before
    returning results.

    Args:
      lp: The LinearProgramContainer where the post processing variables reside.
    """

    self.a_to_b.post_process(lp)
    self.b_to_a.post_process(lp)
    self.rec_a_to_b.post_process(lp)
    self.rec_b_to_a.post_process(lp)

  def get_nameplate_solution_value(self):
    """Gets the linear program solver results for nameplate.

    Must be called after lp.solve() to ensure solver has properly
    converged and has generated results.

    Raises:
      RuntimeError: If called before LinearProgramContainer.solve().

    Returns:
      Float value representing solved nameplate value.

    """
    nameplate_variable = self.nameplate_variable

    if nameplate_variable is None:
      raise RuntimeError('Get_nameplate_solution_value called before solve().')

    return nameplate_variable.solution_value()


class LinearProgramContainer(object):
  """Instantiates and interfaces to LP Solver.

  Example Usage:
  Initialize: lp = LinearProgramContainer()
  Add objects:
    lp.add_demands(<GridDemand>)
    lp.add_sources(<GridSource>)
    lp.add_transmissions(<GridTransmission>)
    lp.solve()

  Attributes:
    carbon_tax: The amount to tax 1 unit of co2 emissions.
    cost_of_money: The amount to multiply variable costs by to
      make yearly costs and fixed costs comparable.
    profiles: time-series profiles indexed by name which map to
      GridDemands and GridNonDispatchableSources.
    number_of_timeslices: int representing one timeslice per profile index.
    time_index_iterable: A simple int range from 0 - number_of_timeslices.

    Constraints:
      conserve_power_constraint: Dict keyed by grid_region_id. Value
        is a list of LP Constraints which ensures that power > demand
        at all times in all grid_regions.

      minimize_costs_objective: The LP Objective which is to minimize costs.

      rps_source_constraints: Dict keyed by grid_region_id. Value is a
        list of LP Constraints which ensures that
        rps_credit[grid_region, t] <= sum(rps_sources[grid_region, t])

      rps_demand_constraints: Dict keyed by grid_region_id.  Value is
        a list of LP Constraints which ensures that
        rps_credit[grid_region, t] <= demand[grid_region, t]

    RPS Variables:
      rps_credit_variables: Dict object keyed by grid_region_id.  Value is a
        list of rps_credit[grid_region, t] variables for calculating rps.

    Post Processing Variables.  Computed after LP converges:
      rps_total: Dict object keyed by grid_region_id.  Value is sum
        (GridSource_power[grid_region, t]) of all rps sources.

      non_rps_total: Dict object keyed by grid_region_id.  Value is sum
        (GridSource_power[grid_region, t]) of all non_rps sources.

      adjusted_demand: Dict object keyed by grid_region_id.  Value is
        Demand[grid_region, t]

      rps_credit_values: Dict object keyed by grid_region_id.  Value is
        rps_credit.value[grid_region, t]

    Grid Elements:
      demands: A list of GridDemand(s).
      sources: A list of GridSource(s).
      storage: A list of GridStorage(s).
      transmission: A list of GridTransmission(s).

    solver: The wrapped pywraplp.Solver.
    solver_precision: A float representing estimated precision of the solver.
  """

  def __init__(self, profiles):
    """Initializes LP Container.

    Args:
      profiles: Time-series pandas dataframe profiles indexed by name
        which map to GridDemands and GridNonDispatchableSources.

    Raises:
      ValueError: If any value in profiles is < 0 or Nan / None.
    """

    self.carbon_tax = 0.0
    self.cost_of_money = 1.0
    self.rps_percent = 0.0

    self.profiles = profiles

    # Constraints
    self.conserve_power_constraint = {}
    self.minimize_costs_objective = None

    # RPS Constraints
    self.rps_source_constraints = {}
    self.rps_demand_constraints = {}

    # RPS Variables
    self.rps_credit_variables = {}

    # Post Processing Variables
    self.rps_total = {}
    self.non_rps_total = {}
    self.adjusted_demand = {}
    self.total_demand = 0
    self.rps_demand = 0

    self.rps_credit_values = {}

    self.demands = []
    self.sources = []
    self.storage = []
    self.transmission = []

    self.solver = None
    self.solver_precision = 1e-3

    # Validate profiles
    if profiles is None:
      raise ValueError('No profiles specified.')

    if profiles.empty:
      raise ValueError('No Data in Profiles.')

    if profiles.isnull().values.any():
      raise ValueError('Profiles may not be Null or None')

    profiles_lt_0 = profiles.values < 0
    if profiles_lt_0.any():
      raise ValueError('Profiles must not be < 0.')

    self.number_of_timeslices = len(profiles)
    self.time_index_iterable = range(self.number_of_timeslices)

  def add_demands(self, *demands):

    """Add all GridDemands in Args to self.demands."""
    for d in demands:
      self.demands.append(d)

  def add_dispatchable_sources(self, *sources):
    """Verify source has no profile associated with it and add to self.sources.

    Args:
      *sources: arbitrary number of GridSources.

    Raises:
      KeyError: if Source has a profile associated with it which would
        indicate the source was non-dispatchable instead of
        dispatchable.
    """

    for source in sources:
      if source.name in self.profiles:
        raise KeyError(
            'Dispatchable Source %s has a profile associated with it' %(
                source.name
            )
        )

      source.solver = _GridSourceDispatchableSolver(source)
      self.sources.append(source)

  def add_nondispatchable_sources(self, *sources):
    """Verify source has a profile associated with it and add to self.sources.

    Args:
      *sources: arbitrary number of GridSources.

    Raises:
      KeyError: if Source has no profile associated with it which would
        indicate the source was dispatchable instead of
        non-dispatchable.
    """

    for source in sources:
      if source.name not in self.profiles:
        known_sources = ','.join(sorted(self.profiles.columns))
        known_source_string = 'Known sources are (%s).' % known_sources
        raise KeyError(
            'Nondispatchable Source %s has no profile. %s' % (
                source.name,
                known_source_string
            )
        )

      source.solver = _GridSourceNonDispatchableSolver(
          source,
          self.profiles[source.name])

      self.sources.append(source)

  def add_storage(self, *storage):
    """Add storage to lp."""

    self.storage.extend(storage)

  def add_transmissions(self, *transmission):
    """Add transmission to lp."""

    self.transmission.extend(transmission)

  def constraint(self, lower, upper, name=None, debug=False):
    """Build a new Constraint which with valid range between lower and upper."""
    return Constraint(self, lower, upper, name, debug)

  def _initialize_solver(self):
    """Initializes solver, declares objective and set constraints.

    Solver is pywraplp.solver.
    Objective is to minimize costs subject to constraints.

    One constraint declared here is to ensure that
    power[grid_region][t] > demand[grid_region][t] for all t and
    grid_regions.

    Also configures GridElements.

    """
    self.solver = pywraplp.Solver('SolveEnergy',
                                  pywraplp.Solver.CLP_LINEAR_PROGRAMMING)

    self.minimize_costs_objective = Objective(self, minimize=True)

    # Initialize GridDemands and GridSources
    demand_sum = 0.0
    for d in self.demands:
      try:
        profiles = self.profiles[d.name]
        self.adjusted_demand[d.grid_region_id] = np.array(profiles.values)
      except KeyError:
        profile_names = str(self.profiles.keys())
        error_string = 'GridDemand %s. No profile found! Known profiles:(%s)' %(
            d.name,
            profile_names
        )

        raise KeyError(error_string)
      self.conserve_power_constraint[d.grid_region_id] = [
          self.constraint(p,
                          self.solver.infinity(),
                          'Conserve Power gid:%d t:%d' %(
                              d.grid_region_id, t))
          for t, p in enumerate(profiles)
      ]

      demand_sum += sum(profiles)

    # Handle RPS which is tricky.  It requires special credit
    # variables[grid_region][time] and 3 constraints.
    #
    # Constraint #1:
    # The overall goal is to have RPS exceed rps_percent of total
    # demand.  Given that:
    #   total_rps_credit := sum(rps_credit[g][t])
    #   total_demand := sum(demand[g][t])
    #
    # The constraint named total_rps_credit_gt_rps_percent_constraint
    # is:
    #   total_rps_credit >= (self.rps_percent / 100) * total_demand
    #
    # Constraint #2:
    # rps_credit[g][t] cannot exceed sum of rps_sources - sum of
    # rps_sinks at each g,t.  An example of rps_sink is the 'REC_STORAGE'
    # part of GridRecStorage which stores rps energy off the grid only
    # to put it back on the grid later as a rps_source.  This is
    # reflected in the constraint named
    # rps_source_constraints[g][t]:
    #   rps_credit[g][t] <= sum(rps_sources[g][t]) - sum(rps_sinks[g][t])
    #
    # Constraint #3
    # rps_credit[g][t] cannot exceed what can be used at each g,t.  if
    # rps_sources generate a Gigawatt at g,t = 0,0 and only 1MW can be
    # used at g,t then we don't want to credit the unused 999 MW.
    #
    # The constraint named rps_demand_constraints is:
    #   rps_credit[g][t] <= demand[g][t]
    #

    self.total_demand = demand_sum
    self.rps_demand = demand_sum * self.rps_percent / 100.
    solver = self.solver
    total_rps_credit_gt_rps_percent_constraint = self.constraint(
        self.rps_demand,
        solver.infinity()
    )

    for d in self.demands:
      profiles = self.profiles[d.name]

      if self.rps_percent > 0.0:
        rps_credit_variables = self.declare_timeslice_variables(
            '__rps_credit__',
            d.grid_region_id
        )
      else:
        rps_credit_variables = [solver.NumVar(0.0, 0.0,
                                              '__bogus rps_credit__ %d %d' %(
                                                  d.grid_region_id, t))
                                for t in self.time_index_iterable]

      rps_demand_constraints = []
      rps_source_constraints = [self.constraint(0.0, solver.infinity())
                                for t in self.time_index_iterable]

      self.rps_source_constraints[d.grid_region_id] = rps_source_constraints

      self.rps_credit_variables[d.grid_region_id] = rps_credit_variables

      for t in self.time_index_iterable:

        # Sum(rps_credit[grid_region, t]) >= rps_percent * total demand.
        total_rps_credit_gt_rps_percent_constraint.set_coefficient(
            rps_credit_variables[t], 1.0)

        # Rps_credit[grid_region, t] <= demand[grid_region, t].
        rps_credit_less_than_demand = self.constraint(-solver.infinity(),
                                                      profiles[t])
        rps_credit_less_than_demand.set_coefficient(rps_credit_variables[t],
                                                    1.0)
        rps_demand_constraints.append(rps_credit_less_than_demand)

        # Rps_credit[grid_region, t] <= (sum(rps_sources[grid_region, t])
        # Constraint also gets adjusted by _GridSource(Non)DispatchableSolver.
        # configure_lp_variables_and_constraints
        rps_source_constraints[t].set_coefficient(rps_credit_variables[t],
                                                  -1.0)

      self.rps_demand_constraints[d.grid_region_id] = rps_demand_constraints

    # Configure sources and storage.
    for s in self.sources + self.storage + self.transmission:
      s.configure_lp_variables_and_constraints(self)

  def solve(self):
    """Initializes and runs linear program.

    This is the main routine to call after __init__.

    Returns:
      True if linear program gave an optimal result.  False otherwise.
    """
    self._initialize_solver()
    status = self.solver.Solve()
    converged = status == self.solver.OPTIMAL

    if converged:
      self._post_process()

    return converged

  def _post_process(self):
    """Generates data used for calculating consumed rps/non-rps values.

    Also double-checks results to make sure they match constraints.

    Raises:
      RuntimeError: If double-checked results do not match constraints.
    """

    # Initialize post_processing totals.
    for d in self.demands:

      # Total amount of rps_sources[g][t] power.
      self.rps_total[d.grid_region_id] = np.zeros(self.number_of_timeslices)
      # Total amount of non-rps_sources[g][t] power.
      self.non_rps_total[d.grid_region_id] = np.zeros(self.number_of_timeslices)

    for s in self.sources + self.storage + self.transmission:
      s.post_process(self)

    # Sanity error check results against constraints.  If any of these
    # get raised, it indicates a bug in the code.
    solver_precision = self.solver_precision
    sum_rps_credits = 0.0
    for g_id in [d.grid_region_id for d in self.demands]:

      power_deficit = (self.adjusted_demand[g_id] -
                       (self.rps_total[g_id] + self.non_rps_total[g_id]))

      lights_kept_on = (power_deficit < solver_precision).all()

      rps_credits = np.array(
          [rcv.solution_value() for rcv in self.rps_credit_variables[g_id]])
      sum_rps_credits += sum(rps_credits)
      self.rps_credit_values[g_id] = rps_credits

      rps_credit_gt_demand = (
          rps_credits > self.adjusted_demand[g_id] + solver_precision).all()

      rps_credit_gt_rps_sources = (
          rps_credits > self.rps_total[g_id] + solver_precision).all()

      storage_exceeds_demand = (
          self.adjusted_demand[g_id] < -solver_precision).all()

      if not lights_kept_on:
        raise DemandNotSatisfiedError(
            'Demand not satisfied by %f for region %d' % (max(power_deficit),
                                                          g_id))

      if rps_credit_gt_demand:
        raise RpsExceedsDemandError(
            'RPS Credits Exceed Demand for region %d' %g_id)

      if rps_credit_gt_rps_sources:
        raise RpsCreditExceedsSourcesError(
            'RPS Credits Exceed RPS Sources for region %d' %g_id)

      if storage_exceeds_demand:
        raise StorageExceedsDemandError(
            'Storage Exceeds Demand for region %d' %g_id)

    # Scale solver_precision by number of timeslices to get precision
    # for a summed comparison.
    sum_solver_precision = solver_precision * self.number_of_timeslices
    if sum_solver_precision + sum_rps_credits < self.rps_demand:
      raise RpsPercentNotMetError(
          'Sum RPS credits (%f) < demand * (%f rps_percent) (%f)' %(
              sum_rps_credits,
              float(self.rps_percent),
              self.rps_demand
          )
      )

  def declare_timeslice_variables(self, name, grid_region_id):
    """Declares timeslice variables for a grid_region.

    Args:
      name: String to be included in the generated variable name.
      grid_region_id: Int which identifies which grid these variables affect.

    Do Not call this function with the same (name, grid_region_id)
    pair more than once.  There may not be identically named variables
    in the same grid_region.

    Returns:
      Array of lp variables, each which range from 0 to infinity.
      Array is mapped so that variable for time-slice x is at index x.
      e.g. variable for first time-slice is variable[0]. variable for
      last time-slice is variable[-1]

    """

    solver = self.solver

    variables = []
    for t in self.time_index_iterable:
      var_name = '__'.join([name,
                            'grid_region_id',
                            str(grid_region_id),
                            'at_t',
                            str(t)])

      variables.append(solver.NumVar(0.0,
                                     solver.infinity(),
                                     var_name))
    return variables

  def declare_nameplate_variable(self, name, grid_region_id):
    """Declares a nameplate variable for a grid_region.

    Args:
      name: String to be included in the generated variable name.
      grid_region_id: Stringifyable object which identifies which grid
        these variables affect.

    Do Not call this function with the same (name, grid_region_id)
    pair more than once.  There may not be identically named variables
    in the same grid_region.

    Returns:
      A lp variable which values range from 0 to infinity.

    """

    nameplate_name = '__'.join([name,
                                'grid_region_id', str(grid_region_id),
                                'peak'])

    solver = self.solver
    return solver.NumVar(0.0,
                         solver.infinity(),
                         nameplate_name)


def extrapolate_cost(cost, discount_rate, time_span_1, time_span_2):
  """Extrapolate cost from one time span to another.

  Args:
    cost: cost incurred during time_span_1 (in units of currency)
    discount_rate: rate that money decays, per year (as decimal, e.g., .06)
    time_span_1: time span when cost incurred (in units of years)
    time_span_2: time span to extrapolate cost (in units of years)

  Returns:
    Cost extrapolated to time_span_2, units of currency.

  Model parameters are costs over time spans. For example, the demand
  may be a time series that lasts 1 year. The variable cost to fulfill
  that demand would then be for 1 year of operation. However, the
  GridModel is supposed to compute the total cost over a longer time
  span (e.g., 30 years).

  If there were no time value of money, the extrapolated cost would be
  the ratio of time_span_2 to time_span_1 (e.g., 30 in the
  example). However, payments in the future are less costly than
  payments in the present.  We extrapolate the cost by first finding
  the equivalent continuous stream of payments over time_span_1 that
  is equivalent to the cost, then assume that stream of payments
  occurs over time_span_2, instead.

  """
  growth_rate = 1.0 + discount_rate
  value_decay_1 = pow(growth_rate, -time_span_2)
  value_decay_2 = pow(growth_rate, -time_span_1)

  try:
    return cost * (1.0 - value_decay_1) / (1.0-value_decay_2)
  except ZeroDivisionError:
    return cost
