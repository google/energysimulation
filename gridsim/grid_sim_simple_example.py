# Copyright 2017 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Example entry-point for using grid_sim_linear_program to analyize energy.

Populates and runs the linear program for a simple case of solar power
and natural gas power.
"""

import os.path as osp

import distutils.sysconfig as sysconfig

import grid_sim_linear_program as gslp

import pandas as pd


def get_data_directory():
  """Returns a path to grid_sim data in site-lib packages."""

  abs_package_path = osp.dirname(gslp.__file__)
  return [abs_package_path, 'data']


def simple_lp(profile_dataframe):
  """Builds a simple LP with natural gas and solar.

  Args:
   profile_dataframe: A pandas dataframe with source and demand names
     in the column header.  Hour of year in indexed column-0.

  Returns:
    A LinearProgramContainer with natural gas and solar sources.
  """

  lp = gslp.LinearProgramContainer(profile_dataframe)


  # Must specify a demand to put proper load on the grid.  GridDemand
  # is linked to the corresponding profile:
  # lp.profiles[GridDemand.name]

  lp.add_demands(gslp.GridDemand('DEMAND'))

  # Nondispatchable sources are intermittent and have profiles which
  # describe their availability.  Building more capacity of these
  # sources scales the profile but cannot provide power when the
  # profile is 0. (e.g. Solar power in the middle of the night, Wind
  # power during a lull)).  A nondispatchable GridSource is linked to
  # its corresponding profile, profile[GridSource.name]

  lp.add_nondispatchable_sources(
      gslp.GridSource(
          name='SOLAR',  # Matches profile column name for nondispatch.
          nameplate_unit_cost=946000,  # Aggressive solar cost $/MW
          variable_unit_cost=0,  # No fuel cost.
          co2_per_electrical_energy=0,  # Clean energy.
          is_rps_source=True))  # In Renewable Portfolio Standard

  # Dispatchable sources can provide power at any-time.  The LP will
  # optimize generation from these sources on an as-needed basis
  # hour-by-hour.

  lp.add_dispatchable_sources(
      gslp.GridSource(
          name='NG',  # Dispatchable, so no name restriction.
          nameplate_unit_cost=1239031,  # Cost for a combined cycle plant. $/MW
          variable_unit_cost=17.5,  # Cheap fuel costs assumes fracking. $/MWh
          co2_per_electrical_energy=0.33,  # Tonnes CO2 / MWh
          is_rps_source=False))  # Not in Renewable Portfolio Standard.

  return lp


def adjust_lp_policy(lp,
                     carbon_tax=0,
                     renewable_portfolio_percentage=30,
                     annual_discount_rate=0.06,
                     lifetime_in_years=30):
  """Configures the LP based upon macro-economic costs and policy.

  Args:
    lp: LinearProgramContainer
    carbon_tax: Float cost of emitting co2 in $ / Tonne-of-CO2.
      Default is no carbon tax.

    renewable_portfolio_percentage: 0. <= Float <= 100. Percentage of
      total generation which must come from sources in the Renewable
      Portfolio Standard.  LP may not converge if this is set to a
      high amount without any storage elements in the LP.  Default is
      30%.

    annual_discount_rate: interest rate used in discounted cash flow
      analysis to determine the present value of future costs. Default
      value is 6% (0.06).

    lifetime_in_years: Float Number of years over which fuel costs
      are paid off.  Default is 30 years.
  """

  hours_per_year = 24 * 365
  lp.carbon_tax = carbon_tax
  lp.rps_percent = renewable_portfolio_percentage
  lp.cost_of_money = gslp.extrapolate_cost(
      1.0,
      annual_discount_rate,
      lp.number_of_timeslices / hours_per_year,
      lifetime_in_years)


def display_lp_results(lp):
  """Prints out costs, generation and co2 results for the lp.

  Must be run after lp.solve() is called.

  Args:
    lp: LinearProgramContainer containing sources and storage.
  """

  system_cost = 0
  system_co2 = 0

  # Loop over sources and display results.
  sources = lp.sources
  for source in sources:
    capacity = source.get_nameplate_solution_value()
    generated = sum(source.get_solution_values())
    co2 = source.co2_per_electrical_energy * generated
    capital_cost = source.nameplate_unit_cost * capacity
    fuel_cost = (source.variable_unit_cost * generated +
                 co2 * lp.carbon_tax) * lp.cost_of_money
    total_source_cost = capital_cost + fuel_cost

    print """SOURCE: %s
  Capacity: %.2f Megawatts
  Generated: %.2f Megawatt-hours
  Emitted: %.2f Tonnes of CO2
  Capital Cost: $%.2f
  Fuel Cost: $%.2f
  Total Cost: $%.2f""" % (source.name,
                          capacity,
                          generated,
                          co2,
                          capital_cost,
                          fuel_cost,
                          total_source_cost)

    system_cost += total_source_cost
    system_co2 += co2

  # Loop over storage and display results.
  for storage in lp.storage:
    capacity = storage.get_nameplate_solution_value()
    unused_stored = storage.get_solution_values()
    charge_capacity = storage.sink.get_nameplate_solution_value()
    discharge_capacity = storage.source.get_nameplate_solution_value()
    storage_cost = sum(
        [capacity * storage.storage_nameplate_cost,
         charge_capacity * storage.sink.nameplate_unit_cost,
         discharge_capacity * storage.source.nameplate_unit_cost])
    print """STORAGE: %s
  Capacity: %.2f Megawatt-hours
  Maximum Charge Power: %.2f Megawatts
  Maximum Discharge Power: %.2f Megawatts
  Total Cost: $%.2f""" % (storage.name,
                          capacity,
                          charge_capacity,
                          discharge_capacity,
                          storage_cost)
    system_cost += storage_cost
    system_co2 += co2

  print 'SYSTEM_COST: $%.2f' % system_cost
  print 'SYSTEM_CO2: %.2f Tonnes' % system_co2


def main():

  profiles_path = get_data_directory() + ['profiles', 'profiles_california.csv']
  profiles_file = osp.join(profiles_path)

  profiles = pd.read_csv(profiles_file, index_col=0, parse_dates=True)
  lp = simple_lp(profiles)
  adjust_lp_policy(lp)

  if not lp.solve():
    raise ValueError("""LP did not converge.
Failure to solve is usually because of high RPS and no storage.""")

  display_lp_results(lp)


if __name__ == '__main__':
  main()
