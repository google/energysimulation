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


"""Example entry-point for using grid_sim_linear_program.

Analyzes energy as it was done by the website
http://energystrategies.org.

First time users are encouraged to look at grid_sim_simple_example.py
first.
"""


import os.path as osp

import grid_sim_linear_program as gslp
import grid_sim_simple_example as simple

import pandas as pd


def configure_sources_and_storage(profile_dataframe,
                                  source_dataframe,
                                  storage_dataframe,
                                  source_dict_index,
                                  storage_names,
                                  rps_names,
                                  hydrolimits=None):
  """Generates a LinearProgramContainer similar to the website.

  Args:
    profile_dataframe: A pandas dataframe with source and demand names
      in the column header.  Hour of year in indexed column-0.

    source_dataframe: A pandas dataframe with source names in indexed
      column-0.

      Index names match the source names in the website with a cost
      option index. (e.g. COAL_0, SOLAR_2, NUCLEAR_1).  Some sources
      only have one cost option associated with them.
      Source names are as follows.
        COAL, Coal fired plants.
        HYDROPOWER, Water turbines fed by a dammed reservoir.
        NGCC, Natural Gas Combined Cycle. A natural gas fired turbine
          whose exhaust heats a steam turbine for additional power.
        NGCT, Natural Gas Conventional Turbine. A natural gas turbine.
        NGCC_AMINE, NGCC + Carbon Capture and Sequestration via amine
          capture.
        NGCC_CRYO, NGCC + Carbon Capture and Sequestration from
          capturing the CO2 by freezing into dry-ice.
        NUCLEAR, Nuclear power.
        SOLAR, Solar power through utility scale photovoltaic panels.
        WIND, Wind power through modern conventional wind turbines.

      Column headers include
        CO2, Amount of CO2 emitted per generated energy. Units are
          Tonnes of CO2 Emitted per Megawatt-hour.
        fixed, Cost of building and maintaining a plant per
          nameplate-capacity of the plant.  Units are $ / Megawatt

        variable, Cost of running the plant based upon generation
          energy.  Units are $ / Megawatt-hour

    storage_dataframe: A pandas dataframe with storage names in indexed
      column-0.

      There are only two kinds of storage considered in this dataframe.
        HYDROGEN, Storage charges by generating and storing hydrogen.
          Storage discharges by buring hydrogen.
        ELECTROCHEMICAL, Storage charges and discharges through batteries.

      Index names match the storage names.

      Column headers include:
        fixed,
        charge_efficiency,
        discharge_efficiency,
        charge_capital,
        discharge_capital,

    source_dict_index: A dict keyed by source name with index values
      for cost.  Acceptable index values are either [0] for sources
      which only had one cost assumed in the webiste or [0,1,2] for
      sources which had 3 choices for cost assumption.

    storage_names: A list of storge names of storage type to add to
      the LP.  Must match names in the storage_dataframe index.

    rps_names: A list of source names which should be considered in
      the Renewable Portfolio Standard.

    hydrolimits: A dict with 'max_power' and 'max_energy' keys.  If
      specified, hydropower will be limited to this power and energy.

  Returns:
    A LinearProgramContainer suitable for simulating.
  """

  lp = gslp.LinearProgramContainer(profile_dataframe)

  # Specify grid load or demand which has a profile in profile_dataframe.DEMAND
  lp.add_demands(gslp.GridDemand('DEMAND'))

  # Configure dispatchable and non-dispatchable sources.
  for source_name, source_index in source_dict_index.iteritems():
    dataframe_row = source_dataframe.loc['%s_%d' % (source_name, source_index)]
    is_rps_source = source_name in rps_names

    source = gslp.GridSource(name=source_name,
                             nameplate_unit_cost=dataframe_row['fixed'],
                             variable_unit_cost=dataframe_row['variable'],
                             co2_per_electrical_energy=dataframe_row['CO2'],
                             is_rps_source=is_rps_source)

    # For energystrategies.org we assumed that the prime hydropower
    # sites have already been developed and built.  So for hydropower
    # sites, we make the capital cost 0.  Here we limit the LP to only
    # use as much power and energy as existing sites already provide.
    # Without this limitation and with capital cost of 0, the LP will
    # assume an infinite supply of cheap hydropower and fulfill demand
    # with 100% hydropower.

    if hydrolimits is not None:
      if source_name == 'HYDROPOWER':
        source.max_power = hydrolimits['max_power']
        source.max_energy = hydrolimits['max_energy']

    # Non-dispatchable sources have profiles associated with them.
    if source_name in profile_dataframe.columns:
      lp.add_nondispatchable_sources(source)
    else:
      lp.add_dispatchable_sources(source)

  for storage_name in storage_names:
    dataframe_row = storage_dataframe.loc[storage_name]
    storage = gslp.GridStorage(
        name=storage_name,
        storage_nameplate_cost=dataframe_row['fixed'],
        charge_nameplate_cost=dataframe_row['charge_capital'],
        discharge_nameplate_cost=dataframe_row['discharge_capital'],
        charge_efficiency=dataframe_row['charge_efficiency'],
        discharge_efficiency=dataframe_row['discharge_efficiency'])

    lp.add_storage(storage)

  return lp


def main():

  region = 'california'

  data_dir = simple.get_data_directory()
  profiles_path = data_dir + ['profiles', 'profiles_%s.csv' % region]
  source_cost_path = data_dir + ['costs', 'source_costs.csv']
  storage_cost_path = data_dir + ['costs', 'storage_costs.csv']
  hydrolimits_path = data_dir + ['costs', 'regional_hydro_limits.csv']

  profiles_file = osp.join(*profiles_path)
  source_costs_file = osp.join(*source_cost_path)
  storage_costs_file = osp.join(*storage_cost_path)
  hydro_limits_file = osp.join(*hydrolimits_path)

  profiles_dataframe = pd.read_csv(profiles_file, index_col=0, parse_dates=True)
  source_costs_dataframe = pd.read_csv(source_costs_file, index_col=0)
  storage_costs_dataframe = pd.read_csv(storage_costs_file, index_col=0)
  hydrolimits_dataframe = pd.read_csv(hydro_limits_file, index_col=0)

  ng_cost_index = 0
  cost_settings = {
      'COAL': 0,
      'HYDROPOWER': 0,
      'NGCC': ng_cost_index,
      'NGCT': ng_cost_index,
      'NGCC_CRYO': ng_cost_index,
      'WIND': 2,
      'SOLAR': 2,
      'NUCLEAR': 2
  }

  storage_names = ['ELECTROCHEMICAL']
  rps_names = ['SOLAR', 'WIND']

  lp = configure_sources_and_storage(
      profile_dataframe=profiles_dataframe,
      source_dataframe=source_costs_dataframe,
      storage_dataframe=storage_costs_dataframe,
      source_dict_index=cost_settings,
      storage_names=storage_names,
      rps_names=rps_names,
      hydrolimits=hydrolimits_dataframe.loc[region]
  )

  simple.adjust_lp_policy(
      lp,
      carbon_tax=50,  # $50 per tonne
      renewable_portfolio_percentage=20,  # 20% generated from rps_names
      annual_discount_rate=0.06,  # 6% annual discount rate.
      lifetime_in_years=30)  # 30 year lifetime

  print 'Solving may take a few minutes...'
  if not lp.solve():
    raise ValueError("""LP did not converge.
Failure to solve is usually because of high RPS and no storage.""")

  simple.display_lp_results(lp)


if __name__ == '__main__':
  main()
