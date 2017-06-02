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


"""Partial Example of using grid_sim_linear_program to match website results.

First time users are encouraged to look at grid_sim_simple_example.py
first.

Analyzes energy similarly to how it was done for the backend to the
website http://energystrategies.org.  Note that this only calculates
for one out of the thirteen regions classified by the EIA.  To match
the results from the site, you must sum the results from all 13
regions as defined in REGION_HUBS.keys()

This backend uses a linear program to optimize the cheapest way to
generate electricity subject to certain constraints.  The constraints
are:
  1. Hourly generation must always exceed demand.  AKA.  Keep the lights on.
  2. Solution must have some percentage [0, 100] of power from a
     'Renewable Portfolio Standard' (RPS).  Different RPS have
     different standards so the code allows the user to specify which
     sources to put in the RPS.

Other factors which affect the linear program output include:
  1. Source fixed and variable costs. ($ / MW and $ / MWH respectively)
  2. Source CO2 per Mwh if there is a Carbon Tax (Tonnes / MWh)
  3. Carbon Tax Value ($ / Tonne)
"""

import math

import os.path as osp

import grid_sim_linear_program as gslp
import grid_sim_simple_example as simple

import pandas as pd


# Where to base a HVDC line.
# key:region, value: (latitude, longitude) in decimal degrees
REGION_HUBS = {'southwest': (33.4484, 112.0740),  # Phoenix
               'midatlantic': (40.4406, 79.9959),  # Pittsburgh
               'tva': (36.1627, 86.7816),  # Nashville
               'midwest': (39.0997, 94.5786),  # Kansas City
               'nyiso': (40.7128, 74.0059),  # New York City
               'ercot': (30.2672, 97.7431),  # Austin
               'northwest': (47.6062, 122.3321),  # Seattle
               'central': (41.2524, 95.9980),  # Omaha
               'carolinas': (35.2271, 80.8431),  # Charlotte
               'southeast': (33.5207, 86.8025),  # Birmingham
               'florida': (33.5207, 86.8025),  # Orlando
               'neiso': (33.5207, 86.8025),  # Boston
               'california': (34.0522, 118.2437)}  # Los Angeles


def circle_route_distance(coordinates1, coordinates2):
  """Calculate the spherical distance between two coordinates on earth.

  As described in https://en.wikipedia.org/wiki/Great-circle_distance

  Args:
    coordinates1: A tuple containint (latitude, longitude) in decimal
      degrees of point 1.
    coordinates2: A tuple containint (latitude, longitude) in decimal
      degrees of point 2.

  Returns:
    Distance on earth in kilometers between the two points.
  """

  lat1, lon1 = coordinates1
  lat2, lon2 = coordinates2

  # Convert to radians.
  lat1 *= math.pi / 180.
  lon1 *= math.pi / 180.
  lat2 *= math.pi / 180.
  lon2 *= math.pi / 180.

  delta_lat = abs(lat2 - lat1)
  delta_lon = abs(lon2 - lon1)

  # Result angle (asin) cannot be computed if point is exactly on
  # opposite side of earth, so special case out that point.  All other
  # values can be correctly calculated with the haversine, even values
  # close to 0, pi.
  if delta_lat == math.pi and delta_lon == math.pi:
    result_angle = math.pi
  else:
    haversine_lat = math.sin(delta_lat / 2) ** 2
    haversine_lon = math.sin(delta_lon / 2) ** 2
    result_angle = 2 * math.asin(math.sqrt(
        haversine_lat + math.cos(lat1) * math.cos(lat2) * haversine_lon))

  earth_radius_km = 6371
  return earth_radius_km * result_angle


def get_transmission_cost_efficiency(coordinates1, coordinates2):
  """Return transmission capital cost $/MW and transmission efficiency.

  Args:
    coordinates1: A tuple containint (latitude, longitude) in decimal
      degrees of point 1.
    coordinates2: A tuple containint (latitude, longitude) in decimal
      degrees of point 2.

  Returns:
    A tuple containing (cost $/MW, efficiency) elements.
  """

  # Assumptions: HVDC is 97% efficient / 1000 km with
  # Costs: $334 / km / MWh

  distance_km = circle_route_distance(coordinates1, coordinates2)
  capital_cost_per_mw = distance_km * 334

  efficiency = pow(0.97, distance_km / 1000)

  return (capital_cost_per_mw, efficiency)


def configure_sources_and_storage(profile_directory,
                                  region,
                                  source_dataframe,
                                  storage_dataframe,
                                  source_dict_index,
                                  storage_names,
                                  rps_names,
                                  hydrolimits=None):
  """Generates a LinearProgramContainer similar to the website.

  Args:
    profile_directory: String filepath where profile files of the form
      profile_<region>.csv exist.  Acceptable values of <region> are
      REGION_HUBS.keys().

    region: String name of the region to generate simulation for.
      Acceptable values are REGION_HUBS.keys().

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
    A Configured LinearProgramContainer suitable for simulating.
  """
  profiles_file = osp.join(profile_directory, 'profiles_usa.csv')

  profiles_dataframe = pd.read_csv(profiles_file, index_col=0, parse_dates=True)

  lp = gslp.LinearProgramContainer(profiles_dataframe)

  # Specify grid load or demand which has a profile in
  # profile_dataframe.<region>_DEMAND
  lp.add_demands(gslp.GridDemand('%s_DEMAND' % region.upper()))

  # Configure dispatchable and non-dispatchable sources.
  for source_name, source_index in source_dict_index.iteritems():
    dataframe_row = source_dataframe.loc['%s_%d' % (source_name, source_index)]
    is_rps_source = source_name in rps_names

    # Adjust source_name by region to match profiles and to
    # differentiate it from sources from other regions.
    regional_source_name = '%s_%s' % (region.upper(), source_name)
    source = gslp.GridSource(name=regional_source_name,
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
    if regional_source_name in profiles_dataframe.columns:
      lp.add_nondispatchable_sources(source)
    else:
      lp.add_dispatchable_sources(source)

  # Add Solar and Wind from other regions.  Adjust by additional
  # transmission costs and efficiency losses for distance traveled if
  # Solar and Wind are in the simulation.  The website assumes Solar
  # and Wind are always present.

  for other_region in REGION_HUBS:
    if other_region != region:
      transmission_cost, efficiency = get_transmission_cost_efficiency(
          REGION_HUBS[region],
          REGION_HUBS[other_region])

      for source in ['SOLAR', 'WIND']:
        if source in source_dict_index:
          cost_row = source_dataframe.loc['%s_%d'
                                          % (source,
                                             source_dict_index[source])]
          lp.add_nondispatchable_sources(
              gslp.GridSource(
                  name='%s_%s' % (other_region.upper(), source),
                  nameplate_unit_cost=cost_row['fixed'] + transmission_cost,
                  variable_unit_cost=cost_row['variable'],
                  co2_per_electrical_energy=cost_row['CO2'],
                  is_rps_source=True,
                  power_coefficient=efficiency)
          )

  for storage_name in storage_names:
    dataframe_row = storage_dataframe.loc[storage_name]
    storage = gslp.GridRecStorage(
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
  source_cost_path = data_dir + ['costs', 'source_costs.csv']
  storage_cost_path = data_dir + ['costs', 'storage_costs.csv']
  hydrolimits_path = data_dir + ['costs', 'regional_hydro_limits.csv']
  profile_path = data_dir + ['profiles']

  source_costs_file = osp.join(*source_cost_path)
  storage_costs_file = osp.join(*storage_cost_path)
  hydro_limits_file = osp.join(*hydrolimits_path)

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

  profile_directory = osp.join(*profile_path)
  lp = configure_sources_and_storage(
      region=region,
      profile_directory=profile_directory,
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
