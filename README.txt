===========
Grid Sim
===========

Grid Sim provides a Linear Program based solution for different types
of energy sources.  It mainly differs from other simulators in that it
uses actual hour-by-hourprofile data for non-dispatchable sources,
such as wind and solar.  This allows for more reliable simulation than
simple LCOE (Levelized Cost of Energy) simulations which assume the
fungibility of energy.  E.g. An LCOE analysis wouldn't consider that
solar power cannot provide power in the darkest of night.

This logic provided the basis for the paper "Analyzing energy
technologies and policies using DOSCOE by John C. Platt and J. Orion
Pritchard".

It is also the basis for data shown on the website
http://energystrategies.org

===========
Installation
===========

Dependencies:
  Grid Sim is dependent on the following packages.
  - numpy
  - pandas
  - protobuf
  - ortools

  All packages are downloadable via PyPi.  When testing installation
  on a linux workstation a simple pip-install of the tarball installed
  everything except for ortools.  Since the recent bundles of ortools
  were in egg form, installing ortools via easy-install proved a
  suitable workaround for this problem.
  
===========
Source Tree
===========

- grid_sim_linear_program.py: Energy Grid objects wrapped around a linear-program
- grid_configuration_pb2.py: compiled protobufs for grid_sim_linear_program.py
- grid_sim_simple_example.py: Executable python script which runs a simple grid simulation.
- grid_sim_website_example.py: Executable python script which runs a
    grid simulation using the input data on the energystrategies.org website.
+ data: data used to generate scenarios for website simulations
  + profiles:
    - profiles_<region>.csv: source and demand profiles for a particular
      electrical region.  Regions are taken as close as possible from
      the EIA definition of the regions.  See
      http://www.eia.gov/beta/realtime_grid/ for a map of the regions.
      Solar and Wind data were take from NREL data which was
      identified by state.  Since some of the EIA regions do not
      perfectly overlap states, bifurcated states were split up into
      EIA regions.  Profile columns are the energy source type.
      Profile rows are hour of the year. Values are in Megawatts for
      the hour.
      
  + costs:
    - costs.csv: source costs as used by the website.  Rows are of the
      form <SOURCE_TYPE>_<COST_INDEX> where <COST_INDEX> matches the
      cost sliders in the website.  Some additional sources are
      included here which ultimately were not shown on the website.
      For the website, all Carbon Capture and Sequestration (CCS) was
      done using <SOURCE_TYPE>_CRYO_<COST_INDEX>.  Columns are

      CO2: CO2 tonnes emitted per Megawatt of energy.

      fixed: Combination of capital and yearly fixed operating costs per
      Megawatt of capacity.

      variable: Costs per Megawatt-hour of generation. Primarily fuel costs.

    - storage_costs.csv: (TODO:(orionp) add storage costs.)
      


