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

"""Linear Program Constraints to simulate different energy assumptions.

This package was originally written to support understanding of energy
solutions within Google Research.  It has subsequently been used to
generate scenarios for a website

http://energystrategies.org

to help show policy makers, and other interested people, the cost
optimal way to reduce CO2 emissions from the electricity sector.  This
package has been released under open-source in order to fully document
how the results of the website were determined and to help further
understanding and analysis of energy costs.  See ../LICENSE for more
details of the release.

The reason for using this model is to avoid the limitations of
Levelized Cost of Electricity (LCOE) which is a popular metric for
computing costs within the energy sector.  The idea behind LCOE is to
find an equivalent cost between different energy sources which have
different capital, fixed and operating costs, and which have different
capacity factors (Percentage of the time when the sources can provide
power e.g. solar power during the day.)

A major limitation of LCOE is that it makes the assumption that
electricity is fungible, that a kilowatt-hour generated during the day
is equivalent to a kilowatt-hour generated at night.  This assumption
falls apart in the real world.  As of 2016, there are many articles
which say that solar is cheaper than fossil fuels.  Yet due to solar's
intermittency, other energy sources must be built to support nighttime
and cloudy day use.  This fundamentally changes the cost optimization
in a way which LCOE does not account for.  Since the other energy
sources must be built anyway to support nighttime use, then the proper
analysis would be to compare solar capital costs for power during the
day against the marginal cost of running the other energy sources
during the day.  LCOE does not consider this.  This package does.

This package does an hour by hour cost optimization of capital,
variable and operating costs of various different energy sources.  It
uses historical hour-by-hour data to determine actual overall demand
and actual power available for intermittent power sources such as
solar and wind power.

This package uses a linear program (LP) to optimize capital and
variable costs subject to the following basic constraint:

  - Total supply power must meet or exceed total demand for all hours.
      (Colloquially called 'Keeping the lights on.')

Additionally optional constaint is:

  - Renewable Portfolio Standard (RPS): Most states in the US have an
      RPS in place.  It requires that a certain percentage of the
      total power grid come from certain sources (such as wind or
      solar).  This package allows selection of RPS percentage as well
      as which sources should be in the RPS.

Power Sources specify capital, operating costs, maximum power, energy
limits and co2 emitted.  Overall the simulation may also specify a
carbon tax and discount rate of money to determine the current value
of future costs while simulating different cost assumptions.
"""
