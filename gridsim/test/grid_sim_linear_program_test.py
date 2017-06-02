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

"""Tests for grid_sim_linear_program."""

import math

import unittest

import gridsim.grid_sim_linear_program as gslp

from gridsim.grid_sim_linear_program import DemandNotSatisfiedError
from gridsim.grid_sim_linear_program import GridDemand
from gridsim.grid_sim_linear_program import GridRecStorage
from gridsim.grid_sim_linear_program import GridSource
from gridsim.grid_sim_linear_program import GridStorage
from gridsim.grid_sim_linear_program import LinearProgramContainer
from gridsim.grid_sim_linear_program import RpsExceedsDemandError
from gridsim.grid_sim_linear_program import RpsPercentNotMetError

import numpy as np
import numpy.testing as npt
import pandas as pd


DEMAND = 'DEMAND'
NG = 'NG'
NG2 = 'NG2'
TIME = 'TIME'
SOLAR = 'SOLAR'
WIND = 'WIND'
NUCLEAR = 'NUCLEAR'
STORAGE = 'STORAGE'


class CheckValidationTest(unittest.TestCase):
  """Check proper validation of bogus / non bogus values."""

  def setUp(self):
    self.dummy_profile = pd.DataFrame({'bogus_source': np.ones(4)})

  def testUnreasonableProfileRejected(self):
    """Verify profile exists."""

    # Test len(profile) == 0 handled properly.
    with self.assertRaises(ValueError):
      unused_lp = LinearProgramContainer(pd.DataFrame())

    # Test profile is None case handled properly.
    with self.assertRaises(ValueError):
      unused_lp = LinearProgramContainer(None)

  def testDemandMatchesProfile(self):
    """Verify lp checks for demand profiles."""

    with self.assertRaises(KeyError):
      lp = LinearProgramContainer(self.dummy_profile)
      lp.add_demands(
          GridDemand('Some Demand', 0),
          GridDemand('Other Demand', 1)
      )
      self.assertTrue(lp.solve())

  def testAddDispatchableSourceWithProfile(self):
    """Ensure Dispatchable Source with profile gets error."""

    with self.assertRaises(KeyError):
      lp = LinearProgramContainer(pd.DataFrame({NG: np.ones(4)}))
      lp.add_dispatchable_sources(GridSource(NG, 1e6, 1e6))

  def testAddNonDispatchableSourceWithoutProfile(self):
    """Ensure NonDispatchable Source without profile gets error."""

    with self.assertRaises(KeyError):
      lp = LinearProgramContainer(self.dummy_profile)
      lp.add_nondispatchable_sources(GridSource(NG, 1e6, 1e6))

  def testSolutionValuesCalledBeforeSolve(self):
    lp = LinearProgramContainer(self.dummy_profile)
    ng = GridSource(NG, 1e6, 1e6)
    lp.add_dispatchable_sources(ng)
    with self.assertRaises(RuntimeError):
      ng.get_solution_values()

    with self.assertRaises(RuntimeError):
      ng.get_nameplate_solution_value()


class FourTimeSliceTest(unittest.TestCase):

  def setUp(self):
    self.demand_profile = np.array([3.0, 0.0, 0.0, 3.0])

    rng = pd.date_range('1/1/2011', periods=len(self.demand_profile), freq='H')

    df = pd.DataFrame.from_dict(
        {DEMAND: self.demand_profile,
         SOLAR: np.array([2.0, 0.0, 0.0, 1.0]),
         WIND: np.array([1.0, 0.0, 0.0, 0.0]),
         TIME: rng}
    )

    self.profiles = df.set_index(TIME,
                                 verify_integrity=True)

    self.lp = LinearProgramContainer(self.profiles)
    self.lp.add_demands(GridDemand(DEMAND))

    self.solar = GridSource(SOLAR, 1e6, 1e6)
    self.ng = GridSource(NG, 1e6, 1e6)


class ProfileVerification(unittest.TestCase):
  """Test we properly validate profiles."""

  def setUp(self):
    self.df = pd.DataFrame(np.arange(24.0).reshape(4, 6))

  def testOk(self):
    """Test an okay profile is passed without error."""

    LinearProgramContainer(self.df)

  def testNan(self):
    """Test Nan is caught."""

    self.df[self.df > 8] = np.nan
    with self.assertRaises(ValueError):
      LinearProgramContainer(self.df)

  def testNegative(self):
    """Test Negative is caught."""

    self.df -= 4
    with self.assertRaises(ValueError):
      LinearProgramContainer(self.df)


class TestInitialization(FourTimeSliceTest):
  """Test initialization routines work as expected."""

  def testCostConstraintCreation(self):
    """Verify cost constraint gets created upon initialization."""
    lp = self.lp

    ng = GridSource(NG, 1e6, 1e6)
    lp.add_dispatchable_sources(ng)
    self.assertIsNone(lp.minimize_costs_objective)
    lp._initialize_solver()
    self.assertIsNotNone(lp.minimize_costs_objective)

  def testVariableCreation(self):
    """Check GridSource creates nameplate / timeslice variables."""
    lp = self.lp
    ng = GridSource(NG, 1e6, 1e6)
    self.assertIsNone(ng.timeslice_variables)
    self.assertIsNone(ng.nameplate_variable)

    lp.add_dispatchable_sources(ng)

    lp._initialize_solver()
    self.assertIsNotNone(ng.nameplate_variable)
    self.assertEqual(len(ng.timeslice_variables), lp.number_of_timeslices)

  def testConservePowerConstraint(self):
    """Verify ConservePowerConstraint created upon demand initialization."""
    # try a non-zero grid_region_id
    lp = self.lp

    demand = GridDemand(DEMAND, 4)
    lp.add_demands(demand)

    self.assertEqual(len(lp.conserve_power_constraint), 0)
    lp._initialize_solver()

    self.assertEqual(len(lp.conserve_power_constraint), 2)
    self.assertTrue(demand.grid_region_id in lp.conserve_power_constraint)
    self.assertEqual(len(lp.conserve_power_constraint[demand.grid_region_id]),
                     lp.number_of_timeslices)


class FourTimeSliceLpResults(FourTimeSliceTest):
  """Tests over 4 timeslices which return results."""

  def testDispatchableOnly(self):
    """One Dispatchable Source should fill demand."""

    # Test over a few different profiles
    offset_sin_wave = np.sin(np.linspace(0, 2 * math.pi, 4)) + 2
    profiles = [self.demand_profile,
                np.zeros(4),
                np.ones(4),
                np.arange(4) % 2,  # 0,1,0,1
                offset_sin_wave]

    for profile in profiles:
      lp = LinearProgramContainer(pd.DataFrame({DEMAND: profile}))
      lp.add_demands(GridDemand(DEMAND))

      ng = self.ng
      lp.add_dispatchable_sources(ng)
      self.assertTrue(lp.solve())
      npt.assert_almost_equal(ng.get_solution_values(), profile)

      self.assertAlmostEqual(ng.get_nameplate_solution_value(),
                             max(profile))

  def testCheaperDispatchableOnly(self):
    """Two Dispatchable Sources. Cheaper one should fill demand."""

    lp = self.lp
    ng1 = self.ng
    ng2 = GridSource(NG2, 2e6, 2e6)
    lp.add_dispatchable_sources(ng1, ng2)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(ng1.get_solution_values(), self.demand_profile)
    npt.assert_almost_equal(ng2.get_solution_values(), np.zeros(4))

    self.assertAlmostEqual(ng1.get_nameplate_solution_value(),
                           max(self.demand_profile))

    self.assertAlmostEqual(ng2.get_nameplate_solution_value(), 0.0)

  def testMaxPowerCheaperDispatchable(self):
    """Two Dispatchable Sources. Cheaper one fills up to max_power."""

    lp = self.lp
    max_power = 1
    ng1 = GridSource(NG, 1e6, 1e6, max_power=max_power)
    ng2 = GridSource(NG2, 2e6, 2e6)
    lp.add_dispatchable_sources(ng1, ng2)
    self.assertTrue(lp.solve())

    max_power_profile = np.array([max_power, 0, 0, max_power])
    remaining = self.demand_profile - max_power_profile
    npt.assert_almost_equal(ng1.get_solution_values(), max_power_profile)
    npt.assert_almost_equal(ng2.get_solution_values(), remaining)

    self.assertAlmostEqual(ng1.get_nameplate_solution_value(),
                           max(max_power_profile))

    self.assertAlmostEqual(ng2.get_nameplate_solution_value(),
                           max(remaining))

  def testMaxEnergyCheaperDispatchable(self):
    """Two Dispatchable Sources. Cheaper one fills up to max_energy."""

    lp = self.lp
    max_energy = 1.0
    ng1 = GridSource(NG, 1e6, 1e6, max_energy=max_energy)
    ng2 = GridSource(NG2, 2e6, 2e6)
    lp.add_dispatchable_sources(ng1, ng2)
    self.assertTrue(lp.solve())

    demand_profiles = self.demand_profile
    demand_energy = sum(demand_profiles)
    max_energy_profile = (max_energy / demand_energy) * demand_profiles
    remaining = demand_profiles - max_energy_profile

    npt.assert_almost_equal(ng1.get_solution_values(), max_energy_profile)
    npt.assert_almost_equal(ng2.get_solution_values(), remaining)

    self.assertAlmostEqual(ng1.get_nameplate_solution_value(),
                           max(max_energy_profile))

    self.assertAlmostEqual(ng2.get_nameplate_solution_value(),
                           max(remaining))

  def testNonDispatchable(self):
    """Ensure a non-dispatchable source with proper profile can fill demand."""
    lp = self.lp

    solar = self.solar
    lp.add_nondispatchable_sources(solar)
    self.assertTrue(lp.solve())

    self.assertAlmostEqual(solar.get_nameplate_solution_value(), 6.0)
    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([6.0, 0, 0, 3.0]))

    self.profiles[DEMAND] = 1
    self.assertFalse(lp.solve())

  def testNonDispatchablePowerCoefficient(self):
    """Verify that a PowerCoefficient of 0.5 doubles the nameplate."""
    lp = self.lp

    solar = self.solar
    solar.power_coefficient = 0.5
    lp.add_nondispatchable_sources(solar)
    self.assertTrue(lp.solve())

    self.assertAlmostEqual(solar.get_nameplate_solution_value(), 12.0)
    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([6.0, 0, 0, 3.0]))

    self.profiles[DEMAND] = 1
    self.assertFalse(lp.solve())


class SimpleStorageTest(FourTimeSliceTest):
  """Preliminary tests of Storage with power and energy limits."""

  def testSimpleStorage(self):
    """Free Storage should backup Solar."""
    solar = GridSource(SOLAR, 2.0e6, 0)
    wind = GridSource(WIND, 5.0e6, 0)
    ng = GridSource(NG, 1.0e10, 0)
    storage = GridStorage(STORAGE, 0)

    lp = self.lp
    lp.add_nondispatchable_sources(solar, wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([4.0, 0, 0, 2.0]))

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([0, 1, 1, 1]))

    npt.assert_almost_equal(storage.source.get_solution_values(),
                            np.array([0, 0, 0, 1]))

    npt.assert_almost_equal(storage.sink.get_solution_values(),
                            np.array([1, 0, 0, 0]))

  def testSimpleStorageWithStorageLimit(self):
    """Free Storage should backup Solar."""
    solar = GridSource(SOLAR, 2.0e6, 0)
    wind = GridSource(WIND, 5.0e6, 0)
    ng = GridSource(NG, 1.0e10, 0)
    storage = GridStorage(STORAGE, 0, max_storage=0.5)

    lp = self.lp
    lp.add_nondispatchable_sources(solar, wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([5.0, 0, 0, 2.5]))

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([0, 0.5, 0.5, 0.5]))

    npt.assert_almost_equal(storage.source.get_solution_values(),
                            np.array([0, 0, 0, 0.5]))

    npt.assert_almost_equal(storage.sink.get_solution_values(),
                            np.array([0.5, 0, 0, 0]))

  def testSimpleStorageWithSourceLimit(self):
    """Tests Solar, Wind, Storage.  Solar limited so wind makes up the rest."""
    solar = GridSource(SOLAR, 2.0e6, 0, max_energy=3)
    wind = GridSource(WIND, 5.0e6, 0)
    ng = GridSource(NG, 1.0e10, 0)
    storage = GridStorage(STORAGE, 0)

    lp = self.lp
    lp.add_nondispatchable_sources(solar, wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([2.0, 0, 0, 1.0]))

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.array([3.0, 0, 0, 0.0]))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([0, 2.0, 2.0, 2.0]))

    npt.assert_almost_equal(storage.source.get_solution_values(),
                            np.array([0, 0, 0, 2.0]))

    npt.assert_almost_equal(storage.sink.get_solution_values(),
                            np.array([2.0, 0, 0, 0]))

  def testSimpleStorageWithChargeEfficiency(self):
    """Free Storage should backup Solar."""
    solar = GridSource(SOLAR, 2.0e6, 0)
    wind = GridSource(WIND, 5.0e6, 0)
    ng = GridSource(NG, 1.0e10, 0)
    storage = GridStorage(STORAGE, 0, charge_efficiency=0.5)

    lp = self.lp
    lp.add_nondispatchable_sources(solar, wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([4.5, 0, 0, 2.25]))

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([0, 0.75, 0.75, 0.75]))

    npt.assert_almost_equal(storage.source.get_solution_values(),
                            np.array([0, 0, 0, 0.75]))

    npt.assert_almost_equal(storage.sink.get_solution_values(),
                            np.array([1.5, 0, 0, 0]))

  def testSimpleStorageWithDischargeEfficiency(self):
    """Free Storage should backup Solar."""
    solar = GridSource(SOLAR, 2.0e6, 0)
    wind = GridSource(WIND, 5.0e6, 0)
    ng = GridSource(NG, 1.0e10, 0)
    storage = GridStorage(STORAGE, 0, discharge_efficiency=0.5)

    lp = self.lp
    lp.add_nondispatchable_sources(solar, wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([4.5, 0, 0, 2.25]))

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([0, 1.5, 1.5, 1.5]))

    npt.assert_almost_equal(storage.source.get_solution_values(),
                            np.array([0, 0, 0, 1.5]))

    npt.assert_almost_equal(storage.sink.get_solution_values(),
                            np.array([1.5, 0, 0, 0]))

  def testSimpleStorageWithStorageEfficiency(self):
    """Free Storage should backup Solar."""
    solar = GridSource(SOLAR, 2.0e6, 0)
    wind = GridSource(WIND, 5.0e6, 0)
    ng = GridSource(NG, 1.0e10, 0)

    se = math.pow(0.5, 1.0 / 3)
    storage = GridStorage(STORAGE, 0, storage_efficiency=se)

    lp = self.lp
    lp.add_nondispatchable_sources(solar, wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([4.5, 0, 0, 2.25]))

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([0, 1.5, 1.5 * se, 1.5 * se * se]))

    npt.assert_almost_equal(storage.source.get_solution_values(),
                            np.array([0, 0, 0, 0.75]))

    npt.assert_almost_equal(storage.sink.get_solution_values(),
                            np.array([1.5, 0, 0, 0]))


class SimpleRecStorageTest(FourTimeSliceTest):
  """Preliminary tests of Storage with power and energy limits."""

  def testSimpleStorage(self):
    """Free Storage should backup Solar."""
    solar = GridSource(SOLAR, 2.0e6, 0)
    wind = GridSource(WIND, 5.0e6, 0)
    ng = GridSource(NG, 1.0e10, 0)
    storage = GridRecStorage(STORAGE, 0)

    lp = self.lp
    lp.add_nondispatchable_sources(solar, wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([4.0, 0, 0, 2.0]))

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([0, 1, 1, 1]))

    npt.assert_almost_equal(storage.get_source_solution_values(),
                            np.array([-1, 0, 0, 1]))

  def testSimpleStorageWithStorageLimit(self):
    """Free Storage should backup Solar."""
    solar = GridSource(SOLAR, 2.0e6, 0)
    wind = GridSource(WIND, 5.0e6, 0)
    ng = GridSource(NG, 1.0e10, 0)
    storage = GridRecStorage(STORAGE, 0, max_storage=0.5)

    lp = self.lp
    lp.add_nondispatchable_sources(solar, wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([5.0, 0, 0, 2.5]))

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([0, 0.5, 0.5, 0.5]))

    npt.assert_almost_equal(storage.get_source_solution_values(),
                            np.array([-0.5, 0, 0, 0.5]))

  def testSimpleStorageWithSourceLimit(self):
    """Tests Solar, Wind, Storage.  Solar limited so wind makes up the rest."""
    solar = GridSource(SOLAR, 2.0e6, 0, max_energy=3)
    wind = GridSource(WIND, 5.0e6, 0)
    ng = GridSource(NG, 1.0e10, 0)
    storage = GridRecStorage(STORAGE, 0)

    lp = self.lp
    lp.add_nondispatchable_sources(solar, wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([2.0, 0, 0, 1.0]))

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.array([3.0, 0, 0, 0.0]))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([0, 2.0, 2.0, 2.0]))

    npt.assert_almost_equal(storage.get_source_solution_values(),
                            np.array([-2.0, 0, 0, 2.0]))

  def testSimpleStorageWithChargeEfficiency(self):
    """Free Storage should backup Solar."""
    solar = GridSource(SOLAR, 2.0e6, 0)
    wind = GridSource(WIND, 5.0e6, 0)
    ng = GridSource(NG, 1.0e10, 0)
    storage = GridRecStorage(STORAGE, 0, charge_efficiency=0.5)

    lp = self.lp
    lp.add_nondispatchable_sources(solar, wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([4.5, 0, 0, 2.25]))

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([0, 0.75, 0.75, 0.75]))

    npt.assert_almost_equal(storage.get_source_solution_values(),
                            np.array([-1.5, 0, 0, 0.75]))

  def testSimpleStorageWithDischargeEfficiency(self):
    """Free Storage should backup Solar."""
    solar = GridSource(SOLAR, 2.0e6, 0)
    wind = GridSource(WIND, 5.0e6, 0)
    ng = GridSource(NG, 1.0e10, 0)
    storage = GridRecStorage(STORAGE, 0, discharge_efficiency=0.5)

    lp = self.lp
    lp.add_nondispatchable_sources(solar, wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([4.5, 0, 0, 2.25]))

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([0, 1.5, 1.5, 1.5]))

    npt.assert_almost_equal(storage.get_source_solution_values(),
                            np.array([-1.5, 0, 0, 1.5]))

  def testSimpleStorageWithStorageEfficiency(self):
    """Free Storage should backup Solar."""
    solar = GridSource(SOLAR, 2.0e6, 0)
    wind = GridSource(WIND, 5.0e6, 0)
    ng = GridSource(NG, 1.0e10, 0)

    se = math.pow(0.5, 1.0 / 3)
    storage = GridRecStorage(STORAGE, 0, storage_efficiency=se)

    lp = self.lp
    lp.add_nondispatchable_sources(solar, wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([4.5, 0, 0, 2.25]))

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(4))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([0, 1.5, 1.5 * se, 1.5 * se * se]))

    npt.assert_almost_equal(storage.get_source_solution_values(),
                            np.array([-1.5, 0, 0, 0.75]))


class TwoTimeSliceTest(unittest.TestCase):
  """Tests with only two time slices."""

  def setUp(self):
    solar_profile = np.array([0.0, 1.0])
    wind_profile = np.array([1.0, 0.0])
    demand_profile = np.array([1.0, 1.0])
    self.demand_profile = demand_profile

    rng = pd.date_range('1/1/2011', periods=len(solar_profile), freq='H')
    df = pd.DataFrame.from_dict(
        {
            SOLAR: solar_profile,
            WIND: wind_profile,
            DEMAND: demand_profile,
            TIME: rng
        }
    )

    self.profiles = df.set_index(TIME,
                                 verify_integrity=True)

    self.lp = LinearProgramContainer(self.profiles)
    self.lp.add_demands(GridDemand(DEMAND))


class CircularStorageTest(TwoTimeSliceTest):
  """Tests related to last storage result affecting first storage result."""

  def testCircularStorageLastHourSource(self):
    """Verify that storage from last hour affects first hour."""
    solar = GridSource(SOLAR, 2.0e6, 0)
    storage = GridStorage(STORAGE, 0)

    lp = self.lp
    lp.add_nondispatchable_sources(solar)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([0.0, 2.0]))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([1.0, 0.0]))

    npt.assert_almost_equal(storage.source.get_solution_values(),
                            np.array([1.0, 0.0]))

    npt.assert_almost_equal(storage.sink.get_solution_values(),
                            np.array([0.0, 1.0]))

  def testCircularStorageFirstHourSource(self):
    """Verify that storage from first hour affects last hour."""
    wind = GridSource(WIND, 2.0e6, 0)
    storage = GridStorage(STORAGE, 0)

    lp = self.lp
    lp.add_nondispatchable_sources(wind)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.array([2.0, 0.0]))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([0.0, 1.0]))

    npt.assert_almost_equal(storage.source.get_solution_values(),
                            np.array([0.0, 1.0]))

    npt.assert_almost_equal(storage.sink.get_solution_values(),
                            np.array([1.0, 0.0]))

  def testFreeStorageMeansCheapestSource(self):
    """Verify that free storage selects the cheapest energy supply."""
    solar = GridSource(SOLAR, 2.0e6, 0)
    wind = GridSource(WIND, 2.2e6, 0)
    storage = GridStorage(STORAGE, 0)

    lp = self.lp
    lp.add_nondispatchable_sources(solar, wind)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.zeros(2))

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([0.0, 2.0]))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([1.0, 0.0]))

    npt.assert_almost_equal(storage.source.get_solution_values(),
                            np.array([1.0, 0.0]))

    npt.assert_almost_equal(storage.sink.get_solution_values(),
                            np.array([0.0, 1.0]))


class CircularRecStorageTest(TwoTimeSliceTest):
  """Tests related to last storage result affecting first storage result."""

  def testCircularRecStorageLastHourSource(self):
    """Verify that storage from last hour affects first hour."""
    solar = GridSource(SOLAR, 2.0e6, 0)
    storage = GridRecStorage(STORAGE, 0)

    lp = self.lp
    lp.add_nondispatchable_sources(solar)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([0.0, 2.0]))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([1.0, 0.0]))

    npt.assert_almost_equal(storage.get_source_solution_values(),
                            np.array([1.0, -1.0]))

  def testCircularRecStorageFirstHourSource(self):
    """Verify that storage from first hour affects last hour."""
    wind = GridSource(WIND, 2.0e6, 0)
    storage = GridRecStorage(STORAGE, 0)

    lp = self.lp
    lp.add_nondispatchable_sources(wind)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.array([2.0, 0.0]))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([0.0, 1.0]))

    npt.assert_almost_equal(storage.get_source_solution_values(),
                            np.array([-1.0, 1.0]))

  def testFreeRecStorageMeansCheapestSource(self):
    """Verify that free storage selects the cheapest energy supply."""
    solar = GridSource(SOLAR, 2.0e6, 0)
    wind = GridSource(WIND, 2.2e6, 0)
    storage = GridRecStorage(STORAGE, 0)

    lp = self.lp
    lp.add_nondispatchable_sources(solar, wind)
    lp.add_storage(storage)
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.zeros(2))

    npt.assert_almost_equal(solar.get_solution_values(),
                            np.array([0.0, 2.0]))

    npt.assert_almost_equal(storage.get_solution_values(),
                            np.array([1.0, 0.0]))

    npt.assert_almost_equal(storage.get_source_solution_values(),
                            np.array([1.0, -1.0]))


class StorageCostsTest(TwoTimeSliceTest):
  """Test Solutions with different storage costs."""

  def testStorageNameplateCost(self):
    """Keep increasing storage costs until ng finally wins out."""
    wind = GridSource(WIND, 1.0e6, 0)
    storage = GridStorage(STORAGE, 0)
    ng = GridSource(NG, 4.6e6, 0)
    lp = self.lp
    lp.add_nondispatchable_sources(wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)

    self.assertTrue(lp.solve())

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.array([2.0, 0.0]))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(2))

    # Change costs.  Total wind + storage cost is now (2 + 1)E6.
    # Still less than ng costs.
    storage.charge_nameplate_cost = 1.0e6
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.array([2.0, 0.0]))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(2))

    # Change Costs. Total wind + storage cost is now (2 + 1 + 1)E6.
    # Still less than ng costs.
    storage.storage_nameplate_cost = 1.0e6
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.array([2.0, 0.0]))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(2))

    # Change costs.  Total wind + storage cost is now
    # (2 + 1 + 1 + 1)E6.  Now more than ng costs.
    storage.discharge_nameplate_cost = 1.0e6
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.zeros(2))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.array([1.0, 1.0]))


class RecStorageCostsTest(TwoTimeSliceTest):
  """Test Solutions with different storage costs."""

  def testStorageNameplateCost(self):
    """Keep increasing storage costs until ng finally wins out."""
    wind = GridSource(WIND, 1.0e6, 0)
    storage = GridRecStorage(STORAGE, 0)
    ng = GridSource(NG, 4.6e6, 0)
    lp = self.lp
    lp.add_nondispatchable_sources(wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)

    self.assertTrue(lp.solve())

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.array([2.0, 0.0]))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(2))

    # Change costs.  Total wind + storage cost is now (2 + 1)E6.
    # Still less than ng costs.
    storage.charge_nameplate_cost = 1.0e6
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.array([2.0, 0.0]))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(2))

    # Change Costs. Total wind + storage cost is now (2 + 1 + 1)E6.
    # Still less than ng costs.
    storage.storage_nameplate_cost = 1.0e6
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.array([2.0, 0.0]))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.zeros(2))

    # Change costs.  Total wind + storage cost is now
    # (2 + 1 + 1 + 1)E6.  Now more than ng costs.
    storage.discharge_nameplate_cost = 1.0e6
    self.assertTrue(lp.solve())

    npt.assert_almost_equal(wind.get_solution_values(),
                            np.zeros(2))

    npt.assert_almost_equal(ng.get_solution_values(),
                            np.array([1.0, 1.0]))


class StorageStepTest(unittest.TestCase):

  def setUp(self):
    self.demand_profile = np.array(
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])

    impulse_profile = np.zeros(len(self.demand_profile))
    impulse_profile[0] = 1.0
    self.impulse = impulse_profile

    rng = pd.date_range('1/1/2011', periods=len(self.demand_profile), freq='H')

    df = pd.DataFrame.from_dict(
        {DEMAND: self.demand_profile,
         SOLAR: impulse_profile,
         TIME: rng}
    )

    self.profiles = df.set_index(TIME,
                                 verify_integrity=True)

    self.lp = LinearProgramContainer(self.profiles)
    self.lp.add_demands(GridDemand(DEMAND))

    self.solar = GridSource(SOLAR, 1e6, 1e6)

  def testRandomSawToothStorage(self):
    lp = self.lp
    solar = self.solar
    lp.add_nondispatchable_sources(solar)
    storage = GridStorage(STORAGE, 0)
    lp.add_storage(storage)

    for storage_efficiency in np.linspace(0.1, 1.0, 4):
      storage.storage_efficiency = storage_efficiency
      self.assertTrue(lp.solve())

      # build up storage profile
      demand_profile = self.demand_profile
      golden_storage_profile = []
      last_storage_value = 0.0
      for d in reversed(demand_profile):
        next_storage_value = (last_storage_value + d) / storage_efficiency
        golden_storage_profile.append(next_storage_value)
        last_storage_value = next_storage_value

      golden_storage_profile.reverse()
      # First generation comes from solar, so set golden_storage[0] = 0.
      golden_storage_profile[0] = 0
      golden_solar_profile = self.impulse * golden_storage_profile[1]

      npt.assert_allclose(solar.get_solution_values(),
                          golden_solar_profile)

      npt.assert_allclose(storage.get_solution_values(),
                          golden_storage_profile,
                          atol=1e-7)


class RecStorageStepTest(unittest.TestCase):

  def setUp(self):
    self.demand_profile = np.array(
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])

    impulse_profile = np.zeros(len(self.demand_profile))
    impulse_profile[0] = 1.0
    self.impulse = impulse_profile

    rng = pd.date_range('1/1/2011', periods=len(self.demand_profile), freq='H')

    df = pd.DataFrame.from_dict(
        {DEMAND: self.demand_profile,
         SOLAR: impulse_profile,
         TIME: rng}
    )

    self.profiles = df.set_index(TIME,
                                 verify_integrity=True)

    self.lp = LinearProgramContainer(self.profiles)
    self.lp.add_demands(GridDemand(DEMAND))

    self.solar = GridSource(SOLAR, 1e6, 1e6)

  def testRandomSawToothRecStorage(self):
    lp = self.lp
    solar = self.solar
    lp.add_nondispatchable_sources(solar)
    storage = GridRecStorage(STORAGE, 0)
    lp.add_storage(storage)

    for storage_efficiency in np.linspace(0.1, 1.0, 4):
      storage.storage_efficiency = storage_efficiency
      self.assertTrue(lp.solve())

      # build up storage profile
      demand_profile = self.demand_profile
      golden_storage_profile = []
      last_storage_value = 0.0
      for d in reversed(demand_profile):
        next_storage_value = (last_storage_value + d) / storage_efficiency
        golden_storage_profile.append(next_storage_value)
        last_storage_value = next_storage_value

      golden_storage_profile.reverse()
      # First generation comes from solar, so set golden_storage[0] = 0.
      golden_storage_profile[0] = 0
      golden_solar_profile = self.impulse * golden_storage_profile[1]

      npt.assert_allclose(solar.get_solution_values(),
                          golden_solar_profile)

      npt.assert_allclose(storage.get_solution_values(),
                          golden_storage_profile,
                          atol=1e-7)


class RpsTest(TwoTimeSliceTest):
  """Test RPS constraints."""

  def testCombinedSolarWindNgRps(self):
    """Sweep RPS."""

    solar = GridSource(SOLAR, 2.0e6, 0, is_rps_source=True)
    wind = GridSource(WIND, 4.0e6, 0, is_rps_source=True)
    ng = GridSource(NG, 0, 1.0e6)

    lp = self.lp
    lp.add_dispatchable_sources(ng)
    lp.add_nondispatchable_sources(solar, wind)

    # As rps is swept, it will fill RPS requirement with cheaper solar
    # first then wind.  NG is cheapest but not in RPS so it will fill
    # in the blanks.

    # Wind is really expensive so check and make sure the LP doesn't
    # cheat by requesting more cheaper solar than it can actually use
    # in order to satisfy rps.  Instead, after it's maxed out on
    # solar, it has to use the expensive wind.

    for rps in np.arange(0, 1.2, 0.1):
      lp.rps_percent = rps * 100.0
      if lp.solve():
        self.assertTrue(rps <= 1.0)

        solar_nameplate = rps * 2 if rps <= 0.5 else 1.0
        wind_nameplate = (rps - 0.5) * 2 if rps > 0.5 else 0.0

        solar_golden_profile = self.profiles[SOLAR] * solar_nameplate
        wind_golden_profile = self.profiles[WIND] * wind_nameplate

        ng_golden_profile = (self.profiles[DEMAND] -
                             solar_golden_profile -
                             wind_golden_profile)

        npt.assert_almost_equal(solar.get_solution_values(),
                                solar_golden_profile)

        npt.assert_almost_equal(wind.get_solution_values(),
                                wind_golden_profile)

        npt.assert_almost_equal(ng.get_solution_values(),
                                ng_golden_profile)

      else:
        # Ensure lp doesn't solve if rps > 1.0
        self.assertTrue(rps > 1.0)

  def testRpsStorage(self):
    """Storage credits rps if timeslice exceeds demand."""

    lp = self.lp
    wind = GridSource(WIND, 2.0e6, 0, is_rps_source=True)
    ng = GridSource(NG, 0, 1.0e6)
    storage = GridRecStorage(STORAGE, 1, 1, 1)

    lp.add_nondispatchable_sources(wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)

    demand_at_0 = self.profiles[DEMAND][0]
    demand_at_1 = self.profiles[DEMAND][1]

    total = sum(self.profiles[DEMAND])
    for rps in range(0, 120, 10):
      lp.rps_percent = rps

      rps_total = total * rps / 100.0

      # Wind is only on for one time-slice.
      wind_at_0 = rps_total
      wind_nameplate = wind_at_0
      golden_wind = wind_nameplate * self.profiles[WIND]

      # Either we charge at 0, or ng fills remaining.
      if wind_at_0 >= demand_at_0:
        storage_at_0 = wind_at_0 - demand_at_0
        ng_at_0 = 0
      else:
        storage_at_0 = 0
        ng_at_0 = demand_at_0 - wind_at_0

      # Discharge everything at 1.
      storage_discharge_power = storage_at_0
      ng_at_1 = demand_at_1 - storage_discharge_power

      if lp.solve():
        self.assertTrue(rps <= 100)
        npt.assert_almost_equal(wind.get_solution_values(),
                                golden_wind)

        npt.assert_almost_equal(ng.get_solution_values(),
                                np.array([ng_at_0, ng_at_1]))

        # Storage at t=1 shows what charged at t=0.
        npt.assert_almost_equal(storage.get_solution_values(),
                                np.array([0.0, storage_at_0]))
      else:
        # Verify no convergence because we asked for a ridiculous rps number.
        self.assertTrue(rps > 100)


class FourTimeSliceRpsTest(unittest.TestCase):

  def setUp(self):
    self.demand_profile = np.array([0.0, 50, 50, 0.0])

    rng = pd.date_range('1/1/2011', periods=len(self.demand_profile), freq='H')

    df = pd.DataFrame.from_dict(
        {DEMAND: self.demand_profile,
         SOLAR: np.array([0.0, 0.0, 0.0, 1.0]),
         WIND: np.array([1.0, 0.0, 0.0, 0.0]),
         TIME: rng}
    )

    self.profiles = df.set_index(TIME,
                                 verify_integrity=True)

    self.lp = LinearProgramContainer(self.profiles)
    self.lp.add_demands(GridDemand(DEMAND))

  def testCircularRecAccounting(self):
    """Verify that RECs get stored and accounted for properly."""
    lp = self.lp
    solar = GridSource(SOLAR, 2e6, 2e6, is_rps_source=True)
    wind = GridSource(WIND, 1e6, 1e6)
    storage = GridRecStorage(STORAGE, 1, 1, 1, discharge_efficiency=0.5)

    lp.add_nondispatchable_sources(solar, wind)
    lp.add_storage(storage)

    for rps in np.arange(0, 110, 10):
      lp.rps_percent = rps
      self.assertTrue(lp.solve())

      npt.assert_almost_equal(solar.get_solution_values(),
                              self.profiles[SOLAR].values *
                              rps / storage.discharge_efficiency)

      npt.assert_almost_equal(wind.get_solution_values(),
                              self.profiles[WIND] *
                              (100 - rps) / storage.discharge_efficiency)

      rps_credit = np.array([v.solution_value()
                             for v in lp.rps_credit_variables[0]])
      self.assertAlmostEqual(sum(rps_credit), rps)

      rec_storage = storage.rec_storage
      no_rec_storage = storage.no_rec_storage

      npt.assert_almost_equal(rps_credit,
                              rec_storage.source.get_solution_values() *
                              storage.discharge_efficiency)

      npt.assert_almost_equal(self.profiles[DEMAND] - rps_credit,
                              no_rec_storage.source.get_solution_values() *
                              storage.discharge_efficiency)


class MockPostProcessingGridSourceSolver(object):
  """Class which modifies solution_values to verify post processing tests.

  Class must be instantiated after LinearProgramContainer.solve().
  Then solution_values may be manipulated to induce errors.

  """

  def __init__(self, grid_source):
    self.original_solution_values = np.array(grid_source.get_solution_values())
    self.reset()
    grid_source.solver = self

  def reset(self):
    self.solution_values = np.array(self.original_solution_values)

  def get_solution_values(self):
    return self.solution_values


class PostProcessingTest(TwoTimeSliceTest):

  def testPostProcessing(self):
    lp = self.lp
    wind = GridSource(WIND, 2.0e6, 0, is_rps_source=True)
    ng = GridSource(NG, 1.0e6, 1.0e6)
    storage = GridStorage(STORAGE, 1, 1, 1)

    lp.add_nondispatchable_sources(wind)
    lp.add_dispatchable_sources(ng)
    lp.add_storage(storage)

    lp.rps_percent = 20
    self.assertTrue(lp.solve())

    wind_solution = MockPostProcessingGridSourceSolver(wind)
    storage_sink_solution = MockPostProcessingGridSourceSolver(storage.sink)

    wind_solution.solution_values -= 0.1

    with self.assertRaises(DemandNotSatisfiedError):
      lp._post_process()

    # After reset it shouldn't error out.
    wind_solution.reset()
    try:
      lp._post_process()
    except RuntimeError:
      self.fail()

    storage_sink_solution.solution_values += 20

    with self.assertRaises(DemandNotSatisfiedError):
      lp._post_process()

    # After reset it shouldn't error out.
    storage_sink_solution.reset()
    try:
      lp._post_process()
    except RuntimeError:
      self.fail()

    lp.rps_demand *= 100
    with self.assertRaises(RpsPercentNotMetError):
      lp._post_process()


class ExtrapolateCostsTest(unittest.TestCase):

  def testZeroCostOfMoney(self):
    self.assertEqual(gslp.extrapolate_cost(1.0,
                                           0.0,
                                           1,
                                           30),
                     1.0)


if __name__ == '__main__':
  unittest.main()
