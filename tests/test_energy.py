from __future__ import annotations

import math

from dockwise.config import StationConfig
from dockwise.energy import StationPlant


def test_station_energy_conservation_surplus_and_no_simultaneous_flow() -> None:
    plant = StationPlant(StationConfig())
    before = plant.energy_wh
    flows = plant.step(60.0, pv_w=30.0, requested_robot_charge_w=10.0)
    charger_bus = flows.charger_output_wh + flows.charger_loss_wh
    assert math.isclose(
        flows.pv_wh,
        flows.idle_wh + charger_bus + flows.station_charge_input_wh + flows.curtailed_wh,
        abs_tol=1e-10,
    )
    assert math.isclose(plant.energy_wh - before, flows.station_charge_stored_wh, abs_tol=1e-10)
    assert flows.station_withdrawn_wh == 0.0


def test_station_energy_conservation_deficit() -> None:
    plant = StationPlant(StationConfig())
    before = plant.energy_wh
    flows = plant.step(60.0, pv_w=0.0, requested_robot_charge_w=10.0)
    charger_bus = flows.charger_output_wh + flows.charger_loss_wh
    assert math.isclose(
        flows.pv_wh + flows.station_discharge_bus_wh,
        flows.idle_wh + charger_bus,
        abs_tol=1e-10,
    )
    assert math.isclose(before - plant.energy_wh, flows.station_withdrawn_wh, abs_tol=1e-10)
    assert math.isclose(
        flows.station_withdrawn_wh,
        flows.station_discharge_bus_wh + flows.storage_loss_wh,
        abs_tol=1e-10,
    )
    assert flows.station_charge_stored_wh == 0.0

