# coding: utf-8

import functools
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, set_ak_column
from columnflow.production.util import attach_coffea_behavior

# Task 3.
# Produce variables for ZZ, Z1, Z2
# Hint: import the following
# First you need to define build_4sf in util.py
from h4l.util import build_2e2mu, build_4sf

np = maybe_import("numpy")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses=(
        {
            f"{field}.{var}"
            for field in ["Electron", "Muon"]
            for var in ["pt", "mass", "eta", "phi", "charge"]
        } | {
            attach_coffea_behavior,
        }
    ),
    produces={
        "m4l",  "mz1", "mz2"
    },
)
def four_lep_invariant_mass(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Construct four-lepton invariant mass given the Electron and Muon arrays.
    """

    # attach coffea behavior for four-vector arithmetic
    events = self[attach_coffea_behavior](
        events,
        collections=["Electron", "Muon"],
        **kwargs,
    )

    # four-vector sum of first four elements of each
    # lepton collection (possibly fewer)
    dielectron = events.Electron[:, :4].sum(axis=1)
    dimuon = events.Muon[:, :4].sum(axis=1)

    # sum the results to form the four-lepton four-vector
    # TODO # 
    # Task 3.
    # Produce variables for ZZ, Z1, Z2
    # Hint: drop the fourlep and do the correct definition
    # Hint: Build 2e2mu, 4e, 4mu separately and then
    # Hint: zz_inclusive = ak.concatenate([zz_2e2mu, zz_4e, zz_4mu], axis=1)
    # Hint: ak.firsts(zz_inclusive.zz.mass) could be useful
    # fourlep = dielectron + dimuon

    # total number of leptons per event
    n_leptons = (
        ak.num(events.Electron, axis=1) +
        ak.num(events.Muon, axis=1)
    )

    muons_plus = events.Muon[events.Muon.charge > 0]
    muons_minus = events.Muon[events.Muon.charge < 0]
    electrons_plus = events.Electron[events.Electron.charge > 0]
    electrons_minus = events.Electron[events.Electron.charge < 0]

    zz_2e2mu = build_2e2mu(muons_plus, muons_minus, electrons_plus, electrons_minus)
    zz_4e    = build_4sf(electrons_plus, electrons_minus)
    zz_4mu   = build_4sf(muons_plus, muons_minus)
    zz_inclusive = ak.concatenate([zz_2e2mu, zz_4e, zz_4mu], axis=1)


    # four-lepton mass, taking into account only events with at least four leptons,
    # and otherwise substituting a predefined EMPTY_FLOAT value
    fourlep_mass = ak.where(
        n_leptons >= 4,
        ak.firsts(zz_inclusive.zz.mass),
        EMPTY_FLOAT,
    )
    z1_mass = ak.where(
        n_leptons >= 4,
        ak.firsts(zz_inclusive.z1.mass),
        EMPTY_FLOAT,
    )
    z2_mass = ak.where(
        n_leptons >= 4,
        ak.firsts(zz_inclusive.z2.mass),
        EMPTY_FLOAT,
    )

    # write out the resulting mass to the `events` array,
    events = set_ak_column_f32(
        events,
        "m4l",
        fourlep_mass,
    )
    events = set_ak_column_f32(
        events,
        "mz1",
        z1_mass,
    )
    events = set_ak_column_f32(
        events,
        "mz2",
        z2_mass,
    )

    # return the events
    return events
