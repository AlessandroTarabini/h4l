# coding: utf-8

"""
Column production methods related to higher-level features.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.production.util import attach_coffea_behavior
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
# TODO #
# Task 1.1
# Add electrons and muons SFs
# Start by uncommenting these lines and
# add electron_weights, muon_weights below to uses and produces:
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
PT_THRESHOLD = 15

from h4l.production.invariant_mass import four_lep_invariant_mass

ak = maybe_import("awkward")
coffea = maybe_import("coffea")
np = maybe_import("numpy")
maybe_import("coffea.nanoevents.methods.nanoaod")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        attach_coffea_behavior,
        deterministic_seeds,
        electron_weights,  muon_weights,
        category_ids, normalization_weights,
        four_lep_invariant_mass,
        "process_id",
    },
    produces={
        attach_coffea_behavior,
        deterministic_seeds,
        electron_weights,  muon_weights,
        category_ids, normalization_weights,
        four_lep_invariant_mass,
        "process_id"
    }
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Build categories
    events = self[category_ids](events, **kwargs)

    # deterministic seeds
    events = self[deterministic_seeds](events, **kwargs)

    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[normalization_weights](events, **kwargs)

        # electron weights
        pt = events.Electron.pt
        high_pt_mask = pt >= PT_THRESHOLD

        events_high_pt = ak.with_field(events, events.Electron[high_pt_mask], "Electron")
        events_high_pt = self[electron_weights](events_high_pt, **kwargs)

        for postfix in ["", "_up", "_down"]:
            colname = f"{electron_weights.weight_name}{postfix}"
            if hasattr(events_high_pt, colname):
                sf_high = getattr(events_high_pt, colname)
                events = set_ak_column(events, colname, sf_high, value_type=np.float32)

        # muon weights
        muon_pt = events.Muon.pt
        high_pt_mask = muon_pt >= PT_THRESHOLD

        events_high_pt_muons = ak.with_field(events, events.Muon[high_pt_mask], "Muon")
        events_high_pt_muons = self[muon_weights](events_high_pt_muons, **kwargs)

        for postfix in ["", "_up", "_down"]:
            colname = f"{muon_weights.weight_name}{postfix}"
            if hasattr(events_high_pt_muons, colname):
                sf_high = getattr(events_high_pt_muons, colname)
                events = set_ak_column(events, colname, sf_high, value_type=np.float32)

    events = self[four_lep_invariant_mass](events, **kwargs)

    return events

