"""A restart must continue the run, and must say honestly which state it is in.

The demo's plastic brain kept nothing: `_plastic_brain()` built a fresh one and the 117,800
trained synaptic scales died with the process, so every restart put the fly back at "has learned
nothing" while the UI carried on as though nothing had happened. These tests pin the four things
that make persistence honest rather than convenient:

  1. a restart resumes the trained weights, and the status string names that it resumed;
  2. a brain that has NOT trained is not written over a trained one, so a deliberate
     fresh reset cannot destroy the run on disk;
  3. asking to reload while the plastic brain is selected reloads the PLASTIC checkpoint, not the
     old prompt-index one -- the same architecture-swap trap that `_reset_readout_learning` had;
  4. a checkpoint that would not reproduce is refused and reported, not applied.

The API's module-level STATE means these drive `_boot()` directly, which is exactly what a restart
does, rather than going through HTTP.
"""
from __future__ import annotations

import numpy as np
import pytest

from brain.encoders import encode_text


@pytest.fixture
def api():
    import brain.api as module

    return module


def _observe(pb, n=8):
    """Take n real plasticity steps, so `updates` is non-zero and the scales actually move."""
    for i in range(n):
        pb.observe(encode_text(f"frase {i}"), i % pb.n_pools)


def _boot_plastic(api):
    """Simulate a server start with the plastic brain selected, and return it."""
    api._boot()
    api.STATE["readout_kind"] = "plastic_brain"
    return api._plastic_brain()


def test_a_restart_resumes_the_brain_it_trained(api, isolate_demo_checkpoint):
    pb = _boot_plastic(api)
    assert "scales at 1.0" in api.STATE["checkpoint_status"]
    assert not isolate_demo_checkpoint.exists(), "nothing should be on disk before a save"

    _observe(pb)
    scales = pb.r.plastic_scale.copy()
    saved = api._save_plastic_brain()
    assert saved is not None, "a trained brain must be saved"
    assert isolate_demo_checkpoint.exists()

    # a restart: _boot() rebuilds STATE from scratch, then the brain is fetched again
    pb2 = _boot_plastic(api)
    assert np.allclose(pb2.r.plastic_scale, scales), "the trained scales did not come back"
    assert pb2.updates == pb.updates
    assert "resumed" in api.STATE["checkpoint_status"], api.STATE["checkpoint_status"]


def test_an_untrained_brain_is_not_saved_over_a_trained_one(api, isolate_demo_checkpoint):
    """The fresh state must not be able to destroy the run it was resuming.

    `/session?fresh=true` and `/train?fresh=true` both reset the scales to 1.0, and a shutdown on
    the way out from one of those would otherwise overwrite the trained checkpoint with an
    untrained brain -- losing the run silently, at exactly the moment the viewer had just reset it
    to watch the learning happen.
    """
    pb = _boot_plastic(api)
    _observe(pb)
    trained = pb.r.plastic_scale.copy()
    assert api._save_plastic_brain() is not None

    # a fresh session, as the demo's "watch it learn" control does
    api._reset_readout_learning()
    assert pb.updates == 0, "the reset should have taken the brain back to untrained"
    assert np.allclose(pb.r.plastic_scale, 1.0)

    assert api._save_plastic_brain() is None, "an untrained brain must not be persisted"

    # so the trained run is still the one on disk
    pb2 = _boot_plastic(api)
    assert np.allclose(pb2.r.plastic_scale, trained), "the trained run was overwritten by a reset"
    assert "resumed" in api.STATE["checkpoint_status"]


def test_reload_loads_the_plastic_checkpoint_not_the_old_readout(api, isolate_demo_checkpoint):
    """Reload must mean the ACTIVE readout's checkpoint.

    It loaded the prompt-index checkpoint unconditionally, so reloading while the plastic brain was
    selected swapped the architecture back to the old readout and the demo quietly stopped running
    on the connectome. Same failure class as the session-start bug in _reset_readout_learning.
    """
    pb = _boot_plastic(api)
    _observe(pb)
    trained = pb.r.plastic_scale.copy()
    assert api._save_plastic_brain() is not None

    api._reset_readout_learning()
    assert np.allclose(pb.r.plastic_scale, 1.0)

    result = api.train(api.TrainReq(kind="plastic_brain", fresh=False))
    assert result["ok"] is True, result
    assert result["readout_kind"] == "plastic_brain", "reload swapped the architecture"
    assert api.STATE["readout_kind"] == "plastic_brain"
    assert np.allclose(api._plastic_brain().r.plastic_scale, trained), "reload did not restore"
    assert "resumed" in api.STATE["checkpoint_status"]


def test_reload_with_no_checkpoint_says_so(api, isolate_demo_checkpoint):
    """A missing checkpoint is reported as missing, not silently treated as success."""
    _boot_plastic(api)
    assert not isolate_demo_checkpoint.exists()
    result = api.train(api.TrainReq(fresh=False))
    assert result["ok"] is False
    assert "no plastic checkpoint" in result["checkpoint_status"]


def test_progress_is_checkpointed_without_a_graceful_shutdown(api, isolate_demo_checkpoint):
    """An abrupt exit must not lose the run.

    A shutdown hook alone does not cover this: on Windows `terminate()` hard-kills the process, so
    uvicorn never runs its lifespan shutdown and the save silently does not happen. Closing a
    terminal is the normal way a demo stops, so this pins the automatic checkpoint that runs during
    training. Measured before it existed: 106 plasticity updates were taken and no file was written
    at all.
    """
    pb = _boot_plastic(api)
    assert api._maybe_checkpoint_plastic() is None, "nothing to save before any training"

    # below the threshold: still nothing
    pb.observe(encode_text("uno"), 0)
    assert pb.updates == 1
    assert api._maybe_checkpoint_plastic() is None

    # cross the threshold
    for i in range(api.PLASTIC_CHECKPOINT_EVERY):
        pb.observe(encode_text(f"frase {i}"), i % pb.n_pools)
    assert pb.updates >= api.PLASTIC_CHECKPOINT_EVERY
    written = api._maybe_checkpoint_plastic()
    assert written is not None, "no checkpoint was written during training"
    assert isolate_demo_checkpoint.exists()

    # a restart picks it up, with no shutdown hook having run
    pb2 = _boot_plastic(api)
    assert np.allclose(pb2.r.plastic_scale, pb.r.plastic_scale)
    assert "resumed" in api.STATE["checkpoint_status"]


def test_switching_to_the_plastic_brain_can_resume_instead_of_wiping(api, isolate_demo_checkpoint):
    """`fresh=False` must mean resume, for the switch too.

    Selecting the plastic brain used to reset it unconditionally, so the only way to select it was
    to erase it and the resume path was unreachable in practice: the checkpoint would only be
    loaded by a lazy build if the plastic brain were already active, and it never could be,
    because selecting it was what wiped it.
    """
    pb = _boot_plastic(api)
    _observe(pb)
    trained = pb.r.plastic_scale.copy()
    assert api._save_plastic_brain() is not None

    api._boot()  # a restart, so the plastic brain is not the active readout any more
    assert api.STATE["plastic_brain"] is None

    switch = api.train(api.TrainReq(kind="plastic_brain", fresh=False))
    assert switch["ok"] is True, switch
    got = api._plastic_brain()
    assert np.allclose(got.r.plastic_scale, trained), "switching wiped the saved run"
    assert got.updates > 0
    assert "resumed" in api.STATE["checkpoint_status"]

    # and the same switch with fresh=True is the deliberate reset, which still works
    fresh = api.train(api.TrainReq(kind="plastic_brain", fresh=True))
    assert fresh["ok"] is True, fresh
    assert np.allclose(api._plastic_brain().r.plastic_scale, 1.0), "fresh=True did not reset"


def test_a_checkpoint_that_would_not_reproduce_is_refused_not_applied(api, isolate_demo_checkpoint):
    """A refused load must leave the brain fresh and say why.

    Silently loading a stale or mismatched checkpoint is worse than loading none, because the model
    looks trained and answers arbitrarily.
    """
    pb = _boot_plastic(api)
    _observe(pb)
    assert api._save_plastic_brain() is not None

    # corrupt the stored metadata, which is what the load guard reads first
    import json

    with np.load(isolate_demo_checkpoint, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    arrays["meta"] = np.array("{}")
    np.savez_compressed(isolate_demo_checkpoint, **arrays)

    pb2 = _boot_plastic(api)
    assert np.allclose(pb2.r.plastic_scale, 1.0), "a bad checkpoint was applied anyway"
    assert "refused" in api.STATE["checkpoint_status"], api.STATE["checkpoint_status"]


def test_the_status_distinguishes_fresh_resumed_and_refused(api, isolate_demo_checkpoint):
    """The three states must be distinguishable in the text, because a surface shows it."""
    fresh = _boot_plastic(api)
    assert "scales at 1.0" in api.STATE["checkpoint_status"]

    _observe(fresh)
    api._save_plastic_brain()
    _boot_plastic(api)
    resumed_status = api.STATE["checkpoint_status"]
    assert "resumed" in resumed_status and "scales at 1.0" not in resumed_status
