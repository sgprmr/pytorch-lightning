import os

import pytest

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from tests.base import EvalModelTemplate


def test_mc_called_on_fastdevrun(tmpdir):
    seed_everything(1234)
    os.environ['PL_DEV_DEBUG'] = '1'

    train_val_step_model = EvalModelTemplate()

    # fast dev run = called once
    # train loop only, dict, eval result
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(train_val_step_model)

    # checkpoint should have been called once with fast dev run
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 1

    # -----------------------
    # also called once with no val step
    # -----------------------
    train_step_only_model = EvalModelTemplate()
    train_step_only_model.validation_step = None

    # fast dev run = called once
    # train loop only, dict, eval result
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(train_step_only_model)

    # make sure only training step was called
    assert train_step_only_model.training_step_called
    assert not train_step_only_model.validation_step_called
    assert not train_step_only_model.test_step_called

    # checkpoint should have been called once with fast dev run
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 1


def test_mc_called(tmpdir):
    seed_everything(1234)
    os.environ['PL_DEV_DEBUG'] = '1'

    # -----------------
    # TRAIN LOOP ONLY
    # -----------------
    train_step_only_model = EvalModelTemplate()
    train_step_only_model.validation_step = None

    # no callback
    trainer = Trainer(max_epochs=3, checkpoint_callback=False)
    trainer.fit(train_step_only_model)
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 0

    # -----------------
    # TRAIN + VAL LOOP ONLY
    # -----------------
    val_train_model = EvalModelTemplate()
    # no callback
    trainer = Trainer(max_epochs=3, checkpoint_callback=False)
    trainer.fit(val_train_model)
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 0


@pytest.mark.parametrize("period", [0.2, 0.3, 0.5, 0.8, 1.])
@pytest.mark.parametrize("epochs", [1, 2])
def test_model_checkpoint_period(tmpdir, epochs, period):
    os.environ['PL_DEV_DEBUG'] = '1'

    model = EvalModelTemplate()
    model.validation_step = model.validation_step__decreasing

    checkpoint_callback = ModelCheckpoint(filepath=tmpdir, save_top_k=-1)
    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=False,
        checkpoint_callback=checkpoint_callback,
        max_epochs=epochs,
        val_check_interval=period,
    )
    trainer.fit(model)

    extra_on_train_end = (1 / period) % 1 > 0
    # check that the correct ckpts were created
    expected_calls = epochs * int(1 / period) + int(extra_on_train_end)
    assert len(trainer.dev_debugger.checkpoint_callback_history) == expected_calls