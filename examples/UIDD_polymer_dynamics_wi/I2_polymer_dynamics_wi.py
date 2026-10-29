import os
os.environ["JAX_PLATFORM_NAME"] = "cuda"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.96"
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import sys
sys.path.append('..')
sys.path.append('../..')
import equinox as eqx
from datasets import load_from_disk
from net.dynamics import OnsagerNet, UIDD2
from examples.utils.data import get_path

from net.models import (
    PotentialResMLP,
    DissipationMatrixMLP,
    ConservationMatrixMLP,
    DiffusionMLP,
    DiffusionCholeskyMLP,
    HamiltonianMLP,
    PotentialResMLP_scale,
    PotentialResMLP_Vscale,
)

from net._augmentations import (
    RandomChoiceAugmentation,
    ReducedHeadTailFlip,
    ReducedReflectionX
)

from net.trainers import MLETrainer

import hydra
import logging

# ------------------------- Typing imports ------------------------- #
from omegaconf import DictConfig
from net.dynamics import SDE

from typing import Any
import jax.tree_util as tree
def get_filter_spec(model) -> Any:
    filter_spec = tree.tree_map(lambda _: True, model)

    # 然后手动将 'name' 设为 False，表示 static
    filter_spec = eqx.tree_at(
        lambda m: m.J,  # 这里的路径是 model.name
        filter_spec,
        replace=False      # False -> static
    )
    return filter_spec

def get_flips(z):
    z_flip_2 = z * jnp.array([1.0, -1.0, 1.0])
    z_flip_3 = z * jnp.array([1.0, 1.0, -1.0])
    z_flip_2_3 = z * jnp.array([1.0, -1.0, -1.0])
    return z, z_flip_2, z_flip_3, z_flip_2_3


# def call_symmetric(base, x, args):
#     transformed_xs = jnp.array(get_flips(x))
#     outputs = jax.vmap(base, in_axes=(0, None))(transformed_xs, args)
#     return jnp.mean(outputs, axis=0)


def call_symmetric(base, x, args):
    # stack the four flipped inputs -> shape (4, dim)
    transformed_xs = jnp.stack(get_flips(x))

    # vectorise base over the first axis of transformed_xs; args is shared
    outputs = jax.vmap(lambda xi: base(xi, args))(transformed_xs)

    # outputs may be an ndarray (4, ...) or a pytree with leading axis 4.
    # average across the leading axis in a pytree-safe way.
    averaged = jax.tree_util.tree_map(lambda v: jnp.mean(v, axis=0), outputs)

    return averaged


class SymmetricPotential(eqx.Module):
    base: PotentialResMLP

    def __call__(self, x, args):
        return call_symmetric(self.base, x, args)

def build_model(config: DictConfig) -> SDE:
    """
    Builds the model for polymer dynamics using the OnsagerNet framework.

    Args:
        config (DictConfig): Configuration object containing model parameters.
        dataset (Dataset): Dataset object used to guide model configuration.

    Returns:
        SDE: The constructed OnsagerNet model for polymer dynamics.
    """
    # Initialize random keys for model initialization
    init_keys = jax.random.PRNGKey(config.model.seed)
    v_key, m_key, w_key, d_key = jax.random.split(init_keys, 4)

    # Initialize each model component
    if config.model.potential.get("Vscale", False):
        potential = PotentialResMLP_Vscale(
            key=v_key,
            dim=config.dim,
            units=config.model.potential.units,
            activation=config.model.potential.activation,
            n_pot=config.model.potential.n_pot,
            scale=config.model.potential.scale,
            alpha=config.model.potential.alpha,
            param_idx=config.model.potential.param_idx,
        )
    else:
        potential = PotentialResMLP_scale(
            key=v_key,
            dim=config.dim,
            units=config.model.potential.units,
            activation=config.model.potential.activation,
            n_pot=config.model.potential.n_pot,
            scale=config.model.potential.scale,
            alpha=config.model.potential.alpha,
            param_idx=config.model.potential.param_idx,
        )

    dissipation = DissipationMatrixMLP(
        key=m_key,
        dim=config.dim,
        units=config.model.dissipation.units,
        activation=config.model.dissipation.activation,
        alpha=config.model.dissipation.alpha,
        param_idx=config.model.dissipation.param_idx,
        is_bounded=config.model.dissipation.is_bounded,
    )
    # conservation = ConservationMatrixMLP(
    #     key=w_key,
    #     dim=config.dim,
    #     activation=config.model.conservation.activation,
    #     units=config.model.conservation.units,
    #     param_idx=config.model.conservation.param_idx,
    #     is_bounded=config.model.conservation.is_bounded,
    # ) 
    diffusion = DiffusionCholeskyMLP( 
        key = d_key,
        dim=config.dim,
        units= config.model.diffusion.units,
        activation= config.model.diffusion.activation,
        alpha= config.model.diffusion.alpha,
        param_idx=config.model.diffusion.param_idx,
    )

    Hamiltonian = HamiltonianMLP(
        key=w_key,
        dim=config.dim,
        activation=config.model.hamiltonian.activation,
        units=config.model.hamiltonian.units,
        param_idx=config.model.hamiltonian.param_idx,
        is_bounded=config.model.hamiltonian.is_bounded,
    )
    # Apply symmetry if specified in the config
    if config.model.get("symmetric", False):
        potential = SymmetricPotential(base=potential)

    # Construct the OnsagerNet model using the individual components
    sde = UIDD2(
        D=config.dim,
        potential=potential,
        Hamiltonian=Hamiltonian, 
        diffusion=diffusion,
    )

    return sde


@hydra.main(
    config_path="./config",
    config_name="I2_polymer_dynamics_wi",
    version_base=None,
)
def train_model(config: DictConfig) -> None:
    """
    Main training script for polymer dynamics with closure modelling. This function
    handles data loading, model building, and model training.

    Args:
        config (DictConfig): Configuration object containing training parameters and paths.
    """
    # Get the runtime directory from Hydra's configuration
    runtime_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Use Hydra's default logger
    logger = logging.getLogger(__name__)
    train_path = get_path(config.data.cache_path, f"{config.data.filename}_train")
    dataset = load_from_disk(train_path).with_format("jax")
    # dataset = load_and_process_data(config)
    logger.info(f"Loaded training dataset from {train_path}")
    logger.info("="*60)

    # Build the model using the configuration and dataset
    logger.info("Building model...")
    model = build_model(config)

    # Load model if specified in the configuration
    if config.model.get("load_model", None):
        model_path = config.model.load_model
        logger.info(f"Loading model from {model_path}...")
        model = eqx.tree_deserialise_leaves(model_path, model)
    logger.info("="*60)

    # Initialize the MLE trainer with configuration options
    logger.info("Setting up data augmentation...")
    ht_aug = ReducedHeadTailFlip()
    ref_aug = ReducedReflectionX()
    aug = RandomChoiceAugmentation([ht_aug, ref_aug])

    logger.info("Configured augmentations:")
    logger.info("ReducedHeadTailFlip: Flips z2 - [1, -1, 1]")
    logger.info("ReducedReflectionX: Flips z2,z3 - [1, -1, -1]")
    logger.info("RandomChoiceAugmentation: Randomly selects one per batch")
    logger.info(f"Augmentation probability: {config.train.aug_prob}")
    logger.info("="*60)

    filter_spec = get_filter_spec(model)
    trainer = MLETrainer(
        opt_options=config.train.opt,
        rop_options=config.train.rop,
        data_augmentation=aug,
        augmentation_prob=config.train.aug_prob,
    )

    # Start training the model using the trainer
    logger.info(f"Training OnsagerNet for {config.train.num_epochs} epochs...")
    logger.info("="*60)

    # Create a random key for augmentation
    training_key = jax.random.PRNGKey(config.train.get("seed", 123))

    trained_model, _, _ = trainer.train(
        model=model,
        dataset=dataset,
        num_epochs=config.train.num_epochs,
        batch_size=config.train.batch_size,
        logger=logger,
        filter_spec=filter_spec,
        checkpoint_dir=runtime_dir,  # Directory to save checkpoints
        checkpoint_every=config.train.checkpoint_every,  # Frequency to save checkpoints
        rng_key=training_key,
    )

    # Log the completion of training and save the trained model
    logger.info(f"Saving output to {runtime_dir}")
    eqx.tree_serialise_leaves(os.path.join(runtime_dir, "model.eqx"), trained_model)


if __name__ == "__main__":
    train_model()
