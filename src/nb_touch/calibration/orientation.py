"""Taxel orientation calibration of Pan et al. 2026, Section 3.3: the taxel orientations
`R_i` that best explain the recorded joint torques. See
`docs/theory/taxel-orientation.md`.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from nb_touch.calibration.samples import Jacobian, calibration_samples
from nb_touch.contact.contacts import ContactSettings
from nb_touch.geometry.frames import (
    normals_from_orientations,
    with_estimated_orientations,
)
from nb_touch.geometry.so3 import project_to_so3, project_to_so3_torch
from nb_touch.io.recordings import Recording
from nb_touch.io.sensor_geometry import SensorGeometry

LEARNING_RATE = 0.1
"""Adam learning rate, from the paper (Appendix A)."""

STEPS = 100
"""Adam steps, from the paper (Appendix A)."""


@dataclass(frozen=True)
class CalibrationSettings:
    """Settings of the taxel orientation calibration.

    Attributes:
        contact: Active taxel threshold, fewest active taxels per sample, and CoP
            model σ and λ.
        learning_rate: Adam learning rate.
        steps: Adam steps.
    """

    contact: ContactSettings = field(default_factory=ContactSettings)
    learning_rate: float = LEARNING_RATE
    steps: int = STEPS


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of the calibration.

    Attributes:
        geometry: The sensor geometry with the calibrated orientations.
        losses: `(steps + 1,)` torque MSE, in (N·m)^2, before each step and at the
            end.
        num_samples: Number of samples with enough active taxels, used in the fit.
    """

    geometry: SensorGeometry
    losses: np.ndarray
    num_samples: int


def predict_torques(orientations: torch.Tensor, samples: dict) -> torch.Tensor:
    """Return the predicted joint torques `τ = -J^T f_cop` of every sample, `(S, J)`.

    The paper's static equilibrium, gravity neglected: `f_cop` is the force acting
    on the sensor, and `τ` the torques the joints apply to hold the finger still.

    Args:
        orientations: Taxel orientations `(T, 3, 3)`, taxel frame to sensor frame.
        samples: `CalibrationSamples` fields as tensors, from `samples_to_tensors`.
    """
    rotated = torch.einsum(
        "eij,ej->ei", orientations[samples["taxel_index"]], samples["forces"]
    )
    parts = torch.einsum("eij,ej->ei", samples["operators"], rotated)
    num_samples = len(samples["torques"])
    cop_forces = parts.new_zeros((num_samples, 3))
    cop_forces = cop_forces.index_add(0, samples["sample_index"], parts)
    return -torch.einsum("sij,si->sj", samples["jacobians"], cop_forces)


def torque_loss(orientations: torch.Tensor, samples: dict) -> torch.Tensor:
    """Return the mean squared error between predicted and recorded joint torques."""
    return torch.mean(
        (predict_torques(orientations, samples) - samples["torques"]) ** 2
    )


def samples_to_tensors(samples) -> dict:
    """Return the fields of `CalibrationSamples` as float64 or index tensors."""
    return {
        name: (
            torch.as_tensor(value)
            if name.endswith("_index")
            else torch.as_tensor(value, dtype=torch.float64)
        )
        for name, value in vars(samples).items()
    }


def fit_orientations(
    prior, samples: dict, settings: CalibrationSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Fit free 3×3 matrices, projected to rotations, with Adam.

    Args:
        prior: Initial orientations `(T, 3, 3)`.
        samples: From `samples_to_tensors`.
        settings: `learning_rate` and `steps` are used.

    Returns:
        `(orientations, losses)`: orientations `(T, 3, 3)` and the loss before each step
        and at the end, `(steps + 1,)`.
    """
    matrices = torch.tensor(prior, dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.Adam([matrices], lr=settings.learning_rate)
    losses = []
    for _ in range(settings.steps):
        optimiser.zero_grad()
        loss = torque_loss(project_to_so3_torch(matrices), samples)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    orientations = project_to_so3(matrices.detach().numpy())
    with torch.no_grad():
        losses.append(torque_loss(torch.as_tensor(orientations), samples).item())
    return orientations, np.array(losses)


def calibrate_orientations(
    recording: Recording,
    geometry: SensorGeometry,
    jacobian: Jacobian,
    link: str | None = None,
    settings: CalibrationSettings = CalibrationSettings(),
) -> CalibrationResult:
    """Estimate the taxel orientations from a recording of random fingertip contacts.

    The prior orientations (from the geometry, or estimated from the positions if
    missing) are the starting point and give the normals of the CoP model. Taxels
    never active keep their prior orientation.

    Args:
        recording: Taxel forces in their taxel frames, joint angles and torques.
        geometry: Sensor geometry, with or without orientations.
        jacobian: `jacobian(q, link, point) -> (3, J)`, see `Jacobian`.
        link: Link the sensor is mounted on, passed to `jacobian`.
        settings: Calibration settings.

    Raises:
        ValueError: If the recording and geometry disagree on the taxel count, or
            no sample has enough active taxels.
    """
    if recording.num_taxels != geometry.num_taxels:
        raise ValueError(
            f"recording has {recording.num_taxels} taxels, "
            f"geometry {geometry.num_taxels}"
        )
    prior = with_estimated_orientations(geometry)
    samples = calibration_samples(
        recording,
        prior.positions,
        normals_from_orientations(prior.orientations),
        jacobian,
        link,
        settings.contact,
    )
    orientations, losses = fit_orientations(
        prior.orientations, samples_to_tensors(samples), settings
    )
    return CalibrationResult(
        geometry=SensorGeometry(geometry.name, geometry.positions, orientations),
        losses=losses,
        num_samples=samples.num_samples,
    )
