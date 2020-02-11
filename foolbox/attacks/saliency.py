from typing import Union, Tuple  # Optional
import eagerpy as ep

from ..devutils import flatten
from ..devutils import atleast_kd

from ..models import Model

from ..criteria import TargetedMisclassification

from .base import MinimizationAttack
from .base import T
from .base import get_criterion
from .base import get_is_adversarial


class SaliencyAttack(MinimizationAttack):
    """Implements the Saliency Map Attack.
    The attack was introduced in [1]_.
    References
    ----------
    .. [1] Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson,
           Z. Berkay Celik, Ananthram Swami, "The Limitations of Deep Learning
           in Adversarial Settings", https://arxiv.org/abs/1511.07528
    """

    def __init__(
        self,
        steps: int = 50,
        fast: bool = True,
        theta: float = 0.1,
        max_perturbations_per_pixel: int = 7,
    ):
        self.steps = steps
        self.fast = fast
        self.theta = theta
        self.max_perturbations_per_pixel = max_perturbations_per_pixel

    def __call__(
        self, model: Model, inputs: T, criterion: Union[TargetedMisclassification, T],
    ) -> T:
        x, restore_type = ep.astensor_(inputs)
        del inputs

        criterion_ = get_criterion(criterion)

        is_adversarial = get_is_adversarial(criterion_, model)

        if isinstance(criterion_, TargetedMisclassification):
            classes = criterion_.target_classes
        else:
            raise ValueError("unsupported criterion")

        min_, max_ = model.bounds

        # the mask defines the search domain
        # each modified pixel with border value is set to zero in mask
        mask = ep.ones_like(x)

        # count tracks how often each pixel was changed
        counts = ep.zeros_like(x)

        # TODO: stop if mask is all zero
        for step in range(self.steps):
            is_adv = is_adversarial(x)
            if is_adv.all():
                break  # TODO: check

            # get pixel location with highest influence on class
            idx, p_sign = self.saliency_map(model, x, classes, mask, fast=self.fast)

            # apply perturbation
            x[idx] += -p_sign * self.theta * (max_ - min_)

            # tracks number of updates for each pixel
            counts[idx] += 1

            # remove pixel from search domain if it hits the bound
            if x[idx] <= min_ or x[idx] >= max_:
                mask[idx] = 0

            # remove pixel if it was changed too often
            if counts[idx] >= self.max_perturbations_per_pixel:
                mask[idx] = 0

            x = ep.clip(x, min_, max_)

        return restore_type(x)

    @staticmethod
    def unravel_index(self: ep.Tensor, shape: tuple):
        indices = [[] for _ in range(len(shape))]
        for index in self:
            for i, dim in enumerate(reversed(shape)):
                j = len(shape) - i - 1
                indices[j].append(index.item() % dim)
                index = index // dim

        return tuple(indices)

    def saliency_map(
        self,
        model: Model,
        x: ep.Tensor,
        targets: ep.Tensor,
        mask: ep.Tensor,
        fast: bool = False,
    ):
        def loss_fn(
            inputs: ep.Tensor, labels: ep.Tensor
        ) -> Tuple[ep.Tensor, ep.Tensor]:
            logits = model(inputs)
            loss = ep.softmax(logits, -1)[range(len(x)), labels].sum()

            return loss, logits.shape[-1]

        loss_num_classes_and_grad = ep.value_and_grad_fn(x, loss_fn, has_aux=True)

        _, num_classes, alphas = loss_num_classes_and_grad(x, targets)
        alphas *= mask

        if fast:
            betas = -ep.ones_like(alphas)
        else:
            betas = ep.zeros_like(alphas)
            labels = targets
            for _ in range(num_classes - 1):
                labels = (labels + 1) % num_classes
                _, _, beta = loss_num_classes_and_grad(x, labels)
                betas += beta * (mask - atleast_kd(alphas, x.ndim))

        # compute saliency map
        # (take into account both pos. & neg. perturbations)
        salmap = ep.abs(alphas) * ep.abs(betas) * ep.sign(alphas * betas)

        # find optimal pixel & direction of perturbation
        idx = flatten(salmap).argmin(-1) + ep.arange(x, 0, len(x)) * flatten(x).shape[1]
        idx = SaliencyAttack.unravel_index(idx, mask.shape)
        import numpy as np

        idx = np.array(idx)
        print(len(idx), alphas.shape, idx.shape)
        pix_sign = ep.sign(alphas)[idx]
        print(pix_sign.shape)

        return idx, pix_sign
