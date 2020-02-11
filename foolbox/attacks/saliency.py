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
        stepsize: float = 0.1,
        fast: bool = True,
        max_perturbations_per_pixel: int = 7,
    ):
        self.steps = steps
        self.fast = fast
        self.stepsize = stepsize
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
        mask = ep.where(
            ep.logical_or(x > min_, x < max_), ep.ones_like(x), ep.zeros_like(x)
        )

        # count tracks how often each pixel was changed
        counts = ep.zeros_like(x)

        # TODO: stop if mask is all zero
        for step in range(self.steps):
            is_adv = is_adversarial(x)
            if is_adv.all() or (mask == 0).all():
                break  # TODO: modify this if we allow multiple (random) targets

            # get pixel location with highest influence on class
            p1_idx, p2_idx, p_sign = self.saliency_map(
                model, x, classes, mask, fast=self.fast
            )

            # apply perturbation
            # x = x.index_update(idx, x[idx] - p_sign * self.stepsize * (max_ - min_))
            x = x.index_update(
                p1_idx, x[p1_idx] - p_sign * self.stepsize * (max_ - min_)
            )
            x = x.index_update(
                p2_idx, x[p2_idx] - p_sign * self.stepsize * (max_ - min_)
            )

            # tracks number of updates for each pixel
            counts = counts.index_update(p1_idx, counts[p1_idx] + 1)
            counts = counts.index_update(p2_idx, counts[p2_idx] + 1)

            # remove pixel from search domain if it hits the bound
            mask = mask.index_update(
                p1_idx,
                ep.where(
                    ep.logical_or(x[p1_idx] <= min_, x[p1_idx] >= max_), 0, mask[p1_idx]
                ),
            )
            mask = mask.index_update(
                p2_idx,
                ep.where(
                    ep.logical_or(x[p2_idx] <= min_, x[p2_idx] >= max_), 0, mask[p2_idx]
                ),
            )

            # remove pixel if it was changed too often
            mask = mask.index_update(
                p1_idx,
                ep.where(
                    counts[p1_idx] >= self.max_perturbations_per_pixel, 0, mask[p1_idx]
                ),
            )
            mask = mask.index_update(
                p2_idx,
                ep.where(
                    counts[p2_idx] >= self.max_perturbations_per_pixel, 0, mask[p2_idx]
                ),
            )

            print(mask[0].sum())

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

        N = len(x)
        dim_x = flatten(x).shape[1]

        _, num_classes, grads_targets = loss_num_classes_and_grad(x, targets)
        # alphas *= mask
        grads_targets = flatten(grads_targets)
        alphas = grads_targets.reshape((N, 1, -1)) + grads_targets.reshape((N, -1, 1))

        if fast:
            betas = -ep.ones_like(alphas)
        else:
            grads_others = ep.zeros_like(grads_targets)
            labels = targets
            for _ in range(num_classes - 1):
                labels = (labels + 1) % num_classes
                _, _, grad_others = loss_num_classes_and_grad(x, labels)
                grads_others += grad_others
                # betas += beta * (mask - atleast_kd(alphas, x.ndim))

            betas = grads_others.reshape((N, 1, -1)) + grads_others.reshape((N, -1, 1))

        # compute saliency map
        # (take into account both pos. & neg. perturbations)
        # salmap = ep.abs(alphas) * ep.abs(betas) * ep.sign(alphas * betas)

        scores_mask = ep.logical_or(alphas < 0, betas < 0)
        salmap = (
            scores_mask.float() * atleast_kd(mask, scores_mask.ndim) * (-alphas * betas)
        )

        salmap = salmap.view(N, dim_x * dim_x)
        max_idx = ep.argmax(salmap, -1)
        p1 = max_idx % dim_x
        p2 = max_idx // dim_x

        p1 += ep.arange(x, 0, N) * dim_x
        p2 += ep.arange(x, 0, N) * dim_x

        p1 = SaliencyAttack.unravel_index(p1, x.shape)
        p2 = SaliencyAttack.unravel_index(p2, x.shape)

        p_sign = ep.sign(alphas)[max_idx]

        # find optimal pixel & direction of perturbation
        # idx = flatten(salmap).argmin(-1) + ep.arange(x, 0, len(x)) * flatten(x).shape[1]
        # idx = SaliencyAttack.unravel_index(idx, mask.shape)

        # pix_sign = ep.sign(alphas)[idx]

        return p1, p2, p_sign


"""

def _sum_pair(self, grads, dim_x):
    return grads.view(-1, dim_x, 1) + grads.view(-1, 1, dim_x)

def _and_pair(self, cond, dim_x):
    return cond.view(-1, dim_x, 1) & cond.view(-1, 1, dim_x)

# alpha in Algorithm 3 line 2
gradsum_target = self._sum_pair(grads_target, dim_x)
# alpha in Algorithm 3 line 3
gradsum_other = self._sum_pair(grads_other, dim_x)

if self.theta > 0:
    scores_mask = (
        torch.gt(gradsum_target, 0) & torch.lt(gradsum_other, 0))
else:
    scores_mask = (
        torch.lt(gradsum_target, 0) & torch.gt(gradsum_other, 0))

scores_mask &= self._and_pair(search_space.ne(0), dim_x)
scores_mask[:, range(dim_x), range(dim_x)] = 0

if self.comply_cleverhans:
    valid = torch.ones(scores_mask.shape[0]).byte()
else:
    valid = scores_mask.view(-1, dim_x * dim_x).any(dim=1)

scores = scores_mask.float() * (-gradsum_target * gradsum_other)
best = torch.max(scores.view(-1, dim_x * dim_x), 1)[1]
p1 = torch.remainder(best, dim_x)
p2 = (best / dim_x).long()
return p1, p2, valid


"""
