from dataclasses import dataclass
import torch


@dataclass
class OTResult:
    plan: torch.Tensor
    diagnostics: dict


def log_sinkhorn(cost, epsilon, tolerance=1e-6, max_iterations=10000):
    cost = cost.double()
    if cost.ndim != 2 or min(cost.shape) < 2 or epsilon <= 0 or not torch.isfinite(cost).all():
        raise ValueError('OT 需要有限的至少2×2代价与正epsilon')
    n, m = cost.shape
    a = cost.new_full((n,), 1/n)
    b = cost.new_full((m,), 1/m)
    log_k = -(cost-cost.min())/epsilon
    u, v = torch.zeros_like(a), torch.zeros_like(b)
    converged = False
    residual = float('inf')
    for step in range(max_iterations):
        old_u = u.clone()
        u = a.log()-torch.logsumexp(log_k+v[None, :], dim=1)
        v = b.log()-torch.logsumexp(log_k+u[:, None], dim=0)
        if step % 10 == 0 or step == max_iterations-1:
            plan = (log_k+u[:, None]+v[None, :]).exp()
            residual = float(torch.maximum((plan.sum(1)-a).abs().max(), (plan.sum(0)-b).abs().max()))
            if residual <= tolerance:
                converged = True
                break
    diag = dict(converged=converged, iterations=step+1, marginal_error=residual,
                iteration_residual=float((u-old_u).abs().max()), total_mass=float(plan.sum()),
                entropy=float(-(plan*plan.clamp_min(1e-300).log()).sum()))
    if not converged:
        raise RuntimeError(f'Sinkhorn 未收敛，不生成替代配对: {diag}')
    return OTResult(plan.detach(), diag)


@torch.no_grad()
def solve_conditional_ot(sources, targets, source_condition_ids, target_condition_ids, reference_variance,
                         reduction='sum', tolerance=1e-6, max_iterations=10000):
    if len(source_condition_ids) != len(sources) or len(target_condition_ids) != len(targets):
        raise ValueError('condition_id 数量不符')
    if len(set(source_condition_ids+target_condition_ids)) != 1:
        raise ValueError('禁止跨 condition_id 的 OT 配对')
    x, y = sources.flatten(1).double(), targets.flatten(1).double()
    if x.shape[1] != y.shape[1]:
        raise ValueError('端点必须处于相同图像空间')
    # Avoid allocating [n,m,d]. Full canvas Euclidean coordinates, float64.
    cost = .5*(x.square().sum(1)[:, None]+y.square().sum(1)[None, :]-2*x@y.T).clamp_min(0)
    epsilon = reference_variance
    if reduction == 'mean':
        cost, epsilon = cost/x.shape[1], epsilon/x.shape[1]
    elif reduction != 'sum':
        raise ValueError('cost reduction 必须为sum或mean')
    result = log_sinkhorn(cost, epsilon, tolerance, max_iterations)
    result.diagnostics.update(condition_id=source_condition_ids[0], cost_reduction=reduction,
                              epsilon=epsilon, coordinates=x.shape[1])
    return result


def sample_pair(result, generator=None):
    index = int(torch.multinomial(result.plan.flatten(), 1, generator=generator))
    return index//result.plan.shape[1], index % result.plan.shape[1]
