import copy
import numpy as np
# --- Your misfit ---
def misfit(estimated_image, ground_truth_image):
    """MSE misfit between estimated and ground truth images."""
    est = np.asarray(estimated_image, dtype=np.float32)
    gt  = np.asarray(ground_truth_image, dtype=np.float32)
    return np.mean((est - gt) ** 2)

# --- Flatten/unflatten helpers for PSO ---
def build_param_spec(param_ranges):
    """
    Returns a list of parameter specs in a fixed order.
    Each spec: dict(step=..., key=..., low=..., high=..., kind=..., post=callable)
    """
    specs = []
    for step, keys in param_ranges.items():
        for key, rng in keys.items():
            lo, hi = float(rng["low"]), float(rng["high"])

            # Heuristic typing rules
            kind = "float"
            if (step, key) in [("nlm", "patch_size"),
                               ("nlm", "patch_distance"),
                               ("bias_correct", "sigma"),
                               ("tv", "max_num_iter"),
                               ("otsu", "min_size"),
                               ("otsu", "opening_radius"),
                               ("otsu", "closing_radius")]:
                kind = "int"

            # Post-processing / constraints
            def make_post(step_, key_, kind_):
                def post(v):
                    # clamp
                    v = float(v)
                    # cast
                    if kind_ == "int":
                        v = int(round(v))
                        # common constraint: patch_size odd and >= 1
                        if step_ == "nlm" and key_ == "patch_size":
                            v = max(1, v)
                            if v % 2 == 0:
                                v += 1  # make odd
                        if key_ in ("opening_radius", "closing_radius"):
                            v = max(0, v)
                        if key_ == "min_size":
                            v = max(0, v)
                        if step_ == "tv" and key_ == "max_num_iter":
                            v = max(1, v)
                        return v
                    else:
                        return v
                return post

            specs.append({
                "step": step,
                "key": key,
                "low": lo,
                "high": hi,
                "kind": kind,
                "post": make_post(step, key, kind),
            })
    return specs


def vector_to_params(x_vec, specs, base_params):
    """
    Convert a PSO particle vector into your PARAMS dict (deep-copied and updated).
    """
    params = copy.deepcopy(base_params)

    for v, sp in zip(x_vec, specs):
        step, key = sp["step"], sp["key"]
        val = sp["post"](np.clip(v, sp["low"], sp["high"]))

        if step not in params:
            params[step] = {}
        params[step][key] = val

    # sanity constraints that involve multiple params
    if "contrast_stretch" in params:
        p_low = float(params["contrast_stretch"].get("p_low", 0.5))
        p_high = float(params["contrast_stretch"].get("p_high", 99.5))
        # ensure p_low < p_high
        if p_low >= p_high:
            mid = 0.5 * (p_low + p_high)
            params["contrast_stretch"]["p_low"] = max(0.0, mid - 0.5)
            params["contrast_stretch"]["p_high"] = min(100.0, mid + 0.5)

    return params


# --- Simple PSO implementation ---
def pso_optimize(
    objective_fn,
    dim,
    bounds_low,
    bounds_high,
    n_particles=24,
    n_iters=40,
    w=0.72,
    c1=1.49,
    c2=1.49,
    seed=0,
    verbose=True
):
    rng = np.random.default_rng(seed)

    # Initialize positions/velocities
    X = rng.uniform(bounds_low, bounds_high, size=(n_particles, dim))
    V = rng.uniform(-(bounds_high - bounds_low), (bounds_high - bounds_low), size=(n_particles, dim)) * 0.1

    # Personal bests
    pbest_X = X.copy()
    pbest_f = np.array([objective_fn(x) for x in X], dtype=float)

    # Global best
    g_idx = int(np.argmin(pbest_f))
    gbest_X = pbest_X[g_idx].copy()
    gbest_f = float(pbest_f[g_idx])

    if verbose:
        print(f"[PSO] init best = {gbest_f:.6e}")

    for it in range(1, n_iters + 1):
        r1 = rng.random(size=(n_particles, dim))
        r2 = rng.random(size=(n_particles, dim))

        # Velocity & position update
        V = w * V + c1 * r1 * (pbest_X - X) + c2 * r2 * (gbest_X - X)
        X = X + V

        # Clamp to bounds
        X = np.clip(X, bounds_low, bounds_high)

        # Evaluate
        fvals = np.array([objective_fn(x) for x in X], dtype=float)

        # Update personal bests
        improved = fvals < pbest_f
        pbest_X[improved] = X[improved]
        pbest_f[improved] = fvals[improved]

        # Update global best
        new_g_idx = int(np.argmin(pbest_f))
        new_g_f = float(pbest_f[new_g_idx])
        if new_g_f < gbest_f:
            gbest_f = new_g_f
            gbest_X = pbest_X[new_g_idx].copy()

        if verbose and (it == 1 or it % 5 == 0 or it == n_iters):
            print(f"[PSO] iter {it:03d}/{n_iters} best = {gbest_f:.6e}")

    return gbest_X, gbest_f


from joblib import Parallel, delayed

def pso_optimize_parallel(
    objective_fn,
    dim,
    bounds_low,
    bounds_high,
    n_particles=24,
    n_iters=40,
    w=0.72,
    c1=1.49,
    c2=1.49,
    seed=0,
    verbose=True,
    n_jobs=-1,          # -1 = use all cores
    backend="loky",     # process-based (best for CPU-heavy work)
    batch_size="auto",
):
    rng = np.random.default_rng(seed)

    bounds_low = np.asarray(bounds_low, dtype=float)
    bounds_high = np.asarray(bounds_high, dtype=float)
    span = bounds_high - bounds_low

    # Initialize positions/velocities
    X = rng.uniform(bounds_low, bounds_high, size=(n_particles, dim))
    V = rng.uniform(-span, span, size=(n_particles, dim)) * 0.1

    # Helper: parallel eval
    def eval_swarm(X_):
        return np.asarray(
            Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size)(
                delayed(objective_fn)(X_[i]) for i in range(X_.shape[0])
            ),
            dtype=float,
        )

    # Personal bests
    pbest_X = X.copy()
    pbest_f = eval_swarm(X)

    # Global best
    g_idx = int(np.argmin(pbest_f))
    gbest_X = pbest_X[g_idx].copy()
    gbest_f = float(pbest_f[g_idx])

    if verbose:
        print(f"[PSO] init best = {gbest_f:.6e}")

    for it in range(1, n_iters + 1):
        r1 = rng.random(size=(n_particles, dim))
        r2 = rng.random(size=(n_particles, dim))

        # Velocity & position update
        V = w * V + c1 * r1 * (pbest_X - X) + c2 * r2 * (gbest_X - X)
        X = X + V

        # Clamp
        X = np.clip(X, bounds_low, bounds_high)

        # Parallel evaluate
        fvals = eval_swarm(X)

        # Update personal bests
        improved = fvals < pbest_f
        pbest_X[improved] = X[improved]
        pbest_f[improved] = fvals[improved]

        # Update global best
        new_g_idx = int(np.argmin(pbest_f))
        new_g_f = float(pbest_f[new_g_idx])
        if new_g_f < gbest_f:
            gbest_f = new_g_f
            gbest_X = pbest_X[new_g_idx].copy()

        if verbose and (it == 1 or it % 5 == 0 or it == n_iters):
            print(f"[PSO] iter {it:03d}/{n_iters} best = {gbest_f:.6e}")

    return gbest_X, gbest_f
