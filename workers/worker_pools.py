import multiprocessing as mp
from typing import Optional
from configuration import Config


class WorkerPools:
    """
    Standalone pool manager so modules that need workers don't rely on diffusion internals.

    Currently provides:
    - `nucleation_pool`: mp.Pool for nucleation subblock workers
    - `dissolution_pool`: mp.Pool for dissolution subblock workers
    - optional `jmatpro_pool`: persistent JMatPro worker system (created lazily)
    """

    def __init__(self, n_outward_workers: int, n_inward_workers: int):
        self.n_outward_workers = int(n_outward_workers)
        self.n_inward_workers = int(n_inward_workers)

        # These are used by precips/dissolution when using the shared-memory subblock kernels.
        self.nucleation_pool = mp.Pool(
            self.n_inward_workers,
            maxtasksperchild=int(getattr(Config, "MAX_TASK_PER_CHILD", 0)) or None,
        )
        self.dissolution_pool = mp.Pool(
            self.n_outward_workers,
            maxtasksperchild=int(getattr(Config, "MAX_TASK_PER_CHILD", 0)) or None,
        )

        self._jmatpro_pool = None

        # Diffusion pools: one mp.Pool per concurrently diffused element.
        self._diffusion_pools = None
        self._diffusion_signature = None  # tuple of pool sizes in the same order as elements

    def get_diffusion_pools(self, elements):
        """
        Return one diffusion mp.Pool per element (order-preserving).

        outward elements get pools of size `n_outward_workers`,
        inward elements get pools of size `n_inward_workers`.
        Pools are cached and recreated if the requested element_type sequence changes.
        """
        needed_sizes = []
        for e in elements:
            et = getattr(e, "element_type", "outward")
            needed_sizes.append(self.n_outward_workers if et == "outward" else self.n_inward_workers)
        signature = tuple(needed_sizes)

        if self._diffusion_pools is not None and signature == self._diffusion_signature:
            return self._diffusion_pools

        # Recreate pools when element set/ordering changes.
        if self._diffusion_pools is not None:
            for p in self._diffusion_pools:
                p.close()
                p.join()

        maxtasks = int(getattr(Config, "MAX_TASK_PER_CHILD", 0)) or None
        self._diffusion_pools = [mp.Pool(sz, maxtasksperchild=maxtasks) for sz in needed_sizes]
        self._diffusion_signature = signature
        return self._diffusion_pools

    def get_jmatpro_pool(
        self,
        *,
        num_workers: int,
        temperature: float,
        task_timeout: float = 300.0,
        max_retries: int = 3,
    ):
        """
        Lazily create a JMatProWorkerPool and return it.
        Uses a separate worker system (processes + task queues) from mp.Pool.
        """
        if self._jmatpro_pool is None:
            # Lazy import so CA doesn't require JMatPro dependencies unless used.
            from thermodynamics import JMatProWorkerPool

            self._jmatpro_pool = JMatProWorkerPool(
                num_workers=int(num_workers),
                temperature=float(temperature),
                task_timeout=float(task_timeout),
                max_retries=int(max_retries),
            )
        return self._jmatpro_pool

    def close(self):
        """Close mp.Pool instances and any lazily-created JMatPro worker pool."""
        if self._diffusion_pools is not None:
            for p in self._diffusion_pools:
                p.close()
                p.join()
            self._diffusion_pools = None
            self._diffusion_signature = None

        if self.nucleation_pool is not None:
            self.nucleation_pool.close()
            self.nucleation_pool.join()
            self.nucleation_pool = None
        if self.dissolution_pool is not None:
            self.dissolution_pool.close()
            self.dissolution_pool.join()
            self.dissolution_pool = None
        if self._jmatpro_pool is not None:
            try:
                self._jmatpro_pool.shutdown()
            finally:
                self._jmatpro_pool = None

    @property
    def jmatpro_pool(self):
        return self._jmatpro_pool

