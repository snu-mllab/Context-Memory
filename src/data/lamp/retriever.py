from .utils import ppep
import numpy as np
class BaseRetriever:
    def __init__(self, k, seed=42, random_k=False, eval_seed=1337):
        self.k = k
        self.rng = np.random.default_rng(seed=seed)
        self.random_k = random_k
        print(f"Random k: {random_k}")
        print(f"Eval seed: {eval_seed}")
        self.eval_seed = eval_seed
        self.eval_rng = np.random.default_rng(seed=eval_seed)
        self.k_rng = np.random.default_rng(seed=0)

    def reset_eval_seed(self):
        print("Reset Eval Seed!")
        del self.eval_rng
        self.eval_rng = np.random.default_rng(seed=self.eval_seed)

    def __call__(self, input_, profile, eval: bool=False):
        """
        returns selected profile w.r.t. inputs
        """

        if eval:
            n = len(profile)
            k = 16
            idx = self.eval_rng.choice(n, size=k, replace=n < k)

            ret = [profile[i] for i in idx]
            if len(ret) >= self.k:
                ret = ret[:self.k]
        else:
            n = len(profile)
            k = self.k
            if self.random_k:
                k = self.rng.integers(low=1, high=self.k+1)

            idx = self.rng.choice(n, size=k, replace=n < k)
            ret = [profile[i] for i in idx]

        return ret
