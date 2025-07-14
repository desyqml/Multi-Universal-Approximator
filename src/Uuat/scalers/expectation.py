import numpy as np
from scipy.special import erf, erfinv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler


class Gaussian:
    def __init__(self):
        scale_factor = 3
        self.clip_min = None
        self.clip_max = None

        self.transformer=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("power", PowerTransformer()),
                    (
                        "erf_transform",
                        FunctionTransformer(
                            func=lambda x: erf(x / scale_factor),
                            inverse_func=lambda x: scale_factor * erfinv(x),
                            check_inverse=False,
                        ),
                    ),
                    (
                        "clip",
                        FunctionTransformer(
                            func=lambda x: np.clip(x, self.clip_min, self.clip_max),
                            inverse_func=lambda x: x,
                        ),
                    ),
                ]
            )
        

    def fit(self, decoded_data: np.ndarray):
        self.clip_min = np.min(decoded_data, axis=0)
        self.clip_max = np.max(decoded_data, axis=0)
        self.transformer.fit(decoded_data)

    def encode(self, decoded_data: np.ndarray) -> np.ndarray:
        """Encode data into normalized range."""
        return self.transformer.transform(decoded_data)

    def decode(self, encoded_data: np.ndarray) -> np.ndarray:
        """Decode normalized data into original range."""
        return self.transformer.inverse_transform(encoded_data)