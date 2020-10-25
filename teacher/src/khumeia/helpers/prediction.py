"""
High level helpers for predicting. Contains mainly a tool to automatically convert a classification keras notebook into
a Predictor class
"""
import numpy as np

from khumeia.inference.predictor import Predictor


class KerasPredictor(Predictor):
    def __init__(self, model, batch_size, decision_threshold, rescale=1 / 255.):
        """

        Args:
            batch_size:
            model(Model):
        """
        super(KerasPredictor, self).__init__(batch_size=batch_size)
        self.model = model
        self.decision_threshold = decision_threshold
        self.rescale = rescale

    def predict_on_batch(self, tiles_data):
        tiles_data = np.asarray(tiles_data).astype(np.float32)
        tiles_data *= self.rescale
        tiles_results = self.model.predict_on_batch(tiles_data)
        tiles_results = tiles_results[:, 1]

        def _decision(proba):
            return "aircraft" if proba > self.decision_threshold else "background"

        tiles_results = list(map(_decision, tiles_results))
        return tiles_results


def keras_model_to_predictor(keras_model, batch_size=32, decision_threshold=0.5, rescale=1 / 255.):
    """

    Args:
        keras_model(Model):
        batch_size(int):
        decision_threshold(float):
        rescale(float):

    Returns:

    """
    return KerasPredictor(model=keras_model,
                          batch_size=batch_size,
                          decision_threshold=decision_threshold,
                          rescale=rescale)
