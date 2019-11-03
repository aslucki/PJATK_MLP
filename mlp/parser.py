import numpy as np


class InputParser:

    def __init__(self):
        self._input = []

    @property
    def input(self):
        return self._input

    def parse(self, input_path):
        with open(input_path, 'r') as f:
            lines = f.readlines()

        processed_input = []
        for line in lines:
            if line == '\n':
                continue
            processed_input.append(line.strip().split())

        input_data = np.asarray(processed_input, dtype=np.float32).T

        self._input = input_data


class ConfigParser:
    def __init__(self):
        self._weights = []
        self._thresholds = []

    def weights(self):
        return self._weights

    def thresholds(self):
        return self._thresholds

    @staticmethod
    def _retrieve_thresholds(read_lines):
        def process_thresholds(thresholds):
            thresholds_arrays = []
            for threshold_list in thresholds:
                converted =\
                    np.asarray(threshold_list, dtype=np.float32).T
                thresholds_arrays.append(converted)

            return thresholds_arrays

        thresholds = []
        retrieve_line = True

        for line in read_lines:

            if line == '\n':
                retrieve_line = not retrieve_line
                continue

            if retrieve_line:
                thresholds.append(line.strip().split())

        return process_thresholds(thresholds)

    @staticmethod
    def _retrieve_weights(read_lines):

        def process_weights(weights):
            weights_arrays = []
            for weight_list in weights:
                transposed =\
                    np.asarray(weight_list, dtype=np.float32).T
                weights_arrays.append(transposed)

            return weights_arrays

        all_layers_weights = []
        weights = []
        retrieve_line = False

        for line in read_lines:
            if line == '\n':
                retrieve_line = not retrieve_line
                if weights:
                    all_layers_weights.append(weights)
                weights = []
                continue

            if retrieve_line:
                weights.append(line.strip().split())

        if weights:
            all_layers_weights.append(weights)

        return process_weights(all_layers_weights)

    def parse(self, config_path):
        with open(config_path, 'r') as f:
            lines = f.readlines()

        self._thresholds = self._retrieve_thresholds(lines)
        self._weights = self._retrieve_weights(lines)
