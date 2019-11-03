import argparse

from mlp.parser import ConfigParser, InputParser
from mlp.perceptron import Activation, DenseLayer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--input', required=True)

    return ap.parse_args()


def forward_pass(input_data, thresholds, weights):
    for i, (thresholds, weights)in enumerate(zip(thresholds, weights)):
        input_shape = input_data.shape[-1]

        layer = DenseLayer(weights)
        activation = Activation(thresholds)

        layer_output = layer(input_data)
        outputs = input_data = activation(layer_output)

        print(f"Layer {i}")
        print("   input shape:", input_shape)
        print("   output shape:", outputs.shape[-1], '\n')

    return outputs


def display_outputs(outputs):
    for i, output in enumerate(outputs.T):
        print(f'Output_{i} [size {len(output)}]:  {output}')


def save_outputs(outputs, path='output.txt'):
    output_lines = []
    for i, output in enumerate(outputs.T):
        output = map(lambda x: str(int(x)), output)
        output_lines.append(" ".join(output) + '\n')

    with open(path, 'w') as f:
        f.writelines(output_lines)


if __name__ == '__main__':
    args = parse_args()

    cfp = ConfigParser()
    cfp.parse(args.config)

    inp = InputParser()
    inp.parse(args.input)

    outputs = forward_pass(inp.input, cfp.thresholds(), cfp.weights())

    display_outputs(outputs)
    save_outputs(outputs)
