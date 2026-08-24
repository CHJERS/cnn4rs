package io.github.chjers.cnn4rs.core;

import java.util.List;
import java.util.Objects;

/** Performs deterministic, accumulated stochastic-gradient updates on a sequential classifier. */
public final class Trainer {
    private final SequentialModel model;
    private final float learningRate;
    private final CrossEntropyLoss lossFunction;

    public Trainer(SequentialModel model, float learningRate) {
        this(model, learningRate, new CrossEntropyLoss());
    }

    public Trainer(SequentialModel model, float learningRate, CrossEntropyLoss lossFunction) {
        this.model = Objects.requireNonNull(model, "model must not be null");
        if (!Float.isFinite(learningRate) || learningRate <= 0f) {
            throw new IllegalArgumentException("learning rate must be finite and positive");
        }
        this.learningRate = learningRate;
        this.lossFunction = Objects.requireNonNull(lossFunction, "loss function must not be null");
        List<Layer> layers = model.layers();
        if (layers.isEmpty() || !(layers.get(layers.size() - 1) instanceof Softmax)) {
            throw new IllegalArgumentException("training requires Softmax as the final layer");
        }
    }

    public float loss(List<TrainingExample> examples) {
        List<TrainingExample> batch = requireBatch(examples);
        double total = 0d;
        for (TrainingExample example : batch) {
            total += lossFunction.calculate(model.predict(example.input()), example.label());
        }
        return checkedAverage(total, batch.size());
    }

    public TrainingResult trainBatch(List<TrainingExample> examples) {
        List<TrainingExample> batch = requireBatch(examples);
        List<Layer> layers = model.layers();
        LayerGradients[] gradients = createGradientAccumulators(layers);
        double totalLoss = 0d;

        for (TrainingExample example : batch) {
            Tensor[] values = forwardForTraining(layers, example.input());
            Tensor probabilities = values[values.length - 1];
            totalLoss += lossFunction.calculate(probabilities, example.label());

            float[] combinedGradient = probabilities.toArray();
            combinedGradient[example.label()] -= 1f;
            Tensor gradient = Tensor.of(combinedGradient, probabilities.shape().dimensions());
            for (int index = layers.size() - 1; index >= 0; index--) {
                Layer layer = layers.get(index);
                if (index == layers.size() - 1 && layer instanceof Softmax) {
                    continue;
                }
                gradient = backward(layer, values[index], values[index + 1], gradient,
                        gradients[index]);
            }
        }

        float scale = learningRate / batch.size();
        for (int index = 0; index < layers.size(); index++) {
            apply(layers.get(index), gradients[index], scale);
        }
        return new TrainingResult(checkedAverage(totalLoss, batch.size()));
    }

    private static List<TrainingExample> requireBatch(List<TrainingExample> examples) {
        Objects.requireNonNull(examples, "training examples must not be null");
        if (examples.isEmpty()) {
            throw new IllegalArgumentException("training batch must not be empty");
        }
        for (int index = 0; index < examples.size(); index++) {
            Objects.requireNonNull(examples.get(index),
                    "training example " + index + " must not be null");
        }
        return examples;
    }

    private static float checkedAverage(double total, int count) {
        float average = (float) (total / count);
        if (!Float.isFinite(average)) {
            throw new IllegalStateException("batch loss is not finite");
        }
        return average;
    }

    private static Tensor[] forwardForTraining(List<Layer> layers, Tensor input) {
        Tensor[] values = new Tensor[layers.size() + 1];
        values[0] = input;
        for (int index = 0; index < layers.size(); index++) {
            values[index + 1] = layers.get(index).forward(values[index]);
        }
        return values;
    }

    private static LayerGradients[] createGradientAccumulators(List<Layer> layers) {
        LayerGradients[] gradients = new LayerGradients[layers.size()];
        for (int index = 0; index < layers.size(); index++) {
            Layer layer = layers.get(index);
            if (layer instanceof Dense dense) {
                gradients[index] = new LayerGradients(
                        dense.inputSize() * dense.outputSize(), dense.outputSize());
            } else if (layer instanceof Conv2D convolution) {
                gradients[index] = new LayerGradients(
                        convolution.outputChannels() * convolution.inputChannels()
                                * convolution.kernelSize() * convolution.kernelSize(),
                        convolution.outputChannels());
            }
        }
        return gradients;
    }

    private static Tensor backward(Layer layer, Tensor input, Tensor output,
                                   Tensor outputGradient, LayerGradients gradients) {
        if (layer instanceof Dense dense) {
            return backwardDense(dense, input, outputGradient, gradients);
        }
        if (layer instanceof Conv2D convolution) {
            return backwardConvolution(convolution, input, outputGradient, gradients);
        }
        if (layer instanceof ReLU) {
            return backwardRelu(input, outputGradient);
        }
        if (layer instanceof MaxPool2D pool) {
            return backwardPool(pool, input, outputGradient);
        }
        if (layer instanceof Softmax) {
            return backwardSoftmax(output, outputGradient);
        }
        throw new IllegalArgumentException("training does not support layer "
                + layer.getClass().getName());
    }

    private static Tensor backwardDense(Dense dense, Tensor input, Tensor outputGradient,
                                        LayerGradients gradients) {
        float[] inputGradient = new float[dense.inputSize()];
        for (int outputIndex = 0; outputIndex < dense.outputSize(); outputIndex++) {
            float gradient = outputGradient.get(outputIndex);
            gradients.biases[outputIndex] += gradient;
            for (int inputIndex = 0; inputIndex < dense.inputSize(); inputIndex++) {
                int weightIndex = outputIndex * dense.inputSize() + inputIndex;
                gradients.weights[weightIndex] += gradient * input.get(inputIndex);
                inputGradient[inputIndex] += gradient * dense.weights().get(outputIndex, inputIndex);
            }
        }
        return Tensor.of(inputGradient, dense.inputSize());
    }

    private static Tensor backwardConvolution(Conv2D convolution, Tensor input,
                                              Tensor outputGradient, LayerGradients gradients) {
        int inputHeight = input.shape().dimension(1);
        int inputWidth = input.shape().dimension(2);
        int outputHeight = outputGradient.shape().dimension(1);
        int outputWidth = outputGradient.shape().dimension(2);
        float[] inputGradient = new float[input.shape().elementCount()];

        for (int outputChannel = 0; outputChannel < convolution.outputChannels(); outputChannel++) {
            for (int outputRow = 0; outputRow < outputHeight; outputRow++) {
                for (int outputColumn = 0; outputColumn < outputWidth; outputColumn++) {
                    float gradient = outputGradient.get(outputChannel, outputRow, outputColumn);
                    gradients.biases[outputChannel] += gradient;
                    for (int inputChannel = 0; inputChannel < convolution.inputChannels(); inputChannel++) {
                        for (int kernelRow = 0; kernelRow < convolution.kernelSize(); kernelRow++) {
                            for (int kernelColumn = 0; kernelColumn < convolution.kernelSize(); kernelColumn++) {
                                int inputRow = outputRow * convolution.stride() + kernelRow;
                                int inputColumn = outputColumn * convolution.stride() + kernelColumn;
                                int weightIndex = (((outputChannel * convolution.inputChannels()
                                        + inputChannel) * convolution.kernelSize() + kernelRow)
                                        * convolution.kernelSize()) + kernelColumn;
                                gradients.weights[weightIndex] += gradient
                                        * input.get(inputChannel, inputRow, inputColumn);
                                int inputIndex = (inputChannel * inputHeight + inputRow) * inputWidth
                                        + inputColumn;
                                inputGradient[inputIndex] += gradient * convolution.weights().get(
                                        outputChannel, inputChannel, kernelRow, kernelColumn);
                            }
                        }
                    }
                }
            }
        }
        return Tensor.of(inputGradient, input.shape().dimensions());
    }

    private static Tensor backwardRelu(Tensor input, Tensor outputGradient) {
        float[] gradient = outputGradient.toArray();
        float[] inputValues = input.toArray();
        for (int index = 0; index < gradient.length; index++) {
            if (inputValues[index] <= 0f) {
                gradient[index] = 0f;
            }
        }
        return Tensor.of(gradient, input.shape().dimensions());
    }

    private static Tensor backwardPool(MaxPool2D pool, Tensor input, Tensor outputGradient) {
        int channels = input.shape().dimension(0);
        int inputHeight = input.shape().dimension(1);
        int inputWidth = input.shape().dimension(2);
        int outputHeight = outputGradient.shape().dimension(1);
        int outputWidth = outputGradient.shape().dimension(2);
        float[] inputGradient = new float[input.shape().elementCount()];
        for (int channel = 0; channel < channels; channel++) {
            for (int outputRow = 0; outputRow < outputHeight; outputRow++) {
                for (int outputColumn = 0; outputColumn < outputWidth; outputColumn++) {
                    int bestRow = 0;
                    int bestColumn = 0;
                    float maximum = Float.NEGATIVE_INFINITY;
                    for (int row = 0; row < pool.windowSize(); row++) {
                        for (int column = 0; column < pool.windowSize(); column++) {
                            float value = input.get(channel, outputRow * pool.stride() + row,
                                    outputColumn * pool.stride() + column);
                            if (value > maximum) {
                                maximum = value;
                                bestRow = row;
                                bestColumn = column;
                            }
                        }
                    }
                    int inputRow = outputRow * pool.stride() + bestRow;
                    int inputColumn = outputColumn * pool.stride() + bestColumn;
                    int inputIndex = (channel * inputHeight + inputRow) * inputWidth + inputColumn;
                    inputGradient[inputIndex] += outputGradient.get(channel, outputRow, outputColumn);
                }
            }
        }
        return Tensor.of(inputGradient, input.shape().dimensions());
    }

    private static Tensor backwardSoftmax(Tensor output, Tensor outputGradient) {
        float dotProduct = 0f;
        for (int index = 0; index < output.shape().dimension(0); index++) {
            dotProduct += outputGradient.get(index) * output.get(index);
        }
        float[] inputGradient = new float[output.shape().dimension(0)];
        for (int index = 0; index < inputGradient.length; index++) {
            inputGradient[index] = output.get(index) * (outputGradient.get(index) - dotProduct);
        }
        return Tensor.of(inputGradient, inputGradient.length);
    }

    private static void apply(Layer layer, LayerGradients gradients, float scale) {
        if (layer instanceof Dense dense) {
            dense.applyGradients(gradients.weights, gradients.biases, scale);
        } else if (layer instanceof Conv2D convolution) {
            convolution.applyGradients(gradients.weights, gradients.biases, scale);
        }
    }

    private static final class LayerGradients {
        private final float[] weights;
        private final float[] biases;

        private LayerGradients(int weightCount, int biasCount) {
            weights = new float[weightCount];
            biases = new float[biasCount];
        }
    }
}
