package io.github.chjers.cnn4rs.core;

import java.util.Objects;
import java.util.Random;

/** A rank-one fully connected layer. */
public final class Dense implements TrainableLayer {
    private final int inputSize;
    private final int outputSize;
    private final Tensor weights;
    private final float[] biases;

    public Dense(int inputSize, int outputSize, Random random) {
        validateConfiguration(inputSize, outputSize);
        Objects.requireNonNull(random, "random must not be null");
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        Shape parameterShape = Shape.of(outputSize, inputSize);
        float[] randomWeights = new float[parameterShape.elementCount()];
        for (int index = 0; index < randomWeights.length; index++) {
            randomWeights[index] = random.nextFloat() - 0.5f;
        }
        this.weights = Tensor.of(randomWeights, parameterShape.dimensions());
        this.biases = new float[outputSize];
    }

    private Dense(int inputSize, int outputSize, Tensor weights, float[] biases) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weights = Tensor.of(weights.toArray(), weights.shape().dimensions());
        this.biases = biases.clone();
    }

    public static Dense withParameters(int inputSize, int outputSize, Tensor weights, float[] biases) {
        validateConfiguration(inputSize, outputSize);
        Objects.requireNonNull(weights, "weights must not be null");
        Objects.requireNonNull(biases, "biases must not be null");
        Shape expected = Shape.of(outputSize, inputSize);
        if (!expected.equals(weights.shape())) {
            throw new InvalidTensorException("dense weights must have shape " + expected
                    + " but were " + weights.shape());
        }
        if (biases.length != outputSize) {
            throw new InvalidTensorException("dense bias count must be " + outputSize
                    + " but was " + biases.length);
        }
        return new Dense(inputSize, outputSize, weights, biases);
    }

    public static Dense withParameters(Tensor weights, float[] biases) {
        Objects.requireNonNull(weights, "weights must not be null");
        Conv2D.requireRank(weights.shape(), 2, "dense weights");
        return withParameters(weights.shape().dimension(1), weights.shape().dimension(0), weights, biases);
    }

    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "input must not be null");
        Conv2D.requireRank(input.shape(), 1, "dense input");
        if (input.shape().dimension(0) != inputSize) {
            throw new InvalidTensorException("dense layer expects " + inputSize + " input values but got "
                    + input.shape().dimension(0));
        }
        float[] output = new float[outputSize];
        for (int outputIndex = 0; outputIndex < outputSize; outputIndex++) {
            float sum = biases[outputIndex];
            for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
                sum += weights.get(outputIndex, inputIndex) * input.get(inputIndex);
            }
            output[outputIndex] = sum;
        }
        return Tensor.of(output, outputSize);
    }

    private static void validateConfiguration(int inputSize, int outputSize) {
        if (inputSize <= 0 || outputSize <= 0) {
            throw new InvalidTensorException("dense input and output sizes must be positive");
        }
    }
}
