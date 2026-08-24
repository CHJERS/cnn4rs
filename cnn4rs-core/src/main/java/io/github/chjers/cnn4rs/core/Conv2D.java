package io.github.chjers.cnn4rs.core;

import java.util.Objects;
import java.util.Random;

/** Valid, CHW two-dimensional convolution with a square kernel. */
public final class Conv2D implements TrainableLayer {
    private final int inputChannels;
    private final int outputChannels;
    private final int kernelSize;
    private final int stride;
    private final Tensor weights;
    private final float[] biases;

    public Conv2D(int inputChannels, int outputChannels, int kernelSize, int stride, Random random) {
        validateConfiguration(inputChannels, outputChannels, kernelSize, stride);
        Objects.requireNonNull(random, "random must not be null");
        this.inputChannels = inputChannels;
        this.outputChannels = outputChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        float[] randomWeights = new float[outputChannels * inputChannels * kernelSize * kernelSize];
        for (int index = 0; index < randomWeights.length; index++) {
            randomWeights[index] = random.nextFloat() - 0.5f;
        }
        this.weights = Tensor.of(randomWeights, outputChannels, inputChannels, kernelSize, kernelSize);
        this.biases = new float[outputChannels];
    }

    private Conv2D(int inputChannels, int outputChannels, int kernelSize, int stride,
                   Tensor weights, float[] biases) {
        this.inputChannels = inputChannels;
        this.outputChannels = outputChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.weights = Tensor.of(weights.toArray(), weights.shape().dimensions());
        this.biases = biases.clone();
    }

    public static Conv2D withParameters(int inputChannels, int outputChannels, int kernelSize,
                                        int stride, Tensor weights, float[] biases) {
        validateConfiguration(inputChannels, outputChannels, kernelSize, stride);
        Objects.requireNonNull(weights, "weights must not be null");
        Objects.requireNonNull(biases, "biases must not be null");
        Shape expected = Shape.of(outputChannels, inputChannels, kernelSize, kernelSize);
        if (!expected.equals(weights.shape())) {
            throw new InvalidTensorException("convolution weights must have shape " + expected
                    + " but were " + weights.shape());
        }
        if (biases.length != outputChannels) {
            throw new InvalidTensorException("convolution bias count must be " + outputChannels
                    + " but was " + biases.length);
        }
        return new Conv2D(inputChannels, outputChannels, kernelSize, stride, weights, biases);
    }

    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "input must not be null");
        Shape inputShape = input.shape();
        requireRank(inputShape, 3, "convolution input");
        if (inputShape.dimension(0) != inputChannels) {
            throw new InvalidTensorException("convolution expects " + inputChannels + " input channels but got "
                    + inputShape.dimension(0));
        }
        int inputHeight = inputShape.dimension(1);
        int inputWidth = inputShape.dimension(2);
        if (inputHeight < kernelSize || inputWidth < kernelSize) {
            throw new InvalidTensorException("convolution kernel " + kernelSize
                    + " exceeds input spatial dimensions " + inputHeight + "x" + inputWidth);
        }
        int outputHeight = (inputHeight - kernelSize) / stride + 1;
        int outputWidth = (inputWidth - kernelSize) / stride + 1;
        float[] output = new float[outputChannels * outputHeight * outputWidth];
        int offset = 0;
        for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
            for (int outputRow = 0; outputRow < outputHeight; outputRow++) {
                for (int outputColumn = 0; outputColumn < outputWidth; outputColumn++) {
                    float sum = biases[outputChannel];
                    for (int inputChannel = 0; inputChannel < inputChannels; inputChannel++) {
                        for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
                            for (int kernelColumn = 0; kernelColumn < kernelSize; kernelColumn++) {
                                sum += input.get(inputChannel, outputRow * stride + kernelRow,
                                                outputColumn * stride + kernelColumn)
                                        * weights.get(outputChannel, inputChannel, kernelRow, kernelColumn);
                            }
                        }
                    }
                    output[offset++] = sum;
                }
            }
        }
        return Tensor.of(output, outputChannels, outputHeight, outputWidth);
    }

    private static void validateConfiguration(int inputChannels, int outputChannels,
                                              int kernelSize, int stride) {
        if (inputChannels <= 0 || outputChannels <= 0 || kernelSize <= 0 || stride <= 0) {
            throw new InvalidTensorException("input channels, output channels, kernel size, and stride must be positive");
        }
    }

    static void requireRank(Shape shape, int expectedRank, String description) {
        if (shape.rank() != expectedRank) {
            throw new InvalidTensorException(description + " must have rank " + expectedRank
                    + " but had rank " + shape.rank());
        }
    }
}
