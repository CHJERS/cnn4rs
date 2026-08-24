package io.github.chjers.cnn4rs.core;

import java.util.Objects;

/** CHW max pooling with square windows. */
public final class MaxPool2D implements Layer {
    private final int windowSize;
    private final int stride;

    public MaxPool2D(int windowSize, int stride) {
        if (windowSize <= 0 || stride <= 0) {
            throw new InvalidTensorException("pool window size and stride must be positive");
        }
        this.windowSize = windowSize;
        this.stride = stride;
    }

    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "input must not be null");
        Shape shape = input.shape();
        Conv2D.requireRank(shape, 3, "pool input");
        int channels = shape.dimension(0);
        int inputHeight = shape.dimension(1);
        int inputWidth = shape.dimension(2);
        if (inputHeight < windowSize || inputWidth < windowSize
                || (inputHeight - windowSize) % stride != 0
                || (inputWidth - windowSize) % stride != 0) {
            throw new InvalidTensorException("pool window " + windowSize + " and stride " + stride
                    + " do not divide input spatial dimensions " + inputHeight + "x" + inputWidth);
        }
        int outputHeight = (inputHeight - windowSize) / stride + 1;
        int outputWidth = (inputWidth - windowSize) / stride + 1;
        float[] output = new float[channels * outputHeight * outputWidth];
        int offset = 0;
        for (int channel = 0; channel < channels; channel++) {
            for (int outputRow = 0; outputRow < outputHeight; outputRow++) {
                for (int outputColumn = 0; outputColumn < outputWidth; outputColumn++) {
                    float maximum = Float.NEGATIVE_INFINITY;
                    for (int row = 0; row < windowSize; row++) {
                        for (int column = 0; column < windowSize; column++) {
                            maximum = Math.max(maximum, input.get(channel, outputRow * stride + row,
                                    outputColumn * stride + column));
                        }
                    }
                    output[offset++] = maximum;
                }
            }
        }
        return Tensor.of(output, channels, outputHeight, outputWidth);
    }
}
