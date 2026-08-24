package io.github.chjers.cnn4rs.core;

import java.util.Arrays;
import java.util.Objects;

/** Immutable dimensions of a tensor. */
public final class Shape {
    private final int[] dimensions;
    private final int elementCount;

    private Shape(int[] dimensions, int elementCount) {
        this.dimensions = dimensions;
        this.elementCount = elementCount;
    }

    public static Shape of(int... dimensions) {
        Objects.requireNonNull(dimensions, "dimensions must not be null");
        if (dimensions.length == 0) {
            throw new InvalidTensorException("a shape must have at least one dimension");
        }
        int count = 1;
        for (int axis = 0; axis < dimensions.length; axis++) {
            int dimension = dimensions[axis];
            if (dimension <= 0) {
                throw new InvalidTensorException("dimension at axis " + axis
                        + " must be positive, but was " + dimension);
            }
            try {
                count = Math.multiplyExact(count, dimension);
            } catch (ArithmeticException exception) {
                throw new InvalidTensorException("shape element count overflows int at axis " + axis);
            }
        }
        return new Shape(Arrays.copyOf(dimensions, dimensions.length), count);
    }

    public int rank() { return dimensions.length; }

    public int[] dimensions() { return Arrays.copyOf(dimensions, dimensions.length); }

    public int dimension(int axis) {
        if (axis < 0 || axis >= dimensions.length) {
            throw new InvalidTensorException("axis " + axis + " is outside rank " + dimensions.length);
        }
        return dimensions[axis];
    }

    int elementCount() { return elementCount; }

    @Override
    public boolean equals(Object other) {
        return other instanceof Shape that && Arrays.equals(dimensions, that.dimensions);
    }

    @Override
    public int hashCode() { return Arrays.hashCode(dimensions); }

    @Override
    public String toString() { return Arrays.toString(dimensions); }
}
