package io.github.chjers.cnn4rs.core;

import java.util.Arrays;
import java.util.Objects;

/** Immutable multidimensional, row-major floating-point data. */
public final class Tensor {
    private final Shape shape;
    private final float[] values;

    private Tensor(Shape shape, float[] values) {
        this.shape = shape;
        this.values = values;
    }

    /** Creates a tensor from values stored in row-major order. */
    public static Tensor of(float[] values, int... shape) {
        Objects.requireNonNull(values, "values must not be null");
        Shape tensorShape = Shape.of(shape);
        if (values.length != tensorShape.elementCount()) {
            throw new InvalidTensorException("value count " + values.length
                    + " does not match shape " + tensorShape + " ("
                    + tensorShape.elementCount() + " elements)");
        }
        return new Tensor(tensorShape, Arrays.copyOf(values, values.length));
    }

    /** Creates an all-zero tensor with the supplied dimensions. */
    public static Tensor zeros(int... shape) {
        Shape tensorShape = Shape.of(shape);
        return new Tensor(tensorShape, new float[tensorShape.elementCount()]);
    }

    public Shape shape() {
        return shape;
    }

    /** Returns the element at the supplied zero-based indices. */
    public float get(int... indices) {
        return values[offset(indices)];
    }

    /** Returns a copy with one element changed. */
    public Tensor with(int[] indices, float value) {
        float[] changed = Arrays.copyOf(values, values.length);
        changed[offset(indices)] = value;
        return new Tensor(shape, changed);
    }

    /** Returns a defensive copy of this tensor's row-major values. */
    public float[] toArray() {
        return Arrays.copyOf(values, values.length);
    }

    private int offset(int... indices) {
        if (indices == null) {
            throw new InvalidTensorException("indices must not be null for shape " + shape);
        }
        if (indices.length != shape.rank()) {
            throw new InvalidTensorException("expected " + shape.rank() + " indices for shape "
                    + shape + " but received " + indices.length);
        }

        int offset = 0;
        for (int axis = 0; axis < indices.length; axis++) {
            int index = indices[axis];
            int dimension = shape.dimension(axis);
            if (index < 0 || index >= dimension) {
                throw new InvalidTensorException("index " + index + " is outside axis " + axis
                        + " with size " + dimension + " for shape " + shape);
            }
            offset = offset * dimension + index;
        }
        return offset;
    }
}
