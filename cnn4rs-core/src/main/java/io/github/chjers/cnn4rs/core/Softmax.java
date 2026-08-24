package io.github.chjers.cnn4rs.core;

import java.util.Objects;

/** Numerically stable rank-one softmax activation. */
public final class Softmax implements Layer {
    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "input must not be null");
        Conv2D.requireRank(input.shape(), 1, "softmax input");
        float[] logits = input.toArray();
        float maximum = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {
            maximum = Math.max(maximum, logit);
        }
        double sum = 0d;
        float[] probabilities = new float[logits.length];
        for (int index = 0; index < logits.length; index++) {
            probabilities[index] = (float) Math.exp(logits[index] - maximum);
            sum += probabilities[index];
        }
        for (int index = 0; index < probabilities.length; index++) {
            probabilities[index] /= (float) sum;
        }
        return Tensor.of(probabilities, logits.length);
    }
}
