package io.github.chjers.cnn4rs.examples;

import io.github.chjers.cnn4rs.core.Models;
import io.github.chjers.cnn4rs.core.SequentialModel;
import io.github.chjers.cnn4rs.core.Tensor;

/** A deterministic, core-only classification example with no external input files. */
public final class InMemoryClassifierExample {
    private InMemoryClassifierExample() {
    }

    public static void main(String[] args) {
        System.out.println("predictedClass=" + classifyFirstSample());
    }

    /** Classifies a fixed two-feature sample with a reproducible model seed. */
    public static int classifyFirstSample() {
        SequentialModel model = Models.denseClassifier(2, 2, 42L);
        Tensor probabilities = model.predict(Tensor.of(new float[] {1f, 0f}, 2));
        return argMax(probabilities);
    }

    static int argMax(Tensor values) {
        float[] probabilities = values.toArray();
        int bestIndex = 0;
        for (int index = 1; index < probabilities.length; index++) {
            if (probabilities[index] > probabilities[bestIndex]) {
                bestIndex = index;
            }
        }
        return bestIndex;
    }
}
