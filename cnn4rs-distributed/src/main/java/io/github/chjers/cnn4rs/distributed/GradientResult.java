package io.github.chjers.cnn4rs.distributed;

import io.github.chjers.cnn4rs.core.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Immutable gradients calculated for a non-empty batch of one model version. */
public record GradientResult(String modelVersion, int sampleCount, List<Tensor> gradients) {
    public GradientResult {
        modelVersion = TrainingTask.requireVersion(modelVersion);
        if (sampleCount <= 0) {
            throw new IllegalArgumentException("sample count must be positive");
        }
        gradients = copyAndValidateGradients(gradients);
    }

    private static List<Tensor> copyAndValidateGradients(List<Tensor> gradients) {
        Objects.requireNonNull(gradients, "gradients must not be null");
        if (gradients.isEmpty()) {
            throw new IllegalArgumentException("gradients must not be empty");
        }
        ArrayList<Tensor> copy = new ArrayList<>(gradients.size());
        for (int gradientIndex = 0; gradientIndex < gradients.size(); gradientIndex++) {
            Tensor gradient = Objects.requireNonNull(gradients.get(gradientIndex),
                    "gradient " + gradientIndex + " must not be null");
            float[] values = gradient.toArray();
            for (int valueIndex = 0; valueIndex < values.length; valueIndex++) {
                if (!Float.isFinite(values[valueIndex])) {
                    throw new IllegalArgumentException("gradient " + gradientIndex
                            + " has a non-finite value at index " + valueIndex);
                }
            }
            copy.add(gradient);
        }
        return List.copyOf(copy);
    }
}
