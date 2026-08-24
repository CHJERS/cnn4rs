package io.github.chjers.cnn4rs.distributed;

import io.github.chjers.cnn4rs.core.TrainingExample;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** An immutable training batch for one specific model version. */
public record TrainingTask(String modelVersion, List<TrainingExample> examples) {
    public TrainingTask {
        modelVersion = requireVersion(modelVersion);
        examples = copyNonEmpty(examples, "training examples");
    }

    static String requireVersion(String modelVersion) {
        Objects.requireNonNull(modelVersion, "model version must not be null");
        if (modelVersion.isBlank()) {
            throw new IllegalArgumentException("model version must not be blank");
        }
        return modelVersion;
    }

    private static List<TrainingExample> copyNonEmpty(List<TrainingExample> examples, String description) {
        Objects.requireNonNull(examples, description + " must not be null");
        if (examples.isEmpty()) {
            throw new IllegalArgumentException(description + " must not be empty");
        }
        ArrayList<TrainingExample> copy = new ArrayList<>(examples.size());
        for (int index = 0; index < examples.size(); index++) {
            copy.add(Objects.requireNonNull(examples.get(index), description + " " + index + " must not be null"));
        }
        return List.copyOf(copy);
    }
}
