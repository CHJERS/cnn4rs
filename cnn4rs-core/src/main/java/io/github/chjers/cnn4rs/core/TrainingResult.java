package io.github.chjers.cnn4rs.core;

/** Summary of one completed batch update. */
public record TrainingResult(float loss) {
    public TrainingResult {
        if (!Float.isFinite(loss) || loss < 0f) {
            throw new IllegalArgumentException("training loss must be finite and non-negative");
        }
    }
}
