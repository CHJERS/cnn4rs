package io.github.chjers.cnn4rs.examples;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ReadmeCommandTest {
    @Test
    void readmeExamplePrintsPrediction() {
        String output = captureStandardOut(() -> InMemoryClassifierExample.main(new String[0]));

        assertTrue(output.startsWith("predictedClass="));
    }

    private static String captureStandardOut(Runnable action) {
        PrintStream original = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        try (PrintStream replacement = new PrintStream(captured, true, StandardCharsets.UTF_8)) {
            System.setOut(replacement);
            action.run();
        } finally {
            System.setOut(original);
        }
        return captured.toString(StandardCharsets.UTF_8);
    }
}
