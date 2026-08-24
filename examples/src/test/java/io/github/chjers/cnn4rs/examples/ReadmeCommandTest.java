package io.github.chjers.cnn4rs.examples;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ReadmeCommandTest {
    @Test
    void readmeExamplePrintsPrediction() {
        String output = captureStandardOut(() -> InMemoryClassifierExample.main(new String[0]));

        assertTrue(output.startsWith("predictedClass="));
    }

    @Test
    void readmeDocumentsPackageAndClasspathLaunchCommands() throws IOException {
        String readme = Files.readString(Path.of("..", "README.md"));
        String packageCommand = "mvn -B -ntp -pl examples -am package";
        String launchCommand = "java -cp examples/target/classes:cnn4rs-core/target/classes "
                + "io.github.chjers.cnn4rs.examples.InMemoryClassifierExample";

        assertEquals(2, occurrences(readme, packageCommand));
        assertEquals(2, occurrences(readme, launchCommand));
    }

    private static int occurrences(String text, String fragment) {
        return text.split(java.util.regex.Pattern.quote(fragment), -1).length - 1;
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
