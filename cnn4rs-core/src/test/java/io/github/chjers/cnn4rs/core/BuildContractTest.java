package io.github.chjers.cnn4rs.core;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BuildContractTest {
    @Test
    void exposesJava17LibraryCoordinates() {
        assertEquals("io.github.chjers.cnn4rs.core", Tensor.class.getPackageName());
    }
}
