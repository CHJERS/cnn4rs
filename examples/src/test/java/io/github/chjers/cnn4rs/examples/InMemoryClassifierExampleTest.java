package io.github.chjers.cnn4rs.examples;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class InMemoryClassifierExampleTest {
    @Test
    void inMemoryExampleReturnsAClassIndex() {
        assertEquals(0, InMemoryClassifierExample.classifyFirstSample());
    }
}
