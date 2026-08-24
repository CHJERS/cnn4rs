# cnn4rs

`cnn4rs` is a Java 17, learning-oriented convolutional neural-network library for remote-sensing research.

The Maven reactor contains `cnn4rs-core`, `cnn4rs-geospatial`, `cnn4rs-distributed`, and `cnn4rs-examples`. The core module uses only the JDK; GDAL is not required to build it. The distributed module is experimental and is not presented as production-ready.

## Legacy implementation

The original 2020 research implementation, including its historical IDE metadata, is preserved unchanged in `legacy/` for reference only. It is excluded from the Maven build and is not part of the public API.

The thesis PDF in `docs/` is preserved unchanged as source material. Please consult its rights notice and confirm any reuse permissions independently.

## Build

Run the full reactor test suite with JDK 17 and Maven:

```bash
mvn test
```

## License

This project is licensed under the [MIT License](LICENSE).
