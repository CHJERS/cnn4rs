# Contributing

Thank you for contributing to cnn4rs. Please open an issue before substantial changes, keep public APIs under `io.github.chjers.cnn4rs`, and add focused tests for behavior changes.

Use JDK 17 and run `mvn -B -ntp test` before submitting a pull request. Keep core code free of GDAL and other native runtime dependencies; optional adapters must provide actionable diagnostics when their runtime prerequisites are absent.

Do not modify files under `legacy/` or the preserved thesis PDF in `docs/`; both are retained unchanged as historical source material. Follow the pull request template, update release documentation when public APIs or setup instructions change, and keep distributed changes explicitly experimental unless the project scope changes.
