# GDAL 配置说明

核心模块和内存示例完全不需要 GDAL。只有调用 `cnn4rs-geospatial` 的 `GdalRasterReader` 或运行 GDAL 示例时，才需要在运行环境提供 GDAL Java 绑定及其匹配的原生库。

## 配置步骤

1. 通过操作系统或组织认可的渠道安装 GDAL，并确认命令行工具能够读取目标栅格。
2. 将与该 GDAL 版本匹配的 Java 绑定加入应用运行时 classpath。
3. 让 JVM 能找到相同版本的 GDAL 原生库；具体路径设置取决于操作系统、JVM 和部署方式。
4. 使用小型、可公开的测试栅格先验证读操作，再接入实际分类流程。

配置完成后，可构建项目并运行带有两个显式路径参数的接线示例：

```bash
mvn -B -ntp -pl examples -am package
java -cp examples/target/classes:cnn4rs-core/target/classes:cnn4rs-geospatial/target/classes io.github.chjers.cnn4rs.examples.GdalClassificationExample input.tif prediction.txt
```

示例读取栅格的前两个值，使用固定种子的演示分类器，并将 `predictedClass=<integer>` 写入输出文本。它不是训练好的遥感模型，也不写出 GeoTIFF。

## 常见问题

- `GdalUnavailableException`：Java 绑定不存在、原生库不可加载，或两者版本不匹配。请核对 classpath、JVM 的原生库搜索路径和 GDAL 版本。
- 无法打开输入文件：确认输入路径、文件权限和 GDAL 驱动支持；读取器始终以只读方式打开数据集。
- CI 不安装 GDAL：持续集成只验证纯 Java 的编译与测试边界，不代替真实部署环境的 GDAL 验证。
