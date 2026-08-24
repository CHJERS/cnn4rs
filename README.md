# cnn4rs

`cnn4rs` 是面向遥感影像分类研究与 CNN 原理学习的 Java 17 类库。它将一个 2020 年研究项目整理为可测试的 Maven 多模块工程：核心张量和模型不依赖原生库，GDAL 与分布式实验能力保持显式、可选且有边界。

它不是通用深度学习框架、生产级遥感处理管线，亦不宣称提供生产级的分布式训练。`cnn4rs-distributed` 是实验性模块；请先评估其适用性与安全边界。

## 模块

| 模块 | 用途 | 外部运行时要求 |
| --- | --- | --- |
| `cnn4rs-core` | 不可变张量、CNN 层、预测、训练与 checkpoint | 仅 JDK |
| `cnn4rs-geospatial` | 栅格读写边界与可选 GDAL 适配 | GDAL 仅在实际调用适配器时需要 |
| `cnn4rs-distributed` | 实验性的训练任务和梯度聚合值对象 | 仅 JDK；不包含传输实现 |
| `examples` | 可执行的内存与 GDAL 接线示例 | 内存示例仅 JDK；GDAL 示例见下文 |
| `legacy` | 2020 年历史实现，仅供溯源参考 | 不参加 Maven 构建，也不是公共 API |

## 快速开始

前提：JDK 17 与 Maven。核心示例不读取文件、不需要 GDAL，也不依赖本机路径：

```bash
git clone https://github.com/chjers/cnn4rs.git
cd cnn4rs
mvn -B -ntp -pl examples -am package
java -cp examples/target/classes:cnn4rs-core/target/classes io.github.chjers.cnn4rs.examples.InMemoryClassifierExample
```

第二条命令直接启动 `InMemoryClassifierExample`；它只使用 core 类路径，不读取文件，也不加载 GDAL。输出为：

```text
predictedClass=0
```

完整回归测试：

```bash
mvn -B -ntp test
```

需要读取实际栅格时，请先阅读 [GDAL 配置说明](docs/gdal-setup.md)。GDAL 示例接收输入栅格和输出文本两个参数；它用于演示可选适配器与确定性模型的接线，并非已训练的遥感分类器。

## English quick start

Use JDK 17 and Maven. The in-memory example is deterministic, core-only, and requires neither GDAL nor local data paths:

```bash
mvn -B -ntp -pl examples -am package
java -cp examples/target/classes:cnn4rs-core/target/classes io.github.chjers.cnn4rs.examples.InMemoryClassifierExample
```

The second command launches the example directly and prints `predictedClass=0`. Run `mvn -B -ntp test` for the complete reactor test suite. See [GDAL setup](docs/gdal-setup.md) before running the optional raster example.

## 架构与迁移

- [架构说明](docs/architecture.md)：模块边界、数据流和实验性功能范围。
- [从历史实现迁移](docs/migration-from-legacy.md)：旧类名到公共 API 的对应关系与限制。
- [GDAL 配置说明](docs/gdal-setup.md)：可选原生依赖的配置与故障诊断。

历史 `legacy/` 目录保持为参考材料，未纳入新 API，也不承诺其未经测试的历史数值行为。特别是，历史代码中的本机路径、原始 `HashMap` 配置与并发假设不应迁移到新代码中。

## 论文来源与权利说明

本项目的研究背景材料为黄嘉俊（2020）的硕士学位论文《基于分布式 Java 环境的卷积神经网络类库实现——以深圳市高分二号卫星影像数据分类为例》，原始 PDF 位于 [`docs/`](docs/) 且未被修改。其学位论文使用授权页载明：

> 研究生在校攻读学位期间论文工作的知识产权单位属深圳大学。学校有权保留学位论文并向国家主管部门或其他机构送交论文的电子版和纸质版，允许论文被查阅和借阅。本人授权深圳大学可以将学位论文的全部或部分内容编入有关数据库进行检索，可以采用影印、缩印或扫描等复制手段保存、汇编学位论文。

请在复用论文或其内容前自行确认适用的授权条件。完整书目信息与出处见[引用说明](docs/citation.md)。

## 参与与安全

项目采用 [MIT License](LICENSE)。提交改动前请阅读[贡献指南](CONTRIBUTING.md)并运行测试；安全问题请遵循 [Security Policy](SECURITY.md) 的私下披露流程。
