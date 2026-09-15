# 图片批量收集工具

目的：把一个文件夹（**递归到各级子文件夹**）里的图片全部收拢到一个目标文件夹里，目标文件夹不存在就自动创建。

典型场景：网盘下载下来的一批素材，`BaiduNetdiskDownload` 下有一级子文件夹、一级里又套着二级子文件夹，图片散在最里层。收拢后便于统一处理（推理、拆数据集、批量改名等）。

```
BaiduNetdiskDownload\2023新课标\第一章\1.jpg  ┐
BaiduNetdiskDownload\2023新课标\第二章\1.jpg  ├─►  全部图片\1.jpg
BaiduNetdiskDownload\2024新课标\第一章\a.bmp  ┘      全部图片\1-1.jpg
                                                      全部图片\a.bmp
```

## 运行环境

只用 Python 标准库 + 本项目的 `config.py`（取图片后缀），**不需要任何第三方依赖**。
在本项目里用 Anaconda 环境 `pcc` 即可（`conda activate pcc`）。

## 使用方法

1）**先预览**（推荐先跑一次，确认要收哪些文件、重名的会改成什么名字）：

```
cd tool\collect_images
python collect_images.py --dir "E:\baiduwangpan\DownLoad\BaiduNetdiskDownload" --out "D:\数据\全部图片" --dry-run
```

预览模式**不写任何文件**，连目标文件夹都不会创建。

2）**正式收集**（默认复制，源文件保留）：

```
python collect_images.py --dir "E:\baiduwangpan\DownLoad\BaiduNetdiskDownload" --out "D:\数据\全部图片"
```

3）**确认无误后要清空源文件夹**，改成移动（源里的空目录会留下，工具不删空壳）：

```
python collect_images.py --dir "E:\baiduwangpan\DownLoad\BaiduNetdiskDownload" --out "D:\数据\全部图片" --move
```

4）**连标注一起收**（如 `.rsml`），或只收指定后缀：

```
python collect_images.py --dir "D:\数据\某批次" --out "D:\数据\全部图片" --exts .png,.rsml
```

## 参数

| 参数 | 说明 |
| --- | --- |
| `--dir` | 源文件夹，必填。会递归扫到所有层级的子文件夹 |
| `--out` | 目标文件夹，必填。不存在则创建；已存在则与新文件合并 |
| `--exts` | 要收的后缀，逗号分隔，默认 `config.IMAGE_EXTS`（`.png .jpg .jpeg .bmp .tif .tiff`） |
| `--move` | 移动而不是复制（默认复制，源文件保留） |
| `--dry-run` | 只预览，不写任何文件 |

## 重名怎么处理

不同子文件夹里的同名文件，按项目规范追加 `-1`、`-2` …（与 `common/naming.unique_path` 同一套规则）。
**磁盘上已有的同名文件一律不覆盖**——所以同一个源反复跑不会毁数据，只是新收进来的那批会带 `-1`、`-2` 后缀。

因为改名后只看文件名已经认不出出处，每次运行都会在目标文件夹写一份 `collect.txt`，逐条记录对应关系：

```
1.jpg <- 2023新课标\第一章\1.jpg
1-1.jpg <- 2023新课标\第二章\1.jpg  (重名改名)
```

目标文件夹里已有 `collect.txt` 时，新的记录写成 `collect-1.txt`、`collect-2.txt`…，不覆盖旧记录。

## 注意事项

- **目标文件夹若就在源文件夹里面**，工具会自动把它排除，不会把上一次收进去的文件又当成源文件收一遍（越收越多）。
- `Thumbs.db`、`.DS_Store`、`desktop.ini` 这类系统文件一律跳过。
- 单个文件复制失败（占用、权限）不会中断整批，失败清单写进 `collect.txt` 末尾，进程退出码为 1。
- 源文件夹里除图片外的其他文件（`.txt`、`.rsml` 等）默认不动，需要时用 `--exts` 指定。
