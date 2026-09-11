"""把预测出的根系折线导出为与标注同格式的 RSML 文件。

格式与 datasets 下的标注一致（节选结构）：
    <rsml ...>
      <metadata>version/unit/last-modified/software/file-key ...</metadata>
      <scene>
        <plant ID="1" label="sugarcane">
          <annotations>annotation</annotations>
          <root ID="1.1" label="primary">
            <geometry>
              <rootnavspline controlpointseparation="50" tension="0.5">
                <point x="1638" y="786" />
                ...
"""
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

_NS = {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
       "xmlns:xsd": "http://www.w3.org/2001/XMLSchema"}


def write_rsml(path, file_key: str, polylines, software: str = "U-Net inference",
               user: str = "PCC", plant_label: str = "sugarcane",
               controlpoint_separation: int = 50, tension: str = "0.5") -> Path:
    """把若干折线写成 RSML 文件（每条折线 = 一个 plant 下的一条 primary 根）。

    polylines: [[(x, y), ...], ...]，来自 skeleton_stats.extract_root_paths。
    返回写入的路径。
    """
    rsml = ET.Element("rsml", _NS)
    md = ET.SubElement(rsml, "metadata")
    ET.SubElement(md, "version").text = "1.0"
    ET.SubElement(md, "unit").text = "pixel"
    ET.SubElement(md, "resolution").text = "xxx dpi"
    ET.SubElement(md, "last-modified").text = datetime.now().strftime(
        "%Y/%m/%d %H:%M:%S")
    ET.SubElement(md, "software").text = software
    ET.SubElement(md, "user").text = user
    ET.SubElement(md, "file-key").text = file_key

    scene = ET.SubElement(rsml, "scene")
    for i, pts in enumerate(polylines, 1):
        if len(pts) < 2:
            continue  # 少于 2 个点无法构成折线
        plant = ET.SubElement(scene, "plant", {"ID": str(i), "label": plant_label})
        ET.SubElement(plant, "annotations").text = "annotation"
        root = ET.SubElement(plant, "root", {"ID": f"{i}.1", "label": "primary"})
        geo = ET.SubElement(root, "geometry")
        spline = ET.SubElement(geo, "rootnavspline", {
            "controlpointseparation": str(controlpoint_separation),
            "tension": tension})
        for (x, y) in pts:
            ET.SubElement(spline, "point", {"x": str(int(x)), "y": str(int(y))})

    ET.indent(rsml, space="  ")
    path = Path(path)
    ET.ElementTree(rsml).write(path, encoding="utf-8", xml_declaration=True)
    return path
