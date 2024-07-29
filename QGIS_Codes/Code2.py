import os

from qgis.core import (
    QgsGeometry,
    QgsMapSettings,
    QgsPrintLayout,
    QgsMapSettings,
    QgsMapRendererParallelJob,
    QgsLayoutItemLabel,
    QgsLayoutItemLegend,
    QgsLayoutItemMap,
    QgsLayoutItemPolygon,
    QgsLayoutItemScaleBar,
    QgsLayoutExporter,
    QgsLayoutItem,
    QgsLayoutPoint,
    QgsLayoutSize,
    QgsUnitTypes,
    QgsProject,
    QgsFillSymbol,
)

from qgis.PyQt.QtGui import (
    QPolygonF,
    QColor,
)

from qgis.PyQt.QtCore import (
    QPointF,
    QRectF,
    QSize,
)
project = QgsProject.instance()
layout = QgsPrintLayout(project)
layout.initializeDefaults()

map = QgsLayoutItemMap(layout)
# Provide an extent to render
map.zoomToExtent(iface.mapCanvas().extent())
layout.addLayoutItem(map)

base_path = os.path.join(QgsProject.instance().homePath())
pdf_path = os.path.join(base_path, "output.png")

exporter = QgsLayoutExporter(layout)
exporter.exportToImage(pdf_path, QgsLayoutExporter.ImageExportSettings())

layout.setName("MyLayout")
project.layoutManager().addLayout(layout)
