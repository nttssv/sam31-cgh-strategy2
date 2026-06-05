import qupath.lib.regions.RegionRequest
import qupath.lib.roi.GeometryTools
import org.locationtech.jts.geom.Coordinate
import javax.imageio.ImageIO
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Font
import java.awt.RenderingHints
import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage

File outRoot = new File('/Volumes/T9/CGH_PA_annotation_1/training_data/sam31_cgh_p2_24tiles_20260605')
if (outRoot.exists()) {
    throw new IllegalStateException('Output already exists, refusing to overwrite: ' + outRoot.getAbsolutePath())
}

File datasetRoot = new File(outRoot, 'dataset')
File imageDir = new File(datasetRoot, 'images')
File cellMaskDir = new File(datasetRoot, 'cell_instance_masks')
File auxMaskDir = new File(datasetRoot, 'auxiliary_masks')
File semanticMaskDir = new File(datasetRoot, 'semantic_masks')
File previewDir = new File(datasetRoot, 'previews')
File metadataDir = new File(datasetRoot, 'metadata')
File yoloTrainImages = new File(datasetRoot, 'yolo_seg_dataset/images/train')
File yoloValImages = new File(datasetRoot, 'yolo_seg_dataset/images/val')
[outRoot, datasetRoot, imageDir, cellMaskDir, auxMaskDir, semanticMaskDir, previewDir, metadataDir, yoloTrainImages, yoloValImages].each { it.mkdirs() }

String csvEscape(value) {
    String s = value == null ? '' : value.toString()
    return '"' + s.replace('"', '""') + '"'
}
String csvRow(List values) {
    return values.collect { csvEscape(it) }.join(',') + '\n'
}
String safe(value) {
    return value == null ? '' : value.toString().replaceAll('[\\t\\r\\n]+', ' ')
}
String tileIdFromName(String name) {
    def p2 = (name =~ /^P2 tile (\d+)$/)
    if (p2.matches())
        return String.format('p2_tile_%02d', Integer.parseInt(p2[0][1]))
    def yolo = (name =~ /^yolo_tile_(\d+)$/)
    if (yolo.matches())
        return String.format('yolo_tile_%02d', Integer.parseInt(yolo[0][1]))
    return name.toLowerCase().replaceAll('[^a-z0-9]+', '_').replaceAll('^_+|_+$', '')
}
int tileOrder(String name) {
    def p2 = (name =~ /^P2 tile (\d+)$/)
    if (p2.matches())
        return Integer.parseInt(p2[0][1])
    def yolo = (name =~ /^yolo_tile_(\d+)$/)
    if (yolo.matches())
        return Integer.parseInt(yolo[0][1])
    return 9999
}

def imageData = getCurrentImageData()
def server = imageData.getServer()
def objs = getAnnotationObjects()
def clsOf = { obj -> obj.getPathClass() == null ? '' : obj.getPathClass().toString() }
def gf = GeometryTools.getDefaultFactory()
def pointFor = { obj ->
    def r = obj.getROI()
    gf.createPoint(new Coordinate(r.getCentroidX(), r.getCentroidY()))
}
def geomOf = { obj -> GeometryTools.ensurePolygonal(GeometryTools.roiToGeometry(obj.getROI())) }

def positiveClasses = ['GT Clear cell boundary', 'GT Compact cell boundary'] as Set
def uncertainClasses = ['GT Uncertain', 'GT Uncertain cell boundary', 'Ignore for training'] as Set
def valTileIds = ['p2_tile_05', 'p2_tile_10', 'p2_tile_15', 'p2_tile_20'] as Set

def canonicalTiles = objs.findAll {
    clsOf(it) == 'GT Training tile' && it.getROI() != null && ((it.getName() ?: '') ==~ /^P2 tile \d+$/ || (it.getName() ?: '') ==~ /^yolo_tile_\d+$/)
}.sort { a, b ->
    int oa = tileOrder(a.getName() ?: '')
    int ob = tileOrder(b.getName() ?: '')
    oa != ob ? oa <=> ob : ((a.getName() ?: '') <=> (b.getName() ?: ''))
}
if (canonicalTiles.size() != 24) {
    throw new IllegalStateException('Expected 24 canonical tiles, found ' + canonicalTiles.size() + ': ' + canonicalTiles.collect { it.getName() }.join(', '))
}

def sortByPosition = { list ->
    list.sort { a, b ->
        int yc = a.getROI().getCentroidY() <=> b.getROI().getCentroidY()
        yc != 0 ? yc : (a.getROI().getCentroidX() <=> b.getROI().getCentroidX())
    }
}
def colorFor = { obj ->
    switch (clsOf(obj)) {
        case 'GT Nucleus': return new Color(0, 90, 255)
        case 'GT Clear cell boundary': return new Color(0, 210, 80)
        case 'GT Compact cell boundary': return new Color(220, 40, 220)
        case 'GT Uncertain cell boundary': return new Color(255, 165, 0)
        case 'Ignore for training': return new Color(255, 200, 0)
        case 'GT Stroma': return new Color(231, 76, 60)
        default: return Color.WHITE
    }
}

File instanceCsv = new File(metadataDir, 'cell_instances.csv')
File tileCsv = new File(metadataDir, 'dataset_manifest.csv')
File qcCsv = new File(metadataDir, 'boundary_qc.csv')
File objectCsv = new File(metadataDir, 'objects_by_tile.csv')
instanceCsv.text = 'tile_id,tile_name,instance_label,boundary_name,boundary_class,nucleus_name,boundary_area_px,local_centroid_x,local_centroid_y,wsi_centroid_x,wsi_centroid_y,nucleus_wsi_x,nucleus_wsi_y,stroma_overlap_px,stroma_overlap_fraction\n'
tileCsv.text = 'tile_id,tile_name,split,image_file,cell_instance_mask,width,height,trainable_instances,clear_trainable,compact_trainable,edge_or_invalid_boundary_ignore,uncertain_ignore_regions,nuclei_in_tile,stroma_regions_intersecting_tile\n'
qcCsv.text = 'tile_id,tile_name,boundary_name,boundary_class,include_for_training,nuclei_inside,nuclei_inside_in_tile,nuclei_names,stroma_overlap_px,stroma_overlap_fraction,metadata_excluded,flags\n'
objectCsv.text = 'tile_id,tile_name,object_name,class,centroid_x,centroid_y,area_px,bbox_x,bbox_y,bbox_w,bbox_h\n'

def summaryRows = []
int totalTrainable = 0
int totalClear = 0
int totalCompact = 0
int totalNuclei = 0
int totalUncertain = 0
int totalStroma = 0
int totalEdgeInvalid = 0

canonicalTiles.each { tile ->
    String tileName = tile.getName() ?: ''
    String tileId = tileIdFromName(tileName)
    String split = valTileIds.contains(tileId) ? 'val' : 'train'
    def tileRoi = tile.getROI()
    def tileGeom = geomOf(tile)
    def centroidInTile = { obj -> obj.getROI() != null && tileGeom.covers(pointFor(obj)) }
    def intersectsTile = { obj ->
        if (obj.getROI() == null)
            return false
        def g = geomOf(obj)
        return g.intersects(tileGeom) && GeometryTools.ensurePolygonal(g.intersection(tileGeom)).getArea() > 0.0d
    }

    int x = Math.round((float) tileRoi.getBoundsX())
    int y = Math.round((float) tileRoi.getBoundsY())
    int w = Math.round((float) tileRoi.getBoundsWidth())
    int h = Math.round((float) tileRoi.getBoundsHeight())
    if (w <= 0 || h <= 0)
        throw new IllegalArgumentException('Invalid tile size for ' + tileName)

    def raw = server.readRegion(RegionRequest.createInstance(server.getPath(), 1.0, x, y, w, h))
    File imageFile = new File(imageDir, tileId + '.png')
    ImageIO.write(raw, 'PNG', imageFile)
    File splitImage = new File(split == 'val' ? yoloValImages : yoloTrainImages, imageFile.getName())
    ImageIO.write(raw, 'PNG', splitImage)

    def tx = AffineTransform.getTranslateInstance(-x, -y)
    def localShape = { geom -> tx.createTransformedShape(GeometryTools.geometryToShape(geom)) }
    def clipGeom = { obj -> GeometryTools.ensurePolygonal(geomOf(obj).intersection(tileGeom)) }

    def drawMask = { File file, List annos, boolean instanceLabels ->
        def mask = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
        def g2 = mask.createGraphics()
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF)
        int label = 1
        annos.each { obj ->
            def geom = clipGeom(obj)
            if (!geom.isEmpty()) {
                int v = instanceLabels ? label : 255
                v = Math.max(0, Math.min(255, v))
                g2.setColor(new Color(v, v, v))
                g2.fill(localShape(geom))
            }
            label++
        }
        g2.dispose()
        ImageIO.write(mask, 'PNG', file)
    }
    def drawSemantic = { File file, Map classValues ->
        def mask = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
        def g2 = mask.createGraphics()
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF)
        classValues.each { cls, row ->
            int v = row.value as int
            row.objects.each { obj ->
                def geom = clipGeom(obj)
                if (!geom.isEmpty()) {
                    g2.setColor(new Color(v, v, v))
                    g2.fill(localShape(geom))
                }
            }
        }
        g2.dispose()
        ImageIO.write(mask, 'PNG', file)
    }

    def nuclei = sortByPosition(objs.findAll { clsOf(it) == 'GT Nucleus' && it.getROI() != null && centroidInTile(it) })
    def stromas = sortByPosition(objs.findAll { clsOf(it) == 'GT Stroma' && it.getROI() != null && intersectsTile(it) })
    def uncertain = sortByPosition(objs.findAll { uncertainClasses.contains(clsOf(it)) && it.getROI() != null && intersectsTile(it) })
    def boundariesAll = sortByPosition(objs.findAll { positiveClasses.contains(clsOf(it)) && it.getROI() != null && centroidInTile(it) })
    def nucleusInsideStroma = { n -> stromas.any { s -> geomOf(s).covers(pointFor(n)) } }

    (nuclei + stromas + uncertain + boundariesAll).each { obj ->
        def roi = obj.getROI()
        double area = 0.0d
        try {
            area = geomOf(obj).intersection(tileGeom).getArea()
        } catch (Exception ignored) {}
        objectCsv << [
                tileId, tileName, safe(obj.getName()), clsOf(obj),
                String.format('%.2f', roi.getCentroidX()), String.format('%.2f', roi.getCentroidY()),
                String.format('%.2f', area),
                String.format('%.2f', roi.getBoundsX()), String.format('%.2f', roi.getBoundsY()),
                String.format('%.2f', roi.getBoundsWidth()), String.format('%.2f', roi.getBoundsHeight())
        ].join(',') + '\n'
    }

    def validRows = []
    def validBoundaries = []
    def edgeOrInvalid = []
    boundariesAll.each { b ->
        def bg = geomOf(b)
        def contained = nuclei.findAll { n -> bg.covers(pointFor(n)) }
        def containedInTile = contained.findAll { n -> centroidInTile(n) }
        def containedInStroma = contained.findAll { n -> nucleusInsideStroma(n) }
        double stromaOverlap = 0.0d
        stromas.each { s ->
            def sg = geomOf(s)
            if (bg.intersects(sg))
                stromaOverlap += GeometryTools.ensurePolygonal(bg.intersection(sg)).getArea()
        }
        double area = GeometryTools.ensurePolygonal(bg.intersection(tileGeom)).getArea()
        double frac = area > 0 ? stromaOverlap / area : 0.0d
        def flags = []
        if (contained.size() == 0) flags << 'ZERO_NUCLEUS'
        if (contained.size() > 1) flags << 'MULTI_NUCLEI'
        if (contained.size() == 1 && containedInTile.size() == 0) flags << 'NUCLEUS_OUTSIDE_TILE'
        if (!containedInStroma.isEmpty()) flags << 'NUCLEUS_INSIDE_STROMA'
        if (frac > 0.02) flags << 'STROMA_OVERLAP'
        boolean metadataExcluded = (b.getMetadata().get('exclude_from_training_export') ?: '').toString().equalsIgnoreCase('true')
        boolean include = !metadataExcluded && contained.size() == 1 && containedInTile.size() == 1 && containedInStroma.isEmpty()
        if (include) {
            validBoundaries << b
            validRows << [boundary: b, nucleus: containedInTile[0], area: area, overlap: stromaOverlap, frac: frac, flags: flags]
        } else {
            edgeOrInvalid << b
        }
        qcCsv << csvRow([
                tileId, tileName, b.getName(), clsOf(b), include, contained.size(), containedInTile.size(),
                contained.collect { it.getName() ?: '' }.join('|'), String.format('%.1f', stromaOverlap),
                String.format('%.4f', frac), metadataExcluded, flags.isEmpty() ? 'OK' : flags.join('|')
        ])
    }

    File cellMaskFile = new File(cellMaskDir, tileId + '.png')
    drawMask(cellMaskFile, validBoundaries, true)
    drawMask(new File(auxMaskDir, tileId + '_gt_nucleus_instances.png'), nuclei, true)
    drawMask(new File(auxMaskDir, tileId + '_gt_nucleus_all_direct_children.png'), nuclei, true)
    drawMask(new File(auxMaskDir, tileId + '_gt_stroma.png'), stromas, false)
    drawMask(new File(auxMaskDir, tileId + '_gt_uncertain_ignore.png'), uncertain, false)
    drawMask(new File(auxMaskDir, tileId + '_edge_or_invalid_cell_ignore.png'), edgeOrInvalid, false)
    drawMask(new File(auxMaskDir, tileId + '_gt_clear_boundary_all.png'), boundariesAll.findAll { clsOf(it) == 'GT Clear cell boundary' }, false)
    drawMask(new File(auxMaskDir, tileId + '_gt_compact_boundary_all.png'), boundariesAll.findAll { clsOf(it) == 'GT Compact cell boundary' }, false)
    drawSemantic(new File(semanticMaskDir, tileId + '_semantic_review.png'), [
            stroma: [value: 4, objects: stromas],
            uncertain: [value: 3, objects: uncertain],
            clear: [value: 1, objects: boundariesAll.findAll { clsOf(it) == 'GT Clear cell boundary' }],
            compact: [value: 2, objects: boundariesAll.findAll { clsOf(it) == 'GT Compact cell boundary' }]
    ])

    File perTileCsv = new File(metadataDir, tileId + '_instances.csv')
    perTileCsv.text = 'instance_label,boundary_name,boundary_class,nucleus_name,boundary_area_px,local_centroid_x,local_centroid_y,wsi_centroid_x,wsi_centroid_y,nucleus_wsi_x,nucleus_wsi_y,stroma_overlap_px,stroma_overlap_fraction\n'
    validRows.eachWithIndex { row, idx ->
        def b = row.boundary
        def n = row.nucleus
        int label = idx + 1
        double localCx = b.getROI().getCentroidX() - x
        double localCy = b.getROI().getCentroidY() - y
        def values = [
                label, b.getName(), clsOf(b), n.getName(), String.format('%.1f', row.area),
                String.format('%.1f', localCx), String.format('%.1f', localCy),
                String.format('%.1f', b.getROI().getCentroidX()), String.format('%.1f', b.getROI().getCentroidY()),
                String.format('%.1f', n.getROI().getCentroidX()), String.format('%.1f', n.getROI().getCentroidY()),
                String.format('%.1f', row.overlap), String.format('%.4f', row.frac)
        ]
        perTileCsv << csvRow(values)
        instanceCsv << csvRow([tileId, tileName] + values)
    }

    int clearTrainable = validRows.count { clsOf(it.boundary) == 'GT Clear cell boundary' }
    int compactTrainable = validRows.count { clsOf(it.boundary) == 'GT Compact cell boundary' }
    tileCsv << csvRow([
            tileId, tileName, split, 'dataset/images/' + imageFile.getName(),
            'dataset/cell_instance_masks/' + cellMaskFile.getName(), w, h, validRows.size(),
            clearTrainable, compactTrainable, edgeOrInvalid.size(), uncertain.size(), nuclei.size(), stromas.size()
    ])

    def preview = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    def g = preview.createGraphics()
    g.drawImage(raw, 0, 0, null)
    g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    g.setFont(new Font('SansSerif', Font.BOLD, 10))
    def overlayOrdered = []
    overlayOrdered.addAll(stromas)
    overlayOrdered.addAll(uncertain)
    overlayOrdered.addAll(edgeOrInvalid)
    overlayOrdered.addAll(boundariesAll)
    overlayOrdered.addAll(nuclei)
    overlayOrdered.each { obj ->
        def geom = clipGeom(obj)
        if (geom.isEmpty())
            return
        def c = colorFor(obj)
        boolean isInvalid = edgeOrInvalid.contains(obj)
        g.setColor(new Color(c.getRed(), c.getGreen(), c.getBlue(), clsOf(obj) == 'GT Stroma' ? 45 : 35))
        if (clsOf(obj) != 'GT Nucleus')
            g.fill(localShape(geom))
        g.setStroke(new BasicStroke(isInvalid ? 4.0f : (clsOf(obj) == 'GT Nucleus' ? 2.0f : 2.5f)))
        g.setColor(isInvalid ? Color.RED : c)
        g.draw(localShape(geom))
        if (clsOf(obj) != 'GT Stroma') {
            int lx = Math.round((float) (obj.getROI().getCentroidX() - x))
            int ly = Math.round((float) (obj.getROI().getCentroidY() - y))
            String label = obj.getName() ?: clsOf(obj)
            if (isInvalid)
                label += ' [IGNORE]'
            int boxW = Math.min(w - Math.max(lx + 3, 0), Math.min(190, label.length() * 7 + 8))
            if (boxW > 10) {
                g.setColor(new Color(255, 255, 255, 210))
                g.fillRect(Math.max(lx + 3, 0), Math.max(ly - 12, 0), boxW, 14)
                g.setColor(isInvalid ? Color.RED : Color.BLACK)
                g.drawString(label, Math.max(lx + 6, 0), Math.max(ly - 2, 10))
            }
        }
    }
    g.dispose()
    File previewFile = new File(previewDir, tileId + '_overlay.png')
    ImageIO.write(preview, 'PNG', previewFile)

    summaryRows << [
            tileId: tileId, tileName: tileName, split: split, width: w, height: h,
            trainable: validRows.size(), clear: clearTrainable, compact: compactTrainable,
            invalid: edgeOrInvalid.size(), uncertain: uncertain.size(), nuclei: nuclei.size(),
            stroma: stromas.size(), preview: previewFile.getAbsolutePath()
    ]
    totalTrainable += validRows.size()
    totalClear += clearTrainable
    totalCompact += compactTrainable
    totalNuclei += nuclei.size()
    totalUncertain += uncertain.size()
    totalStroma += stromas.size()
    totalEdgeInvalid += edgeOrInvalid.size()
}

File qcMd = new File(metadataDir, 'DATASET_QC.md')
qcMd.text = '# SAM31 CGH 24-tile dataset QC\n\n'
qcMd << "Source project: /Volumes/T9/CGH_PA_annotation_1/project.qpproj\n\n"
qcMd << "Source image: ${server.getMetadata().getName()} (${server.getWidth()}x${server.getHeight()})\n\n"
qcMd << "Export root: ${outRoot.getAbsolutePath()}\n\n"
qcMd << "Canonical tiles: ${summaryRows.size()}\n\n"
qcMd << "Trainable cell boundaries: ${totalTrainable} (${totalClear} clear, ${totalCompact} compact)\n\n"
qcMd << "Nuclei: ${totalNuclei}; stroma regions intersecting tiles: ${totalStroma}; uncertain/ignore regions: ${totalUncertain}; invalid positive boundaries masked out: ${totalEdgeInvalid}\n\n"
qcMd << '| tile_id | split | size | trainable | clear | compact | invalid/ignore | uncertain | nuclei | stroma |\n'
qcMd << '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
summaryRows.each { row ->
    qcMd << "| ${row.tileId} | ${row.split} | ${row.width}x${row.height} | ${row.trainable} | ${row.clear} | ${row.compact} | ${row.invalid} | ${row.uncertain} | ${row.nuclei} | ${row.stroma} |\n"
}

File readme = new File(outRoot, 'README.md')
readme.text = """# SAM31 CGH P2 24-tile training dataset

This dataset was exported from the current QuPath project using centroid-in-tile object collection over the full hierarchy.
It intentionally avoids duplicate helper training tiles such as `cell_boundary_clean` and `nuclei_clean`.

## Split

- train: all canonical tiles except `p2_tile_05`, `p2_tile_10`, `p2_tile_15`, `p2_tile_20`
- val/test: `p2_tile_05`, `p2_tile_10`, `p2_tile_15`, `p2_tile_20`

## Main folders

- `dataset/images`: raw image tiles
- `dataset/cell_instance_masks`: trainable clear/compact cell boundary instance masks
- `dataset/auxiliary_masks`: nucleus, stroma, uncertain/ignore, and QC masks
- `dataset/metadata/cell_instances.csv`: instance label to clear/compact class mapping
- `dataset/metadata/DATASET_QC.md`: per-tile export QC

Run `python prepare_sam31_dataset.py` from this folder to regenerate SAM3 COCO JSON.
"""

println 'SAM31_EXPORT_DONE'
println 'out=' + outRoot.getAbsolutePath()
println 'tiles=' + summaryRows.size()
println 'train_tiles=' + summaryRows.count { it.split == 'train' }
println 'val_tiles=' + summaryRows.count { it.split == 'val' }
println 'trainable_cell_boundaries=' + totalTrainable
println 'clear=' + totalClear
println 'compact=' + totalCompact
println 'nuclei=' + totalNuclei
println 'uncertain_or_ignore_regions=' + totalUncertain
println 'invalid_positive_boundaries_masked=' + totalEdgeInvalid
