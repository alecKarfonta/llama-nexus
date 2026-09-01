import io
import zipfile

from modules.graphrag.document_pipeline import (
    DocumentElement,
    GraphCandidateRelationship,
    _build_chunks,
    _build_local_graph_artifact,
    _build_local_graph_reasoning_index,
    _candidate_entities,
    _candidate_relationships,
    _chunk_graph_context,
    _chunk_evidence_cards,
    _embedding_evidence_cards,
    _embedding_input_sample,
    _extract_csv,
    _extract_pptx,
    _relationship_sample_category,
)


def _minimal_rich_pptx() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr(
            "ppt/presentation.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
    xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:sldSz cx="9144000" cy="5143500"/>
  <p:sldIdLst>
    <p:sldId id="256" r:id="rId1"/>
  </p:sldIdLst>
</p:presentation>""",
        )
        package.writestr(
            "ppt/_rels/presentation.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
    Target="slides/slide1.xml"/>
</Relationships>""",
        )
        package.writestr(
            "ppt/slides/slide1.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
    xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
    xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      <p:sp>
        <p:nvSpPr><p:cNvPr id="2" name="Title 1" title="Launch Approval Packet"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
        <p:spPr><a:xfrm><a:off x="457200" y="274320"/><a:ext cx="8229600" cy="685800"/></a:xfrm></p:spPr>
        <p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr b="1"/><a:t>Launch Approval Packet</a:t></a:r></a:p></p:txBody>
      </p:sp>
      <p:grpSp>
        <p:nvGrpSpPr><p:cNvPr id="20" name="Approval dependency cluster" title="Approval Dependency Cluster" descr="Grouped approval gate and launch workflow visuals"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
        <p:grpSpPr><a:xfrm><a:off x="548640" y="1097280"/><a:ext cx="4937760" cy="1097280"/></a:xfrm></p:grpSpPr>
        <p:sp>
          <p:nvSpPr><p:cNvPr id="3" name="Gate narrative"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
          <p:spPr><a:xfrm><a:off x="685800" y="1188720"/><a:ext cx="3657600" cy="914400"/></a:xfrm></p:spPr>
          <p:txBody>
            <a:bodyPr/><a:lstStyle/>
            <a:p><a:pPr lvl="0"/><a:r><a:rPr b="1" sz="2400"><a:solidFill><a:srgbClr val="00A884"/></a:solidFill><a:highlight><a:srgbClr val="FFF2CC"/></a:highlight></a:rPr><a:t>Approval required</a:t></a:r><a:r><a:t> before launch workflow.</a:t></a:r></a:p>
          </p:txBody>
        </p:sp>
        <p:sp>
          <p:nvSpPr><p:cNvPr id="4" name="Launch workflow"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
          <p:spPr><a:xfrm><a:off x="5029200" y="1188720"/><a:ext cx="2743200" cy="914400"/></a:xfrm></p:spPr>
          <p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>Launch workflow</a:t></a:r></a:p></p:txBody>
        </p:sp>
        <p:cxnSp>
          <p:nvCxnSpPr><p:cNvPr id="5" name="approval connector" title="gates"/><p:cNvCxnSpPr><a:stCxn id="3" idx="0"/><a:endCxn id="4" idx="0"/></p:cNvCxnSpPr><p:nvPr/></p:nvCxnSpPr>
          <p:spPr><a:xfrm><a:off x="4114800" y="1463040"/><a:ext cx="914400" cy="0"/></a:xfrm></p:spPr>
        </p:cxnSp>
      </p:grpSp>
      <p:graphicFrame>
        <p:nvGraphicFramePr><p:cNvPr id="6" name="Approval table"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>
        <p:xfrm><a:off x="685800" y="2438400"/><a:ext cx="3657600" cy="914400"/></p:xfrm>
        <a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">
          <a:tbl><a:tblPr/><a:tblGrid><a:gridCol w="1828800"/><a:gridCol w="1828800"/></a:tblGrid>
            <a:tr h="457200"><a:tc><a:txBody><a:bodyPr/><a:p><a:r><a:t>Source</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:bodyPr/><a:p><a:r><a:t>Relation</a:t></a:r></a:p></a:txBody></a:tc></a:tr>
            <a:tr h="457200"><a:tc><a:txBody><a:bodyPr/><a:p><a:r><a:t>Approval required</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:bodyPr/><a:p><a:r><a:t>gates Launch workflow</a:t></a:r></a:p></a:txBody></a:tc></a:tr>
          </a:tbl>
        </a:graphicData></a:graphic>
      </p:graphicFrame>
      <p:graphicFrame>
        <p:nvGraphicFramePr><p:cNvPr id="7" name="Revenue chart"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>
        <p:xfrm><a:off x="5029200" y="2438400"/><a:ext cx="3200400" cy="1600200"/></p:xfrm>
        <a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart"><c:chart r:id="rIdChart1"/></a:graphicData></a:graphic>
      </p:graphicFrame>
      <p:pic>
        <p:nvPicPr><p:cNvPr id="8" name="Risk Map Image" title="Risk Map" descr="Risk map highlights regional launch blockers"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
        <p:blipFill><a:blip r:embed="rIdImage1"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
        <p:spPr><a:xfrm><a:off x="5029200" y="4267200"/><a:ext cx="1828800" cy="457200"/></a:xfrm></p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:controls>
    <p:control r:id="rIdControl1" name="Approval checkbox" spid="8"/>
  </p:controls>
</p:sld>""",
        )
        package.writestr(
            "ppt/slides/_rels/slide1.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdChart1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart"
    Target="../charts/chart1.xml"/>
  <Relationship Id="rIdComments1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
    Target="../comments/comment1.xml"/>
  <Relationship Id="rIdNotes1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"
    Target="../notesSlides/notesSlide1.xml"/>
  <Relationship Id="rIdMedia1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/audio"
    Target="../media/briefing.mp3"/>
  <Relationship Id="rIdImage1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    Target="../media/risk-map.png"/>
  <Relationship Id="rIdPackage1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/package"
    Target="../embeddings/approval.csv"/>
  <Relationship Id="rIdControl1"
    Type="http://schemas.microsoft.com/office/2006/relationships/ctrlProp"
    Target="../ctrlProps/ctrlProp1.xml"/>
</Relationships>""",
        )
        package.writestr(
            "ppt/commentAuthors.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<p:cmAuthorLst xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cmAuthor id="0" name="Maya Reviewer" initials="MR"/>
</p:cmAuthorLst>""",
        )
        package.writestr(
            "ppt/comments/comment1.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<p:cmLst xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
    xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cm authorId="0" dt="2026-07-22T00:00:00Z" idx="1">
    <p:pos x="7772400" y="822960"/>
    <p:text><a:r><a:t>Finance needs approval evidence before launch.</a:t></a:r></p:text>
  </p:cm>
</p:cmLst>""",
        )
        package.writestr(
            "ppt/notesSlides/notesSlide1.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
    xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld><p:spTree>
    <p:sp><p:txBody><a:bodyPr/><a:p><a:r><a:t>Speaker note: cite the Finance Review Board approval evidence.</a:t></a:r></a:p></p:txBody></p:sp>
  </p:spTree></p:cSld>
</p:notes>""",
        )
        package.writestr(
            "ppt/ctrlProps/ctrlProp1.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<ax:ocx xmlns:ax="http://schemas.microsoft.com/office/2006/activeX"
    progId="Forms.CheckBox.1" name="Approval checkbox">
  <ax:ocxPr name="Caption" value="Approval required"/>
  <ax:ocxPr name="Value" value="1"/>
  <ax:ocxPr name="Checked" value="true"/>
</ax:ocx>""",
        )
        package.writestr("ppt/media/briefing.mp3", b"ID3briefing-audio-evidence")
        package.writestr(
            "ppt/media/risk-map.png",
            bytes.fromhex(
                "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                "1f15c4890000000a49444154789c6360000002000100e221bc3300000000"
                "49454e44ae426082"
            ),
        )
        package.writestr(
            "ppt/embeddings/approval.csv",
            "Owner,Decision\nFinance Review Board,Approved\n",
        )
        package.writestr(
            "ppt/charts/chart1.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"
    xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <c:chart>
    <c:title><c:tx><c:rich><a:bodyPr/><a:p><a:r><a:t>Revenue Plan</a:t></a:r></a:p></c:rich></c:tx></c:title>
    <c:plotArea>
      <c:barChart>
        <c:ser>
          <c:idx val="0"/><c:order val="0"/>
          <c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>Approval</c:v></c:pt></c:strCache></c:strRef></c:tx>
          <c:spPr>
            <a:solidFill><a:srgbClr val="00A884"/></a:solidFill>
            <a:ln w="19050"><a:solidFill><a:srgbClr val="123456"/></a:solidFill><a:prstDash val="dash"/></a:ln>
          </c:spPr>
          <c:marker>
            <c:symbol val="diamond"/><c:size val="7"/>
            <c:spPr><a:solidFill><a:srgbClr val="C00000"/></a:solidFill></c:spPr>
          </c:marker>
          <c:dPt><c:idx val="1"/><c:spPr><a:solidFill><a:srgbClr val="FFB000"/></a:solidFill></c:spPr></c:dPt>
          <c:cat><c:strRef><c:strCache><c:pt idx="0"><c:v>Q2</c:v></c:pt><c:pt idx="1"><c:v>Q3</c:v></c:pt></c:strCache></c:strRef></c:cat>
          <c:val><c:numRef><c:numCache><c:pt idx="0"><c:v>18</c:v></c:pt><c:pt idx="1"><c:v>31</c:v></c:pt></c:numCache></c:numRef></c:val>
          <c:dLbls>
            <c:showVal val="1"/><c:showCatName val="1"/><c:dLblPos val="outEnd"/>
            <c:numFmt formatCode="$#,##0"/>
          </c:dLbls>
          <c:trendline>
            <c:name val="approval velocity"/><c:trendlineType val="linear"/>
            <c:dispEq val="1"/><c:dispRSqr val="1"/>
            <c:spPr><a:ln w="12700"><a:solidFill><a:srgbClr val="7030A0"/></a:solidFill><a:prstDash val="sysDot"/></a:ln></c:spPr>
          </c:trendline>
        </c:ser>
      </c:barChart>
      <c:catAx>
        <c:axId val="10"/><c:axPos val="b"/>
        <c:title><c:tx><c:rich><a:bodyPr/><a:p><a:r><a:t>Quarter</a:t></a:r></a:p></c:rich></c:tx></c:title>
      </c:catAx>
      <c:valAx>
        <c:axId val="20"/><c:axPos val="l"/>
        <c:title><c:tx><c:rich><a:bodyPr/><a:p><a:r><a:t>Revenue impact</a:t></a:r></a:p></c:rich></c:tx></c:title>
        <c:numFmt formatCode="$#,##0"/>
      </c:valAx>
    </c:plotArea>
    <c:legend><c:legendPos val="r"/><c:overlay val="0"/></c:legend>
  </c:chart>
  <c:externalData r:id="rIdWorkbook1"/>
</c:chartSpace>""",
        )
        package.writestr(
            "ppt/charts/_rels/chart1.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdWorkbook1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/package"
    Target="../embeddings/approval.csv"/>
</Relationships>""",
        )
    return buffer.getvalue()


def _minimal_grouped_visual_link_pptx() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr(
            "ppt/presentation.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
    xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:sldSz cx="9144000" cy="5143500"/>
  <p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst>
</p:presentation>""",
        )
        package.writestr(
            "ppt/_rels/presentation.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
    Target="slides/slide1.xml"/>
</Relationships>""",
        )
        package.writestr(
            "ppt/slides/slide1.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
    xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      <p:sp>
        <p:nvSpPr><p:cNvPr id="2" name="Title 1" title="Launch Evidence"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
        <p:txBody><a:bodyPr/><a:p><a:r><a:t>Launch Evidence</a:t></a:r></a:p></p:txBody>
      </p:sp>
      <p:grpSp>
        <p:nvGrpSpPr><p:cNvPr id="20" name="Evidence cluster" title="Launch Evidence Cluster" descr="Visual evidence and linked packet"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
        <p:grpSpPr><a:xfrm><a:off x="914400" y="914400"/><a:ext cx="2743200" cy="1828800"/></a:xfrm></p:grpSpPr>
        <p:pic>
          <p:nvPicPr>
            <p:cNvPr id="21" name="Evidence snapshot" title="Evidence Snapshot" descr="Snapshot of launch approval">
              <a:hlinkClick r:id="rIdEvidenceLink" tooltip="Open evidence packet"/>
            </p:cNvPr>
            <p:cNvPicPr/><p:nvPr/>
          </p:nvPicPr>
          <p:blipFill><a:blip r:embed="rIdImage1"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
          <p:spPr><a:xfrm><a:off x="1097280" y="1188720"/><a:ext cx="1828800" cy="914400"/></a:xfrm></p:spPr>
        </p:pic>
      </p:grpSp>
    </p:spTree>
  </p:cSld>
</p:sld>""",
        )
        package.writestr(
            "ppt/slides/_rels/slide1.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdImage1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    Target="../media/evidence.png"/>
  <Relationship Id="rIdEvidenceLink"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
    Target="https://example.com/evidence/approval-gate"
    TargetMode="External"/>
</Relationships>""",
        )
        package.writestr(
            "ppt/media/evidence.png",
            bytes.fromhex(
                "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                "1f15c4890000000a49444154789c6360000002000100e221bc3300000000"
                "49454e44ae426082"
            ),
        )
    return buffer.getvalue()


def test_embedding_inputs_preserve_complex_document_evidence_headers():
    elements = [
        DocumentElement(
            type="chart_series",
            text="Revenue Architecture Q2 point is 18 with a linear trendline.",
            location="slide 3",
            metadata={
                "slide": 3,
                "slide_title": "Revenue Plan",
                "chart_title": "Revenue Architecture",
                "chart_part": "ppt/charts/chart1.xml",
                "chart_source_part": "ppt/embeddings/Microsoft_Excel_Worksheet1.xlsx",
            },
        ),
        DocumentElement(
            type="table_cell",
            text="Owner scorecard shows Alice owns the Q2 mitigation plan.",
            location="slide 3 table 1 row 2 column 2",
            metadata={
                "slide": 3,
                "slide_title": "Revenue Plan",
                "table_name": "Owner Scorecard",
                "table_cell_reference": "B2",
                "style_summary": "bold; fill #FFF2CC",
            },
        ),
        DocumentElement(
            type="form_field",
            text="Approval required is checked.",
            location="slide 3 control 1",
            metadata={
                "slide": 3,
                "slide_title": "Revenue Plan",
                "label": "Approval required",
                "field_type": "checkbox",
                "form_field_checked": True,
            },
        ),
    ]

    chunks = _build_chunks(
        elements,
        max_chars=5000,
        respect_semantic_boundaries=False,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata["chunk_scope_type"] == "mixed"
    assert {"chart", "table", "styled_text", "form_field"}.issubset(
        set(chunk.metadata["evidence_families"]),
    )
    assert "slide 3: Revenue Plan" in chunk.metadata["location_hints"]
    assert "chart Revenue Architecture" in chunk.metadata["location_hints"]
    assert "table Owner Scorecard" in chunk.metadata["location_hints"]
    assert "form Approval required" in chunk.metadata["location_hints"]
    assert chunk.metadata["slide"] == 3
    assert chunk.metadata["slide_title"] == "Revenue Plan"
    source_locations = chunk.metadata["source_locations"]
    assert any(location.get("chart_title") == "Revenue Architecture" for location in source_locations)
    assert any(
        location.get("table_name") == "Owner Scorecard"
        and location.get("table_cell_reference") == "B2"
        for location in source_locations
    )
    assert any(
        location.get("label") == "Approval required"
        and location.get("field_type") == "checkbox"
        for location in source_locations
    )

    embedding_input = _embedding_input_sample(
        chunk,
        filename="strategy.pptx",
        domain="slides",
        content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        sha256="abc123",
        relationships=[],
        local_graph_reasoning=None,
    )
    text = embedding_input["text"]
    metadata = embedding_input["metadata"]

    assert "Evidence types: chart, table, styled text, form field" in text
    assert "Provenance: slide 3: Revenue Plan" in text
    assert "\n[chart_series] slide 3:" in text
    assert "chart Revenue Architecture" in text
    assert "table Owner Scorecard" in text
    assert "form Approval required" in text
    assert "chart_part=" not in text
    assert "form_field_checked=" not in text
    assert "semantic_text" in metadata
    assert "[form_field] slide 3 control 1: Approval required is checked." in metadata["semantic_text"]
    assert metadata["primary_evidence_family"] == "chart"
    assert metadata["primary_location_hint"] == "slide 3: Revenue Plan"
    assert metadata["source_locations"] == source_locations

    chunk_cards = _chunk_evidence_cards(chunks, limit=4)
    assert chunk_cards
    assert chunk_cards[0]["source_locations"] == source_locations
    assert {"chart", "table", "styled_text", "form_field"}.issubset(
        set(chunk_cards[0]["evidence_families"]),
    )
    assert "slide 3: Revenue Plan" in chunk_cards[0]["location_hints"]

    embedding_cards = _embedding_evidence_cards([embedding_input], limit=4)
    assert embedding_cards
    assert embedding_cards[0]["source_locations"] == source_locations
    assert "table" in embedding_cards[0]["evidence_families"]


def test_chunk_graph_context_demotes_weak_mentions_behind_semantic_edges():
    chunks = _build_chunks([
        DocumentElement(
            type="paragraph",
            text="Product Revenue drives Regional Demand. Revenue also appears on the slide.",
            location="slide 1",
            metadata={"slide": 1, "slide_title": "Revenue Architecture"},
        ),
    ])
    assert chunks
    strong = GraphCandidateRelationship(
        source="Product Revenue",
        relation="drives",
        target="Regional Demand",
        evidence="Product Revenue drives Regional Demand.",
        confidence=0.92,
        metadata={
            "element_type": "shape_connector",
            "connector_id": "c1",
            "connector_title": "drives",
            "layout_x_emu": 914400,
            "slide": 1,
            "slide_title": "Revenue Architecture",
        },
    )
    weak = GraphCandidateRelationship(
        source="Revenue",
        relation="co_occurs_with",
        target="Product Revenue",
        evidence="Revenue also appears on the slide.",
        confidence=0.54,
        metadata={"slide": 1},
    )

    assert _relationship_sample_category(weak) == "weak_mention"
    context = _chunk_graph_context(chunks[0], [weak, strong], limit=2)

    assert context[0]["relation"] == "drives"
    assert context[0]["source"] == "Product Revenue"
    assert "slide 1: Revenue Architecture" in context[0]["text"]
    assert "layout_x_emu" not in context[0]["text"]
    assert "connector_id" not in context[0]["text"]
    assert context[-1]["relation"] == "co_occurs_with"


def test_form_field_relationship_prefers_visible_date_value_over_normalized_date():
    relationships = _candidate_relationships(
        "implementation.docx",
        [
            DocumentElement(
                type="form_field",
                text="Review Date: April 15, 2026",
                location="document content control 3",
                metadata={
                    "label": "Review Date",
                    "field_type": "date",
                    "form_kind": "docx_content_control",
                    "value": "April 15, 2026",
                    "form_field_date_format": "MMMM d, yyyy",
                    "form_field_date_full_value": "2026-04-15T00:00:00Z",
                },
            ),
        ],
        [],
    )

    assert any(
        relationship.source == "Review Date"
        and relationship.relation == "has_value"
        and relationship.target == "April 15, 2026"
        and relationship.metadata.get("field_type") == "date"
        and relationship.metadata.get("form_field_date_full_value") == "2026-04-15T00:00:00Z"
        for relationship in relationships
    )
    assert not any(
        relationship.source == "Review Date"
        and relationship.relation == "has_value"
        and relationship.target == "2026-04-15T00:00:00Z"
        for relationship in relationships
    )


def test_form_field_relationship_normalizes_boolean_state_for_graph_value():
    relationships = _candidate_relationships(
        "launch-form.pptx",
        [
            DocumentElement(
                type="form_field",
                text="Approved for launch is checked.",
                location="slide 1 control 1",
                metadata={
                    "label": "Approved for launch",
                    "field_type": "checkbox",
                    "form_kind": "pptx_control",
                    "form_field_checked": True,
                },
            ),
        ],
        [],
    )

    assert any(
        relationship.source == "Approved for launch"
        and relationship.relation == "has_value"
        and relationship.target == "checked"
        and relationship.metadata.get("form_field_checked") is True
        for relationship in relationships
    )
    assert not any(
        relationship.source == "Approved for launch"
        and relationship.relation == "has_value"
        and relationship.target == "True"
        for relationship in relationships
    )


def test_candidate_relationships_preserves_semantic_edges_after_many_form_fields():
    elements = [
        DocumentElement(
            type="form_field",
            text=f"Checklist item {index} is checked.",
            location=f"slide 1 control {index}",
            metadata={
                "label": f"Checklist item {index}",
                "field_type": "checkbox",
                "form_kind": "pptx_control",
                "form_field_checked": True,
            },
        )
        for index in range(1, 10)
    ]
    elements.extend([
        DocumentElement(
            type="paragraph",
            text="Product Revenue drives Regional Demand. Regional Demand references Risk Map.",
            location="slide 2",
            metadata={"slide": 2, "slide_title": "Revenue Architecture"},
        ),
        DocumentElement(
            type="table_row",
            text="Approval Gate | gates | Launch Workflow | Launch requires approval before release",
            location="slide 2 table 1 row 2",
            metadata={
                "headers": ["Source", "Relationship", "Target", "Evidence"],
                "cells": [
                    "Approval Gate",
                    "gates",
                    "Launch Workflow",
                    "Launch requires approval before release",
                ],
                "slide": 2,
                "slide_title": "Revenue Architecture",
                "table": 1,
                "row": 2,
            },
        ),
    ])

    relationships = _candidate_relationships(
        "form-heavy-deck.pptx",
        elements,
        _candidate_entities(elements),
        limit=8,
    )
    direct_edges = {
        (relationship.source, relationship.relation, relationship.target)
        for relationship in relationships
    }

    assert ("Product Revenue", "drives", "Regional Demand") in direct_edges
    assert ("Regional Demand", "references", "Risk Map") in direct_edges
    assert ("Approval Gate", "gates", "Launch Workflow") in direct_edges
    assert any(
        relationship.source == "Checklist item 1"
        and relationship.relation == "has_value"
        and relationship.target == "checked"
        for relationship in relationships
    )


def test_chunk_graph_context_prefers_direct_connector_relation_over_container_edge():
    chunks = _build_chunks([
        DocumentElement(
            type="shape_connector",
            text="Regional Demand -> Risk Map: references",
            location="slide 1",
            metadata={
                "slide": 1,
                "slide_title": "Revenue Architecture",
                "source_label": "Regional Demand",
                "target_label": "Risk Map",
                "relation": "references",
            },
        ),
    ])
    assert chunks

    wrapper = GraphCandidateRelationship(
        source="Revenue Architecture",
        relation="contains_connector",
        target="Regional Demand -> Risk Map: references",
        evidence="Regional Demand -> Risk Map: references",
        confidence=0.6,
        metadata={
            "element_type": "shape_connector",
            "slide": 1,
            "slide_title": "Revenue Architecture",
        },
    )
    direct = GraphCandidateRelationship(
        source="Regional Demand",
        relation="references",
        target="Risk Map",
        evidence="Regional Demand -> Risk Map: references",
        confidence=0.72,
        metadata={
            "element_type": "shape_connector",
            "slide": 1,
            "slide_title": "Revenue Architecture",
            "source_label": "Regional Demand",
            "target_label": "Risk Map",
            "relation": "references",
        },
    )

    assert _relationship_sample_category(direct) == "semantic_connector"
    assert _relationship_sample_category(wrapper) == "connector"

    context = _chunk_graph_context(chunks[0], [wrapper, direct], limit=2)

    assert context[0]["source"] == "Regional Demand"
    assert context[0]["relation"] == "references"
    assert context[0]["target"] == "Risk Map"
    assert context[1]["relation"] == "contains_connector"


def test_chunk_graph_context_matches_visual_connectors_by_shape_provenance():
    chunks = _build_chunks([
        DocumentElement(
            type="shape_connector",
            text="Connector between approval gate and launch workflow.",
            location="slide 1 connector 1",
            metadata={
                "slide": 1,
                "slide_title": "Launch Approval Packet",
                "element_type": "shape_connector",
                "connector_id": "5",
                "connector_title": "gates",
                "source_shape_id": "3",
                "target_shape_id": "4",
                "source_connection_index": "0",
                "target_connection_index": "0",
                "source_label": "Approval required",
                "target_label": "Launch workflow",
            },
        ),
    ])
    assert chunks

    relationship = GraphCandidateRelationship(
        source="Approval required",
        relation="gates",
        target="Launch workflow",
        evidence="Approval gate connector.",
        confidence=0.78,
        metadata={
            "element_type": "shape_connector",
            "slide": 1,
            "slide_title": "Launch Approval Packet",
            "connector_id": "5",
            "connector_title": "gates",
            "source_shape_id": "3",
            "target_shape_id": "4",
            "source_connection_index": "0",
            "target_connection_index": "0",
            "source_label": "Approval required",
            "target_label": "Launch workflow",
        },
    )

    context = _chunk_graph_context(chunks[0], [relationship], limit=1)

    assert context[0]["relation"] == "gates"
    assert "provenance_exact:connector_title" in context[0]["chunk_match_reasons"]
    assert "provenance_exact:source_shape_id" in context[0]["chunk_match_reasons"]
    assert "provenance_exact:target_shape_id" in context[0]["chunk_match_reasons"]
    assert "connector gates" in context[0]["text"]


def test_local_graph_reasoning_summarizes_connector_shape_provenance():
    relationships = [
        GraphCandidateRelationship(
            source="Approval required",
            relation="gates",
            target="Launch workflow",
            evidence="Approval required before launch workflow.",
            confidence=0.78,
            metadata={
                "element_type": "shape_connector",
                "slide": 1,
                "slide_title": "Launch Approval Packet",
                "connector_id": "5",
                "connector_title": "gates",
                "source_shape_id": "3",
                "target_shape_id": "4",
                "source_label": "Approval required",
                "target_label": "Launch workflow",
                "grouped_visual_label": "Launch workflow",
            },
        ),
        GraphCandidateRelationship(
            source="Launch workflow",
            relation="reviewed_by",
            target="Finance Review Board",
            evidence="Finance Review Board approved launch.",
            confidence=0.72,
            metadata={
                "element_type": "table_row",
                "slide": 1,
                "slide_title": "Launch Approval Packet",
                "table_cell_reference": "R2C2",
            },
        ),
    ]

    graph = _build_local_graph_artifact(
        filename="approval-deck.pptx",
        domain="slides",
        entities=[],
        relationships=relationships,
    )
    reasoning = _build_local_graph_reasoning_index(graph)

    launch_bridge = next(
        bridge
        for bridge in reasoning["bridge_entities"]
        if bridge["name"] == "Launch workflow"
    )
    connector_summary = next(
        relationship
        for relationship in launch_bridge["relationships"]
        if relationship["relation"] == "gates"
    )
    assert "connector=gates" in connector_summary["provenance_summary"]
    assert "grouped_visual=Launch workflow" in connector_summary["provenance_summary"]
    assert "shapes=3 -> 4" in connector_summary["provenance_summary"]
    assert connector_summary["metadata"]["grouped_visual_label"] == "Launch workflow"
    assert "connector" in connector_summary["evidence_families"]

    path = next(
        candidate
        for candidate in reasoning["path_candidates"]
        if candidate["bridge"] == "Launch workflow"
        and {
            candidate["source"],
            candidate["target"],
        } == {"Approval required", "Finance Review Board"}
    )
    assert "connector=gates" in path["provenance_summary"][0]
    assert "grouped_visual=Launch workflow" in path["provenance_summary"][0]
    assert "shapes=3 -> 4" in path["relationships"][0]["provenance_summary"]


def test_candidate_relationships_add_visual_layout_neighbors_for_slide_clusters():
    elements = [
        DocumentElement(
            type="chart_series",
            text="Product Revenue chart: Q1 12, Q2 18, Q3 31.",
            location="slide 4 chart 1",
            metadata={
                "slide": 4,
                "slide_title": "Revenue Decision",
                "chart_title": "Product Revenue Forecast",
                "chart_part": "ppt/charts/chart1.xml",
                "series_name": "Product Revenue",
                "layout_center_x_pct": 35.0,
                "layout_center_y_pct": 42.0,
                "layout_zone": "center_left",
                "layout_bbox": "x=2.0in y=2.0in w=3.0in h=2.0in",
            },
        ),
        DocumentElement(
            type="form_field",
            text="Launch approved is checked.",
            location="slide 4 control 1",
            metadata={
                "slide": 4,
                "slide_title": "Revenue Decision",
                "label": "Launch approved",
                "field_type": "checkbox",
                "form_kind": "pptx_control",
                "form_field_checked": True,
                "layout_center_x_pct": 49.0,
                "layout_center_y_pct": 44.0,
                "layout_zone": "center",
                "layout_bbox": "x=4.8in y=2.3in w=1.6in h=0.4in",
            },
        ),
        DocumentElement(
            type="image",
            text="Risk map image.",
            location="slide 4 image 1",
            metadata={
                "slide": 4,
                "slide_title": "Revenue Decision",
                "title": "Risk Map",
                "image_part": "ppt/media/risk.png",
                "layout_center_x_pct": 82.0,
                "layout_center_y_pct": 82.0,
                "layout_zone": "bottom_right",
            },
        ),
    ]

    relationships = _candidate_relationships(
        "layout-deck.pptx",
        elements,
        _candidate_entities(elements),
        limit=24,
    )
    visual_relationships = [
        relationship
        for relationship in relationships
        if relationship.relation == "visually_near"
    ]

    assert [
        (relationship.source, relationship.target)
        for relationship in visual_relationships
    ] == [("Product Revenue Forecast", "Launch approved")]

    visual = visual_relationships[0]
    assert visual.metadata["layout_relation"] == "right_of"
    assert visual.metadata["visual_distance_pct"] < 15
    assert visual.metadata["related_element_type"] == "form_field"
    assert visual.metadata["related_label"] == "Launch approved"
    assert visual.metadata["target_layout_zone"] == "center"
    assert _relationship_sample_category(visual) == "layout"
    assert "Risk Map" not in {relationship.target for relationship in visual_relationships}

    graph = _build_local_graph_artifact(
        filename="layout-deck.pptx",
        domain="slides",
        entities=[],
        relationships=relationships,
    )
    reasoning = _build_local_graph_reasoning_index(graph)
    bridge = next(
        bridge
        for bridge in reasoning["bridge_entities"]
        if bridge["name"] == "Product Revenue Forecast"
    )
    summary = next(
        relationship
        for relationship in bridge["relationships"]
        if relationship["relation"] == "visually_near"
    )
    assert summary["score"] >= 0.75
    assert "layout" in summary["evidence_families"]
    assert summary["metadata"]["layout_relation"] == "right_of"
    assert summary["metadata"]["related_label"] == "Launch approved"
    assert "visual=right_of" in summary["provenance_summary"]
    assert "near=Launch approved" in summary["provenance_summary"]


def test_pptx_grouped_image_links_keep_group_provenance_for_graph_drilldown():
    elements = _extract_pptx(_minimal_grouped_visual_link_pptx())
    link = next(
        element
        for element in elements
        if element.type == "link"
        and element.metadata.get("uri") == "https://example.com/evidence/approval-gate"
    )

    assert link.metadata["parent_group_label"] == "Launch Evidence Cluster"
    assert link.metadata["title"] == "Evidence Snapshot"
    assert link.metadata["description"] == "Snapshot of launch approval"
    assert link.metadata["image_part"] == "ppt/media/evidence.png"

    relationships = _candidate_relationships(
        "linked-visual.pptx",
        elements,
        _candidate_entities(elements),
    )

    assert any(
        relationship.source == "Launch Evidence Cluster"
        and relationship.relation == "contains_link"
        and relationship.target == "Evidence Snapshot"
        and relationship.metadata.get("container_type") == "shape_group"
        and relationship.metadata.get("container_label") == "Launch Evidence Cluster"
        for relationship in relationships
    )


def test_pptx_extraction_promotes_rich_slide_evidence_to_graph_candidates():
    elements = _extract_pptx(_minimal_rich_pptx())
    element_types = {element.type for element in elements}

    assert {
        "slide_metadata",
        "slide_title",
        "slide_text",
        "shape_group",
        "shape_connector",
        "table_cell",
        "table_row",
        "chart_series",
        "chart_point",
        "chart_axis",
        "chart_legend",
        "chart_data_label",
        "chart_trendline",
        "comment",
        "speaker_notes",
        "form_field",
        "image",
        "media",
        "embedded_object",
    }.issubset(element_types)

    styled_slide = next(
        element
        for element in elements
        if element.type == "slide_text" and "Approval required" in element.text
    )
    group = next(element for element in elements if element.type == "shape_group")
    assert group.metadata["group_id"] == "20"
    assert group.metadata["group_label"] == "Approval Dependency Cluster"
    assert group.metadata["group_child_count"] == 3
    assert "Approval required before launch workflow." in group.metadata["group_child_labels"]
    assert "Launch workflow" in group.metadata["group_child_labels"]
    assert "gates" in group.text
    assert group.metadata["layout_bbox"] == "x=0.60in y=1.20in w=5.40in h=1.20in"

    assert styled_slide.metadata["parent_group_id"] == "20"
    assert styled_slide.metadata["parent_group_label"] == "Approval Dependency Cluster"
    assert styled_slide.metadata["emphasis_terms"] == ["Approval required"]
    assert styled_slide.metadata["bullet_levels"] == [0]
    assert styled_slide.metadata["runs"][0]["color"] == "#00A884"
    assert styled_slide.metadata["runs"][0]["highlight_color"] == "#FFF2CC"
    assert styled_slide.metadata["runs"][0]["size_1_100_points"] == "2400"
    assert "highlight #FFF2CC" in styled_slide.metadata["style_summary"]
    assert "24pt" in styled_slide.metadata["style_summary"]

    comment = next(element for element in elements if element.type == "comment")
    assert comment.metadata["author"] == "Maya Reviewer"
    assert comment.metadata["layout_zone"] == "top_right"
    assert "approval evidence" in comment.text

    speaker_notes = next(element for element in elements if element.type == "speaker_notes")
    assert speaker_notes.metadata["notes_part"] == "ppt/notesSlides/notesSlide1.xml"
    assert "Finance Review Board" in speaker_notes.text

    form_field = next(element for element in elements if element.type == "form_field")
    assert form_field.metadata["form_kind"] == "pptx_control"
    assert form_field.metadata["field_type"] == "Forms.CheckBox.1"
    assert form_field.metadata["form_field_checked"] is True
    assert form_field.metadata["label"] == "Approval required"

    relation_cell = next(
        element
        for element in elements
        if element.type == "table_cell"
        and element.metadata.get("table_cell_reference") == "R2C2"
    )
    assert relation_cell.metadata["table_cell_column_header"] == "Relation"
    assert relation_cell.metadata["table_cell_row_header"] == "Approval required"
    assert relation_cell.metadata["table_cell_context"] == "Relation for Approval required"

    connector = next(element for element in elements if element.type == "shape_connector")
    assert connector.metadata["parent_group_id"] == "20"
    assert connector.metadata["parent_group_label"] == "Approval Dependency Cluster"

    media = next(element for element in elements if element.type == "media")
    assert media.metadata["media_kind"] == "audio"
    assert media.metadata["media_part"] == "ppt/media/briefing.mp3"

    image = next(element for element in elements if element.type == "image")
    assert image.metadata["title"] == "Risk Map"
    assert image.metadata["description"] == "Risk map highlights regional launch blockers"
    assert image.metadata["image_part"] == "ppt/media/risk-map.png"
    assert image.metadata["image_mime"] == "image/png"
    assert "Risk map highlights regional launch blockers" in image.text

    embedded = next(element for element in elements if element.type == "embedded_object")
    assert embedded.metadata["embedded_file_filename"] == "approval.csv"
    assert embedded.metadata["media_kind"] == "embedded_package"
    assert any(
        element.type == "table_row"
        and element.metadata.get("embedded_file_filename") == "approval.csv"
        and "Finance Review Board" in element.text
        for element in elements
    )

    chart_series = next(element for element in elements if element.type == "chart_series")
    assert chart_series.metadata["chart_series_fill_color"] == "#00A884"
    assert chart_series.metadata["chart_series_line_color"] == "#123456"
    assert chart_series.metadata["chart_series_line_dash"] == "dash"
    assert chart_series.metadata["chart_marker_symbol"] == "diamond"
    assert chart_series.metadata["chart_marker_fill_color"] == "#C00000"
    assert chart_series.metadata["chart_source_part"] == "ppt/embeddings/approval.csv"

    styled_point = next(
        element
        for element in elements
        if element.type == "chart_point" and element.metadata.get("category") == "Q3"
    )
    assert styled_point.metadata["chart_point_fill_color"] == "#FFB000"
    assert "fill #FFB000" in styled_point.text

    axis_titles = {
        element.metadata.get("chart_axis_type"): element.metadata.get("chart_axis_title")
        for element in elements
        if element.type == "chart_axis"
    }
    assert axis_titles["category"] == "Quarter"
    assert axis_titles["value"] == "Revenue impact"

    legend = next(element for element in elements if element.type == "chart_legend")
    assert legend.metadata["chart_legend_position"] == "r"
    assert legend.metadata["chart_legend_series"] == ["Approval"]

    data_label = next(element for element in elements if element.type == "chart_data_label")
    assert set(data_label.metadata["chart_label_fields"]) == {"value", "category name"}
    assert data_label.metadata["chart_label_position"] == "outEnd"

    trendline = next(element for element in elements if element.type == "chart_trendline")
    assert trendline.metadata["chart_trendline_name"] == "approval velocity"
    assert trendline.metadata["chart_trendline_type"] == "linear"
    assert trendline.metadata["chart_trendline_display_equation"] is True
    assert trendline.metadata["chart_trendline_display_r_squared"] is True
    assert trendline.metadata["chart_trendline_line_color"] == "#7030A0"
    assert "shows equation" in trendline.text

    assert any(
        element.type == "table_row"
        and element.metadata.get("chart_source_part") == "ppt/embeddings/approval.csv"
        and element.metadata.get("chart_source_extraction_method") == "csv_table"
        and "Finance Review Board" in element.text
        for element in elements
    )

    chunks = _build_chunks(elements, max_chars=9000, respect_semantic_boundaries=False)
    assert chunks
    chunk_families = set(chunks[0].metadata["evidence_families"])
    assert {
        "attachment",
        "comment",
        "form_field",
        "media",
        "speaker_notes",
    }.issubset(chunk_families)
    all_chunk_families = {
        family
        for chunk in chunks
        for family in chunk.metadata.get("evidence_families", [])
    }
    all_location_hints = {
        hint
        for chunk in chunks
        for hint in chunk.metadata.get("location_hints", [])
    }
    assert "shape" in all_chunk_families
    assert "image" in all_chunk_families
    assert any("group Approval Dependency Cluster" in hint for hint in all_location_hints)
    assert "cell context Relation for Approval required" in all_location_hints
    image_chunk = next(
        chunk
        for chunk in chunks
        if "image" in chunk.metadata.get("evidence_families", [])
    )

    embedding_input = _embedding_input_sample(
        chunks[0],
        filename="approval-deck.pptx",
        domain="slides",
        content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        sha256="abc123",
        relationships=[],
        local_graph_reasoning=None,
    )
    assert "Evidence types:" in embedding_input["text"]
    assert "comment" in embedding_input["text"]
    assert "speaker notes" in embedding_input["text"]
    assert "form field" in embedding_input["text"]
    assert "Approval Dependency Cluster" in embedding_input["text"]
    assert any(
        location.get("group_label") == "Approval Dependency Cluster"
        for location in embedding_input["metadata"]["source_locations"]
    )
    image_embedding_input = _embedding_input_sample(
        image_chunk,
        filename="approval-deck.pptx",
        domain="slides",
        content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        sha256="abc123",
        relationships=[],
        local_graph_reasoning=None,
    )
    assert "image" in image_embedding_input["text"]
    assert "Risk Map" in image_embedding_input["text"]

    relationships = _candidate_relationships(
        "approval-deck.pptx",
        elements,
        _candidate_entities(elements),
    )
    relationship_categories = {
        _relationship_sample_category(relationship)
        for relationship in relationships
    }

    assert any(
        relationship.relation == "gates"
        and "Approval required" in relationship.source
        and relationship.target == "Launch workflow"
        for relationship in relationships
    )
    assert any(
        relationship.relation == "emphasizes"
        and relationship.target == "Approval required"
        and "highlight #FFF2CC" in relationship.evidence
        for relationship in relationships
    )
    assert any(
        relationship.relation == "uses_bullet_structure"
        and "bullet levels 0" in relationship.target
        for relationship in relationships
    )
    assert any(
        relationship.relation == "has_value"
        and relationship.source == "Approval required"
        and relationship.target == "checked"
        for relationship in relationships
    )
    assert any(
        relationship.relation == "contains_chart"
        and "Revenue Plan" in relationship.target
        for relationship in relationships
    )
    assert any(
        relationship.relation == "q2_value"
        and relationship.source == "Approval"
        and relationship.target == "18"
        for relationship in relationships
    )
    assert any(
        relationship.relation == "has_chart_value"
        and relationship.source == "Q3"
        and relationship.target == "31"
        and relationship.metadata.get("chart_label") == "Revenue Plan"
        and relationship.metadata.get("series_name") == "Approval"
        for relationship in relationships
    )
    assert any(
        relationship.relation == "shows_category_value"
        and relationship.source == "Revenue Plan"
        and relationship.target == "Q3: 31"
        and relationship.metadata.get("chart_point_label") == "Approval Q3"
        for relationship in relationships
    )
    assert any(
        relationship.relation == "has_axis"
        and "Revenue Plan" in relationship.source
        and "Revenue impact" in relationship.target
        for relationship in relationships
    )
    assert any(
        relationship.relation == "has_data_labels"
        and "Revenue Plan" in relationship.source
        and "Approval data labels" in relationship.target
        for relationship in relationships
    )
    assert any(
        relationship.relation == "has_trendline"
        and "Approval" in relationship.source
        and "linear trendline" in relationship.target
        for relationship in relationships
    )
    assert any(
        relationship.relation == "contains_image"
        and relationship.source == "approval-deck.pptx"
        and relationship.target == "Risk Map"
        for relationship in relationships
    )
    assert any(
        relationship.relation == "contains_shape_group"
        and relationship.source == "approval-deck.pptx"
        and relationship.target == "Approval Dependency Cluster"
        and relationship.metadata.get("group_id") == "20"
        for relationship in relationships
    )
    assert any(
        relationship.relation == "groups_visual"
        and relationship.source == "Approval Dependency Cluster"
        and relationship.target == "Launch workflow"
        and relationship.metadata.get("grouped_visual_label") == "Launch workflow"
        for relationship in relationships
    )
    trendline_relationship = next(
        relationship
        for relationship in relationships
        if relationship.relation == "has_trendline"
        and "Approval" in relationship.source
        and "linear trendline" in relationship.target
    )
    assert _relationship_sample_category(trendline_relationship) == "chart_fact"
    assert {"connector", "shape", "styled_text", "table_fact", "chart_fact"}.issubset(
        relationship_categories,
    )
    assert {"comment", "speaker_notes", "form_field", "media", "attachment"}.issubset(
        relationship_categories,
    )
    assert "image" in relationship_categories


def test_candidate_relationships_extract_direct_relation_verbs_from_text():
    elements = [
        DocumentElement(
            type="heading",
            text="GraphRAG UX Flow Test",
            location="line 1",
        ),
        DocumentElement(
            type="paragraph",
            text=(
                "Product Revenue drives Regional Demand on page 1. "
                "Regional Demand references Risk Map for launch gating."
            ),
            location="line 3",
            metadata={"section": "GraphRAG UX Flow Test"},
        ),
        DocumentElement(
            type="table_row",
            text="Product Revenue | 12 | 18",
            location="line 7",
            metadata={"section": "GraphRAG UX Flow Test", "table": 1, "row": 2},
        ),
    ]

    entities = _candidate_entities(elements)
    entity_names = {entity["name"] for entity in entities}
    assert {"Product Revenue", "Regional Demand", "Risk Map"}.issubset(entity_names)
    assert "Regional Demand. Regional Demand" not in entity_names
    assert "Regional Demand on page" not in entity_names
    assert "Risk Map for" not in entity_names
    assert "Risk Map." not in entity_names

    relationships = _candidate_relationships(
        "graphrag-ui-flow-test.md",
        elements,
        entities,
    )

    direct_edges = {
        (relationship.source, relationship.relation, relationship.target)
        for relationship in relationships
    }
    assert ("Product Revenue", "drives", "Regional Demand") in direct_edges
    assert ("Regional Demand", "references", "Risk Map") in direct_edges
    assert ("Regional Demand", "references", "Risk Map for") not in direct_edges

    drives = next(
        relationship
        for relationship in relationships
        if relationship.source == "Product Revenue"
        and relationship.relation == "drives"
        and relationship.target == "Regional Demand"
    )
    assert drives.confidence >= 0.7
    assert drives.metadata["semantic_relation_extraction"] == "verb_pattern"
    assert drives.metadata["location"] == "line 3"

    chunks = _build_chunks(
        elements,
        max_chars=4000,
        respect_semantic_boundaries=False,
    )
    semantic_chunk = next(
        chunk
        for chunk in chunks
        if "Product Revenue drives Regional Demand" in chunk.text
    )
    embedding_input = _embedding_input_sample(
        semantic_chunk,
        filename="graphrag-ui-flow-test.md",
        domain="test",
        content_type="text/markdown",
        sha256="semantic123",
        relationships=relationships,
        local_graph_reasoning=None,
    )
    assert embedding_input["metadata"]["sample_category"] == "semantic_relation"
    assert "semantic_relation" in embedding_input["metadata"]["sample_categories"]
    assert (
        embedding_input["metadata"]["sample_reason"]
        == "Chunk preserves direct subject-relation-object evidence for graph search and path answers"
    )
    assert embedding_input["metadata"]["graph_context_relationship_count"] >= 2

    embedding_cards = _embedding_evidence_cards([embedding_input], limit=1)
    assert embedding_cards[0]["category"] == "semantic_relation"
    assert "direct subject-relation-object" in embedding_cards[0]["reason"]

    entity_like_relationships = [
        relationship
        for relationship in relationships
        if relationship.relation in {"drives", "references", "co_occurs_with"}
    ]
    assert not any(
        ". " in relationship.source
        or ". " in relationship.target
        or relationship.source.endswith(".")
        or relationship.target.endswith(".")
        for relationship in entity_like_relationships
    )


def test_candidate_relationships_promotes_semantic_csv_row_triples():
    elements = _extract_csv(
        b"Source,Relationship,Target,Evidence,Owner,Status\n"
        b"Approval Gate,gates,Launch Workflow,Launch requires approval before release,Operations,Ready\n"
    )

    relationships = _candidate_relationships(
        "launch-evidence.csv",
        elements,
        _candidate_entities(elements),
    )

    semantic = next(
        relationship
        for relationship in relationships
        if relationship.source == "Approval Gate"
        and relationship.relation == "gates"
        and relationship.target == "Launch Workflow"
    )
    assert semantic.evidence == "Launch requires approval before release"
    assert semantic.confidence >= 0.9
    assert semantic.metadata["semantic_relation_extraction"] == "table_row_triple"
    assert semantic.metadata["table_fact_kind"] == "semantic_row_triple"
    assert semantic.metadata["table_source_header"] == "Source"
    assert semantic.metadata["table_relation_header"] == "Relationship"
    assert semantic.metadata["table_target_header"] == "Target"
    assert semantic.metadata["table_evidence_header"] == "Evidence"
    assert _relationship_sample_category(semantic) == "semantic_connector"

    assert not any(
        relationship.source == "Approval Gate"
        and relationship.relation == "target"
        and relationship.target == "Launch Workflow"
        for relationship in relationships
    )

    graph = _build_local_graph_artifact(
        filename="launch-evidence.csv",
        domain="test",
        entities=_candidate_entities(elements),
        relationships=relationships,
    )
    reasoning = _build_local_graph_reasoning_index(graph)
    approval_bridge = next(
        bridge
        for bridge in reasoning["bridge_entities"]
        if bridge["name"] == "Approval Gate"
    )

    assert approval_bridge["relationships"][0]["relation"] == "gates"
    assert approval_bridge["relationships"][0]["target"] == "Launch Workflow"
    assert approval_bridge["relationships"][0]["evidence_families"][:2] == ["table", "semantic"]
