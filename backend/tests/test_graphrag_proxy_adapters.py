import importlib.util
import asyncio
from email.message import EmailMessage
import io
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, Mock, patch
import zipfile

import httpx
import pytest
from fastapi import BackgroundTasks, Request, UploadFile
from fastapi.exceptions import HTTPException


GRAPHRAG_ROUTES_DIR = Path(__file__).resolve().parents[1] / "routes" / "graphrag"
ROUTES_DIR = GRAPHRAG_ROUTES_DIR.parent


def load_graphrag_route_module(module_name: str):
    """Load a GraphRAG route module without importing routes/__init__.py."""
    routes_pkg = sys.modules.get("routes")
    if routes_pkg is None:
        routes_pkg = ModuleType("routes")
        routes_pkg.__path__ = [str(ROUTES_DIR)]
        sys.modules["routes"] = routes_pkg

    graphrag_pkg = sys.modules.get("routes.graphrag")
    if graphrag_pkg is None:
        graphrag_pkg = ModuleType("routes.graphrag")
        graphrag_pkg.__path__ = [str(GRAPHRAG_ROUTES_DIR)]
        sys.modules["routes.graphrag"] = graphrag_pkg

    qualified_name = f"routes.graphrag.{module_name}"
    spec = importlib.util.spec_from_file_location(
        qualified_name,
        GRAPHRAG_ROUTES_DIR / f"{module_name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def graphrag_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHRAG_ENABLED", "true")
    monkeypatch.setenv("GRAPHRAG_URL", "http://graphrag.test")
    monkeypatch.setenv("GRAPHRAG_PIPELINE_RUNS_FILE", str(tmp_path / "pipeline_runs.json"))
    monkeypatch.delenv("CODE_RAG_URL", raising=False)
    from modules.graphrag.config import reset_config
    from modules.graphrag.pipeline_runs import reset_pipeline_run_store

    reset_config()
    reset_pipeline_run_store()
    yield
    reset_config()
    reset_pipeline_run_store()


def _mock_response(payload, status_code: int = 200):
    response = Mock()
    response.json.return_value = payload
    response.status_code = status_code
    response.content = b'{"ok":true}'
    response.headers = {"content-type": "application/json"}
    response.raise_for_status = Mock()
    return response


def _mock_upload(filename: str, content: bytes, content_type: str = "application/octet-stream"):
    upload = Mock()
    upload.filename = filename
    upload.content_type = content_type
    upload.read = AsyncMock(return_value=content)
    return upload


def _reset_pipeline_run_cache(ingest):
    ingest._PIPELINE_RUNS.clear()
    ingest._PIPELINE_RUNS_LOADED = False
    from modules.graphrag.pipeline_runs import reset_pipeline_run_store

    reset_pipeline_run_store()


def _minimal_eml_bytes() -> bytes:
    message = EmailMessage()
    message["Subject"] = "GraphRAG Launch Evidence"
    message["From"] = "Ada Lovelace <ada@example.com>"
    message["To"] = "Grace Hopper <grace@example.com>, Katherine Johnson <katherine@example.com>"
    message["Cc"] = "Alan Turing <alan@example.com>"
    message["Date"] = "Tue, 14 Jul 2026 12:00:00 -0400"
    message["Message-ID"] = "<launch-evidence@example.com>"
    message.set_content(
        "Please review the GraphRAG launch evidence at https://example.com/runbook.\n"
        "The attached evidence table lists Graph Coverage ownership and readiness."
    )
    message.add_attachment(
        b"Metric,Owner,Status\nGraph Coverage,Grace Hopper,Ready\n",
        maintype="text",
        subtype="csv",
        filename="launch-evidence.csv",
    )
    return message.as_bytes()


def _minimal_pdf_bytes() -> bytes:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page()
    document.set_metadata({
        "title": "GraphRAG Annual Strategy",
        "author": "Ada Lovelace",
        "subject": "Graph extraction readiness",
        "keywords": "GraphRAG; PDF; Strategy",
        "creator": "Llama Nexus",
        "producer": "PyMuPDF",
        "creationDate": "D:20260102030405",
        "modDate": "D:20260103040506",
    })
    page.insert_text((72, 72), "Annual Strategy", fontsize=18, fontname="helv")
    page.insert_text((72, 104), "GraphRAG connects PDF evidence to answers.", fontsize=11, fontname="helv")
    document.set_toc([[1, "Annual Strategy", 1]])
    page.insert_link({
        "kind": fitz.LINK_URI,
        "from": fitz.Rect(72, 124, 240, 144),
        "uri": "https://example.com/graphrag",
    })
    page.insert_link({
        "kind": fitz.LINK_URI,
        "from": fitz.Rect(72, 146, 240, 158),
        "uri": "https://example.com/evidence-index",
    })
    page.insert_link({
        "kind": fitz.LINK_URI,
        "from": fitz.Rect(248, 124, 392, 144),
        "uri": "https://example.com/pdf-source",
    })

    widget = fitz.Widget()
    widget.field_name = "approved_by"
    widget.field_label = "Approved By"
    widget.field_value = "Ada Lovelace"
    widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    widget.rect = fitz.Rect(72, 160, 260, 182)
    page.add_widget(widget)

    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10), 0)
    pixmap.clear_with(0x336699)
    page.insert_image(fitz.Rect(72, 200, 120, 248), pixmap=pixmap)

    table_x = 72
    table_y = 280
    row_height = 24
    column_widths = [100, 130, 90]
    table_rows = [
        ["Metric", "Owner", "Status"],
        ["Evidence", "Ada Lovelace", "Approved"],
    ]
    x_positions = [table_x]
    for width in column_widths:
        x_positions.append(x_positions[-1] + width)
    y_positions = [table_y + row_height * index for index in range(len(table_rows) + 1)]
    for x_position in x_positions:
        page.draw_line(
            fitz.Point(x_position, table_y),
            fitz.Point(x_position, y_positions[-1]),
            color=(0, 0, 0),
            width=0.8,
        )
    for y_position in y_positions:
        page.draw_line(
            fitz.Point(table_x, y_position),
            fitz.Point(x_positions[-1], y_position),
            color=(0, 0, 0),
            width=0.8,
        )
    for row_index, row in enumerate(table_rows):
        for column_index, value in enumerate(row):
            page.insert_text(
                (x_positions[column_index] + 4, table_y + row_height * row_index + 16),
                value,
                fontsize=10,
                fontname="helv",
            )

    annotation = page.add_text_annot(fitz.Point(260, 104), "Review graph-grounded evidence.")
    annotation.set_info(
        title="Grace Hopper",
        subject="Review Note",
        content="Review graph-grounded evidence.",
    )
    annotation.update()

    document.embfile_add(
        "source-data.csv",
        b"Artifact,Curator\nSource Data,Grace Hopper",
        filename="source-data.csv",
        desc="Source data table",
    )

    return document.tobytes()


def _image_only_pdf_bytes() -> bytes:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page()

    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 80, 32), 0)
    pixmap.clear_with(0xFFFFFF)
    page.insert_image(fitz.Rect(72, 72, 360, 188), pixmap=pixmap)

    return document.tobytes()


def _minimal_png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe"
        b"\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _minimal_zip_bundle_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes/plan.md", "# Launch Plan\nGraphRAG bundle evidence.")
        archive.writestr("data/owners.csv", "Artifact,Owner\nBundle Evidence,Grace Hopper")
        archive.writestr("data/metadata.json", '{"project": {"name": "Bundle Graph", "status": "ready"}}')
        archive.writestr("images/snapshot.png", _minimal_png_bytes())
    return buffer.getvalue()


def _minimal_epub_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "OEBPS/chapter1.xhtml",
            """
            <html xmlns="http://www.w3.org/1999/xhtml">
              <body>
                <h1>GraphRAG Field Guide</h1>
                <p>Ada Lovelace maps narrative entities to reusable evidence.</p>
                <table>
                  <tr><th>Entity</th><th>Role</th></tr>
                  <tr><td>GraphRAG</td><td>Knowledge Graph</td></tr>
                </table>
                <a href="https://example.com/field-guide">Field guide source</a>
                <img src="figures/map.png" alt="Relationship map"/>
              </body>
            </html>
            """,
        )
    return buffer.getvalue()


def _minimal_odt_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        archive.writestr(
            "content.xml",
            """
            <office:document-content
              xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
              xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
              xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
              xmlns:dc="http://purl.org/dc/elements/1.1/"
              xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"
              xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
              xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
              xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
              xmlns:xlink="http://www.w3.org/1999/xlink">
              <office:automatic-styles>
                <style:style style:name="StrongTerm" style:family="text">
                  <style:text-properties fo:font-weight="bold"/>
                </style:style>
                <style:style style:name="ReviewChecklist" style:family="paragraph">
                  <style:text-properties fo:font-weight="bold" fo:font-style="italic" fo:color="#126B5D"/>
                </style:style>
              </office:automatic-styles>
              <office:body><office:text>
                <text:tracked-changes>
                  <text:changed-region text:id="ct-insert-1">
                    <text:insertion>
                      <office:change-info>
                        <dc:creator>Alan Turing</dc:creator>
                        <dc:date>2026-03-02T10:30:00Z</dc:date>
                      </office:change-info>
                    </text:insertion>
                  </text:changed-region>
                  <text:changed-region text:id="ct-delete-1">
                    <text:deletion>
                      <office:change-info>
                        <dc:creator>Katherine Johnson</dc:creator>
                        <dc:date>2026-03-03T11:45:00Z</dc:date>
                      </office:change-info>
                      <text:p>Legacy file staging checklist</text:p>
                    </text:deletion>
                  </text:changed-region>
                </text:tracked-changes>
                <text:h text:outline-level="1">Deployment Plan</text:h>
                <text:p>Priority emphasis: <text:span text:style-name="StrongTerm">Graph handoff routing</text:span> keeps citation paths inspectable.</text:p>
                <text:list>
                  <text:list-item>
                    <text:p text:style-name="ReviewChecklist">Citation path review</text:p>
                  </text:list-item>
                </text:list>
                <text:p>Llama Nexus prepares structured GraphRAG uploads from <text:a xlink:href="https://example.com/odf-source">ODF source material</text:a>.</text:p>
                <text:p>Revision trail: <text:change-start text:change-id="ct-insert-1"/>Automated graph trace review<text:change-end text:change-id="ct-insert-1"/> is required.</text:p>
                <text:p>Legacy process <text:change text:change-id="ct-delete-1"/> was removed.</text:p>
                <text:p>Commented span:
                  <office:annotation office:name="annotation-1">
                    <dc:creator>Grace Hopper</dc:creator>
                    <dc:date>2026-03-01T09:00:00Z</dc:date>
                    <text:p>Confirm the OpenDocument evidence path.</text:p>
                  </office:annotation>
                  OpenDocument evidence path
                  <office:annotation-end office:name="annotation-1"/>
                  is confirmed.
                </text:p>
                <draw:frame draw:name="Deployment Diagram">
                  <draw:image xlink:href="Pictures/deployment.png"><svg:title>Deployment Diagram</svg:title></draw:image>
                </draw:frame>
                <table:table table:name="Owners">
                  <table:table-row>
                    <table:table-cell><text:p>Component</text:p></table:table-cell>
                    <table:table-cell><text:p>Owner</text:p></table:table-cell>
                  </table:table-row>
                  <table:table-row>
                    <table:table-cell><text:p>Uploader</text:p></table:table-cell>
                    <table:table-cell><text:p>Ada Lovelace</text:p></table:table-cell>
                  </table:table-row>
                </table:table>
              </office:text></office:body>
            </office:document-content>
            """,
        )
        archive.writestr(
            "meta.xml",
            """
            <office:document-meta
              xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
              xmlns:dc="http://purl.org/dc/elements/1.1/"
              xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0">
              <office:meta>
                <dc:title>OpenDocument Deployment Plan</dc:title>
                <dc:creator>Ada Lovelace</dc:creator>
                <meta:initial-creator>Grace Hopper</meta:initial-creator>
                <meta:keyword>GraphRAG</meta:keyword>
                <meta:keyword>OpenDocument</meta:keyword>
                <meta:generator>LibreOffice</meta:generator>
                <meta:document-statistic meta:table-count="1" meta:image-count="1" meta:word-count="12"/>
              </office:meta>
            </office:document-meta>
            """,
        )
        archive.writestr("Pictures/deployment.png", _tiny_png_bytes())
    return buffer.getvalue()


def _minimal_odp_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.presentation")
        archive.writestr(
            "content.xml",
            """
            <office:document-content
              xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
              xmlns:form="urn:oasis:names:tc:opendocument:xmlns:form:1.0"
              xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
              xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
              xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"
              xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
              xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"
              xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
              xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
              xmlns:xlink="http://www.w3.org/1999/xlink">
              <office:automatic-styles>
                <style:style style:name="SlideStrong" style:family="text">
                  <style:text-properties fo:font-weight="bold" fo:color="#126B5D"/>
                </style:style>
              </office:automatic-styles>
              <office:body><office:presentation>
                <draw:page draw:name="Launch Overview">
                  <draw:frame><draw:text-box>
                    <text:p>Launch Overview</text:p>
                    <text:p>ODP launch evidence links <text:a xlink:href="https://example.com/odp-source">Launch source</text:a> to <text:span text:style-name="SlideStrong">regional rollout</text:span>.</text:p>
                  </draw:text-box></draw:frame>
                  <table:table table:name="Milestones">
                    <table:table-row>
                      <table:table-cell><text:p>Milestone</text:p></table:table-cell>
                      <table:table-cell><text:p>Owner</text:p></table:table-cell>
                    </table:table-row>
                    <table:table-row>
                      <table:table-cell><text:p>Regional Rollout</text:p></table:table-cell>
                      <table:table-cell><text:p>Ada Lovelace</text:p></table:table-cell>
                    </table:table-row>
                    <table:table-row>
                      <table:table-cell table:number-columns-spanned="2" table:style-name="HighlightCell"><text:p>Launch Governance</text:p></table:table-cell>
                      <table:covered-table-cell/>
                    </table:table-row>
                  </table:table>
                  <draw:frame draw:name="Risk Diagram">
                    <draw:image xlink:href="Pictures/odp-risk.png"><svg:title>Risk Diagram</svg:title></draw:image>
                  </draw:frame>
                  <draw:frame draw:name="Risk Burn-down Chart">
                    <draw:object xlink:href="./Object 1"/>
                  </draw:frame>
                  <draw:custom-shape draw:id="shape-demand" draw:name="Demand Node" svg:x="1in" svg:y="4.1in" svg:width="2in" svg:height="0.5in">
                    <svg:title>Regional Demand</svg:title>
                    <svg:desc>Demand signal for launch approval</svg:desc>
                    <text:p>Regional Demand</text:p>
                  </draw:custom-shape>
                  <draw:custom-shape draw:id="shape-approval" draw:name="Approval Gate" svg:x="3.5in" svg:y="4.1in" svg:width="2in" svg:height="0.5in">
                    <svg:title>Approval Gate</svg:title>
                    <svg:desc>Launch governance gate</svg:desc>
                    <text:p>Approval Gate</text:p>
                  </draw:custom-shape>
                  <draw:connector draw:id="connector-approval" draw:name="requires" draw:start-shape="shape-demand" draw:end-shape="shape-approval" svg:x1="3in" svg:y1="4.35in" svg:x2="3.5in" svg:y2="4.35in">
                    <svg:title>requires</svg:title>
                    <svg:desc>Regional demand requires approval gate</svg:desc>
                  </draw:connector>
                  <office:forms>
                    <form:form form:name="Launch Intake">
                      <form:checkbox form:id="approval-required" form:name="Approval Required" form:label="Approval Required" form:current-state="checked"/>
                      <form:listbox form:id="launch-region" form:name="Launch Region" form:label="Launch Region" form:current-value="North America">
                        <form:option form:label="North America" form:value="NA" form:selected="true"/>
                        <form:option form:label="Europe" form:value="EU"/>
                      </form:listbox>
                      <form:text form:id="risk-owner" form:name="Risk Owner" form:label="Risk Owner" form:current-value="Ada Lovelace"/>
                    </form:form>
                  </office:forms>
                  <draw:control draw:control="approval-required"/>
                </draw:page>
                <draw:page draw:name="Hidden Assumptions" presentation:visibility="hidden">
                  <draw:frame><draw:text-box>
                    <text:p>Hidden Assumptions</text:p>
                    <text:p>Confidential ODP assumptions support regional demand.</text:p>
                  </draw:text-box></draw:frame>
                </draw:page>
              </office:presentation></office:body>
            </office:document-content>
            """,
        )
        archive.writestr(
            "meta.xml",
            """
            <office:document-meta
              xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
              xmlns:dc="http://purl.org/dc/elements/1.1/"
              xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0">
              <office:meta>
                <dc:title>OpenDocument Launch Deck</dc:title>
                <dc:creator>Ada Lovelace</dc:creator>
                <meta:keyword>GraphRAG</meta:keyword>
                <meta:keyword>OpenDocument</meta:keyword>
                <meta:generator>LibreOffice Impress</meta:generator>
                <meta:document-statistic meta:slide-count="2" meta:image-count="1"/>
              </office:meta>
            </office:document-meta>
            """,
        )
        archive.writestr(
            "Object 1/content.xml",
            """
            <office:document-content
              xmlns:chart="urn:oasis:names:tc:opendocument:xmlns:chart:1.0"
              xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
              xmlns:dc="http://purl.org/dc/elements/1.1/"
              xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
              xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
              <office:body><office:chart>
                <chart:chart chart:class="chart:bar">
                  <chart:title><text:p>Launch Risk Burn-down</text:p></chart:title>
                  <chart:legend chart:legend-position="end"/>
                  <chart:plot-area table:cell-range-address="RiskBurnDown.A1:RiskBurnDown.B3" chart:data-source-has-labels="both">
                    <chart:axis chart:dimension="x">
                      <chart:title><text:p>Risk stage</text:p></chart:title>
                    </chart:axis>
                    <chart:axis chart:dimension="y">
                      <chart:title><text:p>Open risks</text:p></chart:title>
                    </chart:axis>
                    <chart:series chart:values-cell-range-address="RiskBurnDown.B2:RiskBurnDown.B3" chart:label-cell-address="RiskBurnDown.B1">
                      <chart:domain table:cell-range-address="RiskBurnDown.A2:RiskBurnDown.A3"/>
                    </chart:series>
                  </chart:plot-area>
                  <table:table table:name="RiskBurnDown">
                    <table:table-row>
                      <table:table-cell><text:p>Stage</text:p></table:table-cell>
                      <table:table-cell><text:p>Open Risks</text:p></table:table-cell>
                    </table:table-row>
                    <table:table-row>
                      <table:table-cell><text:p>Discovery</text:p></table:table-cell>
                      <table:table-cell><text:p>8</text:p></table:table-cell>
                    </table:table-row>
                    <table:table-row>
                      <table:table-cell><text:p>Launch</text:p></table:table-cell>
                      <table:table-cell><text:p>3</text:p></table:table-cell>
                    </table:table-row>
                  </table:table>
                </chart:chart>
              </office:chart></office:body>
            </office:document-content>
            """,
        )
        archive.writestr("Pictures/odp-risk.png", _tiny_png_bytes())
    return buffer.getvalue()


def _minimal_ods_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.spreadsheet")
        archive.writestr(
            "content.xml",
            """
            <office:document-content
              xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
              xmlns:dc="http://purl.org/dc/elements/1.1/"
              xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
              xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
              <office:body><office:spreadsheet>
                <table:table table:name="Scorecard">
                  <table:table-row>
                    <table:table-cell office:value-type="string"><text:p>Metric</text:p></table:table-cell>
                    <table:table-cell office:value-type="string"><text:p>Owner</text:p></table:table-cell>
                    <table:table-cell office:value-type="string"><text:p>Score</text:p></table:table-cell>
                  </table:table-row>
                  <table:table-row>
                    <table:table-cell office:value-type="string"><text:p>GraphRAG</text:p></table:table-cell>
                    <table:table-cell office:value-type="string">
                      <text:p>Ada Lovelace</text:p>
                      <office:annotation office:name="owner-comment-1">
                        <dc:creator>Grace Hopper</dc:creator>
                        <dc:date>2026-04-02T12:00:00Z</dc:date>
                        <text:p>Confirm owner before graph publication.</text:p>
                      </office:annotation>
                    </table:table-cell>
                    <table:table-cell office:value-type="float" office:value="18" table:formula="of:=SUM([.C3];[.C4])"><text:p>18</text:p></table:table-cell>
                  </table:table-row>
                </table:table>
              </office:spreadsheet></office:body>
            </office:document-content>
            """,
        )
        archive.writestr(
            "meta.xml",
            """
            <office:document-meta
              xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
              xmlns:dc="http://purl.org/dc/elements/1.1/"
              xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0">
              <office:meta>
                <dc:title>OpenDocument Quality Scorecard</dc:title>
                <dc:creator>Ada Lovelace</dc:creator>
                <meta:keyword>GraphRAG</meta:keyword>
                <meta:generator>LibreOffice Calc</meta:generator>
              </office:meta>
            </office:document-meta>
            """,
        )
    return buffer.getvalue()


def _tiny_png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x01\x01\x01\x00"
        b"\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _minimal_chart_workbook_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            """
            <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
              <sheets><sheet name="ChartData" sheetId="1" r:id="rId1"/></sheets>
            </workbook>
            """,
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
            </Relationships>
            """,
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            """
            <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <si><t>Metric</t></si>
              <si><t>Source Owner</t></si>
              <si><t>Confidence</t></si>
              <si><t>Forecast Risk</t></si>
              <si><t>Ada Lovelace</t></si>
            </sst>
            """,
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """
            <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <sheetData>
                <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>
                <row r="2"><c r="A2" t="s"><v>3</v></c><c r="B2" t="s"><v>4</v></c><c r="C2"><f>SUM(0.4,0.52)</f><v>0.92</v></c></row>
              </sheetData>
            </worksheet>
            """,
        )
    return buffer.getvalue()


def _minimal_embedded_package_workbook_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            """
            <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
              <sheets><sheet name="Operations" sheetId="1" r:id="rId1"/></sheets>
            </workbook>
            """,
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
            </Relationships>
            """,
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            """
            <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <si><t>Artifact</t></si>
              <si><t>Owner</t></si>
              <si><t>Status</t></si>
              <si><t>Operational Workbook</t></si>
              <si><t>Grace Hopper</t></si>
              <si><t>Ready</t></si>
            </sst>
            """,
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """
            <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <sheetData>
                <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>
                <row r="2"><c r="A2" t="s"><v>3</v></c><c r="B2" t="s"><v>4</v></c><c r="C2" t="s"><v>5</v></c></row>
              </sheetData>
            </worksheet>
            """,
        )
    return buffer.getvalue()


def _minimal_pptx_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "docProps/core.xml",
            """
            <cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                               xmlns:dc="http://purl.org/dc/elements/1.1/"
                               xmlns:dcterms="http://purl.org/dc/terms/"
                               xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
              <dc:title>GraphRAG Revenue Strategy</dc:title>
              <dc:creator>Ada Lovelace</dc:creator>
              <cp:keywords>GraphRAG; Revenue; Risk</cp:keywords>
              <dc:subject>Knowledge graph launch plan</dc:subject>
              <cp:lastModifiedBy>Grace Hopper</cp:lastModifiedBy>
              <dcterms:created xsi:type="dcterms:W3CDTF">2026-01-15T10:00:00Z</dcterms:created>
            </cp:coreProperties>
            """,
        )
        archive.writestr(
            "docProps/app.xml",
            """
            <Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
              <Application>Microsoft PowerPoint</Application>
              <Slides>2</Slides>
              <HiddenSlides>1</HiddenSlides>
              <Company>Llama Nexus Labs</Company>
            </Properties>
            """,
        )
        archive.writestr(
            "ppt/presentation.xml",
            """
            <p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
              <p:sldIdLst>
                <p:sldId id="256" r:id="rIdSlide1"/>
                <p:sldId id="257" r:id="rIdSlide2"/>
              </p:sldIdLst>
              <p:sldSz cx="9144000" cy="5143500"/>
            </p:presentation>
            """,
        )
        archive.writestr(
            "ppt/_rels/presentation.xml.rels",
            """
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rIdSlide1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
              <Relationship Id="rIdSlide2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide2.xml"/>
            </Relationships>
            """,
        )
        archive.writestr(
            "ppt/slides/slide1.xml",
            """
	            <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
	                   xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
	                   xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"
	                   xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram"
	                   xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
              <p:cSld><p:spTree>
                <p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
                  <p:spPr><a:xfrm><a:off x="457200" y="274320"/><a:ext cx="8229600" cy="548640"/></a:xfrm></p:spPr>
                  <p:txBody><a:p><a:r><a:t>Revenue Architecture</a:t></a:r></a:p></p:txBody>
                </p:sp>
		                <p:sp><p:spPr><a:xfrm><a:off x="914400" y="914400"/><a:ext cx="4572000" cy="457200"/></a:xfrm></p:spPr><p:txBody><a:p><a:r><a:t>Northwind connects product revenue to regional demand.</a:t></a:r></a:p></p:txBody></p:sp>
			                <p:sp><p:spPr><a:xfrm><a:off x="914400" y="1371600"/><a:ext cx="1828800" cy="457200"/></a:xfrm></p:spPr><p:txBody><a:p><a:r><a:rPr b="1"/><a:t>Forecast Risk</a:t></a:r></a:p></p:txBody></p:sp>
			                <p:sp><p:spPr><a:xfrm><a:off x="914400" y="1828800"/><a:ext cx="1828800" cy="457200"/></a:xfrm></p:spPr><p:txBody><a:p><a:r><a:rPr><a:hlinkClick r:id="rIdLink1"/></a:rPr><a:t>Source Brief</a:t></a:r></a:p></p:txBody></p:sp>
			                <p:sp>
			                  <p:nvSpPr><p:cNvPr id="20" name="Product Revenue Node" title="Product Revenue"/></p:nvSpPr>
			                  <p:spPr><a:xfrm><a:off x="914400" y="2011680"/><a:ext cx="1828800" cy="548640"/></a:xfrm></p:spPr>
			                  <p:txBody><a:p><a:r><a:t>Product Revenue</a:t></a:r></a:p></p:txBody>
			                </p:sp>
			                <p:sp>
			                  <p:nvSpPr><p:cNvPr id="21" name="Regional Demand Node" title="Regional Demand"/></p:nvSpPr>
			                  <p:spPr><a:xfrm><a:off x="3657600" y="2011680"/><a:ext cx="1828800" cy="548640"/></a:xfrm></p:spPr>
			                  <p:txBody><a:p><a:r><a:t>Regional Demand</a:t></a:r></a:p></p:txBody>
			                </p:sp>
			                <p:cxnSp>
			                  <p:nvCxnSpPr>
			                    <p:cNvPr id="60" name="Demand Driver Connector" title="drives" descr="Product revenue drives regional demand"/>
			                    <p:cNvCxnSpPr><a:stCxn id="20" idx="3"/><a:endCxn id="21" idx="1"/></p:cNvCxnSpPr>
			                  </p:nvCxnSpPr>
			                  <p:spPr><a:xfrm><a:off x="2743200" y="2194560"/><a:ext cx="914400" cy="274320"/></a:xfrm></p:spPr>
			                </p:cxnSp>
			                <p:cxnSp>
			                  <p:nvCxnSpPr>
			                    <p:cNvPr id="61" name="Risk Map Connector" title="references" descr="Regional demand references the risk map"/>
			                    <p:cNvCxnSpPr><a:stCxn id="21" idx="3"/><a:endCxn id="4" idx="1"/></p:cNvCxnSpPr>
			                  </p:nvCxnSpPr>
			                  <p:spPr><a:xfrm><a:off x="5486400" y="1828800"/><a:ext cx="914400" cy="548640"/></a:xfrm></p:spPr>
			                </p:cxnSp>
			                <p:sp>
			                  <p:nvSpPr><p:cNvPr id="44" name="Risk Narrative Callout" title="Risk Narrative" descr="Callout shape describing regional launch blockers"/></p:nvSpPr>
			                  <p:spPr><a:xfrm><a:off x="6400800" y="2514600"/><a:ext cx="1828800" cy="685800"/></a:xfrm></p:spPr>
			                  <p:txBody><a:p/></p:txBody>
			                </p:sp>
			                <p:graphicFrame><p:xfrm><a:off x="914400" y="2834640"/><a:ext cx="3657600" cy="1280160"/></p:xfrm><a:graphic><a:graphicData><c:chart r:id="rId5"/></a:graphicData></a:graphic></p:graphicFrame>
			                <p:graphicFrame><p:xfrm><a:off x="5029200" y="2743200"/><a:ext cx="914400" cy="914400"/></p:xfrm><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/diagram">
			                  <dgm:relIds r:dm="rIdDiagram1"/>
			                </a:graphicData></a:graphic></p:graphicFrame>
			                <p:graphicFrame><p:xfrm><a:off x="914400" y="4206240"/><a:ext cx="3657600" cy="731520"/></p:xfrm><a:graphic><a:graphicData><a:tbl>
		                  <a:tr><a:tc><a:txBody><a:p><a:r><a:rPr b="1"/><a:t>Product</a:t></a:r></a:p></a:txBody></a:tc>
		                        <a:tc><a:txBody><a:p><a:r><a:t>Region</a:t></a:r></a:p></a:txBody></a:tc></a:tr>
	                  <a:tr><a:tc><a:txBody><a:p><a:r><a:t>GraphRAG</a:t></a:r></a:p></a:txBody></a:tc>
	                        <a:tc><a:txBody><a:p><a:r><a:t>North America</a:t></a:r></a:p></a:txBody></a:tc></a:tr>
	                  <a:tr><a:tc gridSpan="2">
	                        <a:tcPr anchor="ctr" marL="91440" marR="91440" marT="45720" marB="45720">
	                          <a:solidFill><a:srgbClr val="FFF2CC"/></a:solidFill>
	                          <a:lnL w="12700"><a:solidFill><a:srgbClr val="D6B656"/></a:solidFill></a:lnL>
	                          <a:lnR w="12700"><a:solidFill><a:srgbClr val="D6B656"/></a:solidFill></a:lnR>
	                          <a:lnT w="12700"><a:solidFill><a:srgbClr val="D6B656"/></a:solidFill></a:lnT>
	                          <a:lnB w="12700"><a:solidFill><a:srgbClr val="D6B656"/></a:solidFill></a:lnB>
	                        </a:tcPr>
	                        <a:txBody><a:p><a:r><a:rPr b="1" i="1"><a:solidFill><a:srgbClr val="7F6000"/></a:solidFill></a:rPr><a:t>Strategic Notes</a:t></a:r></a:p></a:txBody></a:tc>
	                        <a:tc hMerge="1"><a:txBody><a:p/></a:txBody></a:tc></a:tr>
	                </a:tbl></a:graphicData></a:graphic></p:graphicFrame>
	                <p:pic>
	                  <p:nvPicPr><p:cNvPr id="4" name="risk-map.png" title="Risk Map" descr="Regional risk heatmap">
	                    <a:hlinkClick r:id="rIdImageLink" tooltip="Open risk register"/>
	                  </p:cNvPr></p:nvPicPr>
	                  <p:blipFill><a:blip r:embed="rId9" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/></p:blipFill>
	                  <p:spPr><a:xfrm><a:off x="6400800" y="914400"/><a:ext cx="1828800" cy="1371600"/></a:xfrm></p:spPr>
	                </p:pic>
	              </p:spTree>
	              <p:controls>
	                <p:control name="LaunchApproval" r:id="rIdControl1" spid="_x0000_s1025" showAsIcon="0"/>
	                <p:control name="LaunchRegion" r:id="rIdControl2" spid="_x0000_s1026" showAsIcon="0"/>
	              </p:controls>
	              </p:cSld>
		            </p:sld>
		            """,
        )
        archive.writestr(
            "ppt/slides/_rels/slide1.xml.rels",
            """
	            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
	              <Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" Target="../charts/chart1.xml"/>
	              <Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/risk-map.png"/>
	              <Relationship Id="rIdLink1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.com/source-brief" TargetMode="External"/>
	              <Relationship Id="rIdImageLink" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.com/risk-register" TargetMode="External"/>
	              <Relationship Id="rIdLayout1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
	              <Relationship Id="rIdComments1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments/comment1.xml"/>
		              <Relationship Id="rIdNotes1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/>
		              <Relationship Id="rIdDiagram1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data1.xml"/>
		              <Relationship Id="rIdAudio1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/audio" Target="../media/narration.mp3"/>
	              <Relationship Id="rIdVideo1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/video" Target="https://example.com/launch-demo.mp4" TargetMode="External"/>
	              <Relationship Id="rIdOle1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/oleObject" Target="../embeddings/oleObject1.bin"/>
		              <Relationship Id="rIdPackage1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/package" Target="../embeddings/operations.xlsx"/>
	              <Relationship Id="rIdControl1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/control" Target="../activeX/activeX1.xml"/>
	              <Relationship Id="rIdControl2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/control" Target="../activeX/activeX2.xml"/>
	            </Relationships>
	            """,
	        )
        archive.writestr(
            "ppt/slides/slide2.xml",
            """
            <p:sld show="0" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                   xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                   xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
              <p:cSld><p:spTree>
                <p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
                  <p:txBody><a:p><a:r><a:t>Hidden Assumptions</a:t></a:r></a:p></p:txBody>
                </p:sp>
                <p:sp><p:txBody><a:p><a:r><a:t>Confidential baseline assumptions support regional demand.</a:t></a:r></a:p></p:txBody></p:sp>
              </p:spTree></p:cSld>
            </p:sld>
            """,
        )
        archive.writestr(
            "ppt/slides/_rels/slide2.xml.rels",
            """
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
            """,
        )
        archive.writestr(
            "ppt/slideLayouts/slideLayout1.xml",
            """
            <p:sldLayout xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                         xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                         xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
              <p:cSld><p:spTree>
                <p:sp>
                  <p:nvSpPr><p:cNvPr id="101" name="Layout Governance Placeholder" title="Layout Governance"/><p:nvPr><p:ph type="body"/></p:nvPr></p:nvSpPr>
                  <p:spPr><a:xfrm><a:off x="457200" y="4572000"/><a:ext cx="3657600" cy="457200"/></a:xfrm></p:spPr>
                  <p:txBody><a:p><a:r><a:rPr b="1"/><a:t>Template Governance Note</a:t></a:r></a:p></p:txBody>
                </p:sp>
                <p:pic>
                  <p:nvPicPr><p:cNvPr id="102" name="layout-logo.png" title="Llama Nexus Logo" descr="Inherited brand logo">
                    <a:hlinkClick r:id="rIdLayoutLogoLink" tooltip="Open brand kit"/>
                  </p:cNvPr></p:nvPicPr>
                  <p:blipFill><a:blip r:embed="rIdLayoutLogo"/></p:blipFill>
                  <p:spPr><a:xfrm><a:off x="7772400" y="228600"/><a:ext cx="914400" cy="457200"/></a:xfrm></p:spPr>
                </p:pic>
              </p:spTree></p:cSld>
            </p:sldLayout>
            """,
        )
        archive.writestr(
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
            """
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rIdLayoutLogo" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/layout-logo.png"/>
              <Relationship Id="rIdLayoutLogoLink" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.com/brand-kit" TargetMode="External"/>
              <Relationship Id="rIdMaster1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
            </Relationships>
            """,
        )
        archive.writestr(
            "ppt/slideMasters/slideMaster1.xml",
            """
            <p:sldMaster xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                         xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
              <p:cSld><p:spTree>
                <p:sp>
                  <p:nvSpPr><p:cNvPr id="201" name="Master Footer" title="Global Footer" descr="Graph evidence required on every slide"/></p:nvSpPr>
                  <p:spPr><a:xfrm><a:off x="457200" y="4800600"/><a:ext cx="8229600" cy="228600"/></a:xfrm></p:spPr>
                  <p:txBody><a:p><a:r><a:t>Global Footer: graph evidence required</a:t></a:r></a:p></p:txBody>
                </p:sp>
              </p:spTree></p:cSld>
            </p:sldMaster>
            """,
        )
        archive.writestr(
            "ppt/activeX/activeX1.xml",
            """
            <ax:ocx xmlns:ax="http://schemas.microsoft.com/office/2006/activeX"
                    ax:name="LaunchApproval"
                    ax:progId="Forms.CheckBox.1">
              <ax:ocxPr ax:name="Caption" ax:value="Approved for launch"/>
              <ax:ocxPr ax:name="Value" ax:value="True"/>
            </ax:ocx>
            """,
        )
        archive.writestr(
            "ppt/activeX/activeX2.xml",
            """
            <ax:ocx xmlns:ax="http://schemas.microsoft.com/office/2006/activeX"
                    ax:name="LaunchRegion"
                    ax:progId="Forms.ComboBox.1">
              <ax:ocxPr ax:name="Caption" ax:value="Launch region"/>
              <ax:ocxPr ax:name="List" ax:value="North America;Europe;APAC"/>
              <ax:ocxPr ax:name="Value" ax:value="Europe"/>
              <ax:ocxPr ax:name="ListIndex" ax:value="1"/>
            </ax:ocx>
            """,
        )
        archive.writestr(
            "ppt/comments/comment1.xml",
            """
            <p:cmLst xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
              <p:cm id="1" authorId="0" dt="2026-01-15T10:30:00Z">
                <p:pos x="1828800" y="914400"/>
                <p:t>Validate regional risk narrative.</p:t>
              </p:cm>
            </p:cmLst>
            """,
        )
        archive.writestr(
            "ppt/commentAuthors.xml",
            """
            <p:cmAuthorLst xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
              <p:cmAuthor id="0" name="Ada Lovelace" initials="AL"/>
            </p:cmAuthorLst>
            """,
        )
        archive.writestr(
            "ppt/notesSlides/notesSlide1.xml",
            """
            <p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                     xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
              <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r>
                <a:t>Speaker note: emphasize graph-grounded launch path.</a:t>
              </a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
            </p:notes>
            """,
        )
        archive.writestr(
            "ppt/charts/chart1.xml",
            """
            <c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"
                          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                          xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
              <c:chart>
                <c:title><c:tx><c:rich><a:p><a:r><a:t>Regional Revenue Forecast</a:t></a:r></a:p></c:rich></c:tx></c:title>
                <c:plotArea>
                  <c:barChart>
                    <c:ser>
                      <c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>Revenue</c:v></c:pt></c:strCache></c:strRef></c:tx>
                      <c:cat><c:strRef><c:strCache><c:pt idx="0"><c:v>Q1</c:v></c:pt><c:pt idx="1"><c:v>Q2</c:v></c:pt></c:strCache></c:strRef></c:cat>
                      <c:val><c:numRef><c:numCache><c:pt idx="0"><c:v>12</c:v></c:pt><c:pt idx="1"><c:v>18</c:v></c:pt></c:numCache></c:numRef></c:val>
                      <c:spPr>
                        <a:solidFill><a:srgbClr val="5B9BD5"/></a:solidFill>
                        <a:ln w="25400"><a:solidFill><a:srgbClr val="1F4E79"/></a:solidFill><a:prstDash val="solid"/></a:ln>
                      </c:spPr>
                      <c:dPt>
                        <c:idx val="1"/>
                        <c:spPr><a:solidFill><a:srgbClr val="C00000"/></a:solidFill></c:spPr>
                      </c:dPt>
                      <c:marker>
                        <c:symbol val="circle"/>
                        <c:size val="7"/>
                        <c:spPr><a:solidFill><a:srgbClr val="70AD47"/></a:solidFill></c:spPr>
                      </c:marker>
                      <c:dLbls><c:dLblPos val="outEnd"/><c:numFmt formatCode="$#,##0.0"/><c:showSerName val="1"/><c:showCatName val="1"/><c:showVal val="1"/></c:dLbls>
                      <c:trendline>
                        <c:name val="Revenue trend"/>
                        <c:spPr><a:ln w="19050"><a:solidFill><a:srgbClr val="7030A0"/></a:solidFill><a:prstDash val="dash"/></a:ln></c:spPr>
                        <c:trendlineType val="linear"/>
                        <c:dispEq val="1"/>
                        <c:dispRSqr val="1"/>
                      </c:trendline>
                    </c:ser>
                    <c:axId val="100"/>
                    <c:axId val="200"/>
                  </c:barChart>
                  <c:catAx>
                    <c:axId val="100"/>
                    <c:axPos val="b"/>
                    <c:title><c:tx><c:rich><a:p><a:r><a:t>Quarter</a:t></a:r></a:p></c:rich></c:tx></c:title>
                  </c:catAx>
                  <c:valAx>
                    <c:axId val="200"/>
                    <c:axPos val="l"/>
                    <c:title><c:tx><c:rich><a:p><a:r><a:t>Revenue ($M)</a:t></a:r></a:p></c:rich></c:tx></c:title>
                    <c:numFmt formatCode="$#,##0"/>
                  </c:valAx>
                </c:plotArea>
                <c:legend><c:legendPos val="r"/><c:overlay val="0"/></c:legend>
              </c:chart>
              <c:externalData r:id="rIdChartWorkbook"/>
            </c:chartSpace>
            """,
        )
        archive.writestr(
            "ppt/charts/_rels/chart1.xml.rels",
            """
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rIdChartWorkbook" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/package" Target="../embeddings/chartWorkbook.xlsx"/>
            </Relationships>
            """,
        )
        archive.writestr(
            "ppt/diagrams/data1.xml",
            """
            <dgm:dataModel xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram"
                           xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
              <dgm:ptLst>
                <dgm:pt modelId="root"><dgm:t><a:p><a:r><a:t>Launch Plan</a:t></a:r></a:p></dgm:t></dgm:pt>
                <dgm:pt modelId="risk"><dgm:t><a:p><a:r><a:t>Risk Review</a:t></a:r></a:p></dgm:t></dgm:pt>
                <dgm:pt modelId="rollout"><dgm:t><a:p><a:r><a:t>Regional Rollout</a:t></a:r></a:p></dgm:t></dgm:pt>
              </dgm:ptLst>
              <dgm:cxnLst>
                <dgm:cxn modelId="c1" type="parOf" srcId="root" destId="risk"/>
                <dgm:cxn modelId="c2" type="parOf" srcId="root" destId="rollout"/>
              </dgm:cxnLst>
            </dgm:dataModel>
            """,
        )
        archive.writestr("ppt/media/risk-map.png", _tiny_png_bytes())
        archive.writestr("ppt/media/layout-logo.png", _tiny_png_bytes())
        archive.writestr("ppt/media/narration.mp3", b"ID3\x04\x00\x00\x00\x00\x00\x21GraphRAG narration cue")
        archive.writestr("ppt/embeddings/chartWorkbook.xlsx", _minimal_chart_workbook_bytes())
        archive.writestr("ppt/embeddings/oleObject1.bin", b"embedded operational workbook placeholder")
        archive.writestr("ppt/embeddings/operations.xlsx", _minimal_embedded_package_workbook_bytes())
    return buffer.getvalue()


def _minimal_xlsx_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "docProps/core.xml",
            """
            <cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                               xmlns:dc="http://purl.org/dc/elements/1.1/">
              <dc:title>GraphRAG Quality Scorecard</dc:title>
              <dc:creator>Ada Lovelace</dc:creator>
              <cp:keywords>GraphRAG; Quality</cp:keywords>
            </cp:coreProperties>
            """,
        )
        archive.writestr(
            "docProps/app.xml",
            """
            <Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
              <Application>Microsoft Excel</Application>
              <Company>Llama Nexus Labs</Company>
            </Properties>
            """,
        )
        archive.writestr(
            "xl/workbook.xml",
            """
            <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
              <sheets>
                <sheet name="Scorecard" sheetId="1" r:id="rId1" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>
                <sheet name="RawData" sheetId="2" state="hidden" r:id="rId2" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>
              </sheets>
              <pivotCaches><pivotCache cacheId="1" r:id="rIdPivotCache1"/></pivotCaches>
              <definedNames>
                <definedName name="QualityOwners" localSheetId="0">Scorecard!$B$2:$B$20</definedName>
                <definedName name="ScoreTableRange">Scorecard!$A$1:$C$2</definedName>
              </definedNames>
            </workbook>
            """,
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
              <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
              <Relationship Id="rIdPivotCache1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/pivotCacheDefinition" Target="pivotCache/pivotCacheDefinition1.xml"/>
            </Relationships>
            """,
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            """
            <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <si><t>Metric</t></si><si><t>Owner</t></si><si><t>Score</t></si>
              <si><t>GraphRAG</t></si><si><t>Ada Lovelace</t></si><si><t>Deployment Gate</t></si>
            </sst>
            """,
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """
            <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
              <sheetData>
                <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c><c r="D1" t="s"><v>5</v></c></row>
                <row r="2"><c r="A2" t="s"><v>3</v></c><c r="B2" t="s"><v>4</v></c><c r="C2"><f>SUM(10,8)</f><v>18</v></c><c r="D2"><f>C2+2</f><v>20</v></c><c r="E2"><f>COUNTA(QualityOwners)</f><v>1</v></c></row>
              </sheetData>
              <mergeCells count="1"><mergeCell ref="D1:F1"/></mergeCells>
              <conditionalFormatting sqref="C2:C20">
                <cfRule type="cellIs" priority="1" operator="greaterThanOrEqual" dxfId="0">
                  <formula>18</formula>
                </cfRule>
              </conditionalFormatting>
              <conditionalFormatting sqref="D2:D20">
                <cfRule type="expression" priority="2">
                  <formula>D2&gt;=C2</formula>
                </cfRule>
              </conditionalFormatting>
              <dataValidations count="1">
                <dataValidation type="list" allowBlank="1" showErrorMessage="1" sqref="B2:B20"
                                promptTitle="Owner" prompt="Choose the accountable owner"
                                errorTitle="Invalid owner" error="Use an approved owner">
                  <formula1>"Ada Lovelace,Grace Hopper,Katherine Johnson"</formula1>
                </dataValidation>
              </dataValidations>
              <hyperlinks><hyperlink ref="A2" r:id="rIdHyperlink1" display="GraphRAG scorecard source"/></hyperlinks>
              <tableParts count="1"><tablePart r:id="rIdTable1"/></tableParts>
              <pivotTableDefinitions count="1"><pivotTableDefinition r:id="rIdPivot1"/></pivotTableDefinitions>
              <drawing r:id="rIdDrawing1"/>
            </worksheet>
            """,
        )
        archive.writestr(
            "xl/worksheets/sheet2.xml",
            """
            <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <sheetData>
                <row r="1">
                  <c r="A1" t="inlineStr"><is><t>Assumption</t></is></c>
                  <c r="B1" t="inlineStr"><is><t>Source</t></is></c>
                </row>
                <row r="2">
                  <c r="A2" t="inlineStr"><is><t>Confidential Baseline</t></is></c>
                  <c r="B2" t="inlineStr"><is><t>Hidden source notes</t></is></c>
                </row>
              </sheetData>
            </worksheet>
            """,
        )
        archive.writestr(
            "xl/worksheets/_rels/sheet1.xml.rels",
            """
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rIdComments1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="../comments1.xml"/>
              <Relationship Id="rIdThreadedComments1" Type="http://schemas.microsoft.com/office/2017/10/relationships/threadedComment" Target="../threadedComments/threadedComment1.xml"/>
              <Relationship Id="rIdTable1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/table" Target="../tables/table1.xml"/>
              <Relationship Id="rIdPivot1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/pivotTable" Target="../pivotTables/pivotTable1.xml"/>
              <Relationship Id="rIdDrawing1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" Target="../drawings/drawing1.xml"/>
              <Relationship Id="rIdHyperlink1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.com/scorecard" TargetMode="External"/>
            </Relationships>
            """,
        )
        archive.writestr(
            "xl/tables/table1.xml",
            """
            <table xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                   id="1" name="OwnerScorecard" displayName="OwnerScorecard" ref="A1:C2">
              <autoFilter ref="A1:C2"/>
              <tableColumns count="3">
                <tableColumn id="1" name="Metric"/>
                <tableColumn id="2" name="Owner"/>
                <tableColumn id="3" name="Score"/>
              </tableColumns>
            </table>
            """,
        )
        archive.writestr(
            "xl/pivotTables/pivotTable1.xml",
            """
            <pivotTableDefinition xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                                  name="QualityPivot" cacheId="1" dataCaption="Values">
              <location ref="H1:J6" firstHeaderRow="1" firstDataRow="2" firstDataCol="2"/>
              <pivotFields count="3">
                <pivotField axis="axisRow" showAll="0"/>
                <pivotField axis="axisCol" showAll="0"/>
                <pivotField dataField="1"/>
              </pivotFields>
              <rowFields count="1"><field x="0"/></rowFields>
              <colFields count="1"><field x="1"/></colFields>
              <dataFields count="1"><dataField name="Average Score" fld="2" subtotal="average"/></dataFields>
            </pivotTableDefinition>
            """,
        )
        archive.writestr(
            "xl/pivotCache/pivotCacheDefinition1.xml",
            """
            <pivotCacheDefinition xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                                  recordCount="1" refreshedBy="Ada Lovelace">
              <cacheSource type="worksheet"><worksheetSource ref="A1:C2" sheet="Scorecard" name="OwnerScorecard"/></cacheSource>
              <cacheFields count="3">
                <cacheField name="Metric"/>
                <cacheField name="Owner"/>
                <cacheField name="Score"/>
              </cacheFields>
            </pivotCacheDefinition>
            """,
        )
        archive.writestr(
            "xl/comments1.xml",
            """
            <comments xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <authors><author>Grace Hopper</author></authors>
              <commentList><comment ref="B2" authorId="0"><text><r><t>Owner requires review</t></r></text></comment></commentList>
            </comments>
            """,
        )
        archive.writestr(
            "xl/persons/person.xml",
            """
            <personList xmlns="http://schemas.microsoft.com/office/spreadsheetml/2018/threadedcomments">
              <person id="person-thread-1" displayName="Katherine Johnson" userId="katherine@example.com" providerId="None"/>
            </personList>
            """,
        )
        archive.writestr(
            "xl/threadedComments/threadedComment1.xml",
            """
            <ThreadedComments xmlns="http://schemas.microsoft.com/office/spreadsheetml/2018/threadedcomments">
              <threadedComment ref="C2" personId="person-thread-1" id="thread-comment-1" dT="2026-01-02T03:04:05Z">
                <text>Score formula needs audit trail</text>
              </threadedComment>
            </ThreadedComments>
            """,
        )
        archive.writestr(
            "xl/drawings/drawing1.xml",
            """
            <xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
                      xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                      xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"
                      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
              <xdr:twoCellAnchor>
                <xdr:graphicFrame><a:graphic><a:graphicData><c:chart r:id="rIdChart1"/></a:graphicData></a:graphic></xdr:graphicFrame>
              </xdr:twoCellAnchor>
              <xdr:twoCellAnchor>
                <xdr:pic>
                  <xdr:nvPicPr><xdr:cNvPr id="7" name="scorecard.png" title="Scorecard Snapshot" descr="GraphRAG scorecard visual"/></xdr:nvPicPr>
                  <xdr:blipFill><a:blip r:embed="rIdImage1"/></xdr:blipFill>
                </xdr:pic>
              </xdr:twoCellAnchor>
            </xdr:wsDr>
            """,
        )
        archive.writestr(
            "xl/drawings/_rels/drawing1.xml.rels",
            """
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rIdChart1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" Target="../charts/chart1.xml"/>
              <Relationship Id="rIdImage1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/scorecard.png"/>
            </Relationships>
            """,
        )
        archive.writestr(
            "xl/charts/chart1.xml",
            """
            <c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart">
              <c:chart><c:plotArea><c:lineChart><c:ser>
                <c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>Quality</c:v></c:pt></c:strCache></c:strRef></c:tx>
                <c:cat><c:strRef><c:strCache><c:pt idx="0"><c:v>Baseline</c:v></c:pt><c:pt idx="1"><c:v>GraphRAG</c:v></c:pt></c:strCache></c:strRef></c:cat>
                <c:val><c:numRef><c:numCache><c:pt idx="0"><c:v>71</c:v></c:pt><c:pt idx="1"><c:v>94</c:v></c:pt></c:numCache></c:numRef></c:val>
              </c:ser></c:lineChart></c:plotArea></c:chart>
            </c:chartSpace>
            """,
        )
        archive.writestr("xl/media/scorecard.png", _tiny_png_bytes())
    return buffer.getvalue()


def _minimal_docx_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "docProps/core.xml",
            """
            <cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                               xmlns:dc="http://purl.org/dc/elements/1.1/">
              <dc:title>GraphRAG Implementation Plan</dc:title>
              <dc:creator>Ada Lovelace</dc:creator>
              <cp:keywords>GraphRAG; Deployment; Governance</cp:keywords>
              <dc:subject>Deployment integration</dc:subject>
            </cp:coreProperties>
            """,
        )
        archive.writestr(
            "docProps/app.xml",
            """
            <Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
              <Application>Microsoft Word</Application>
              <Pages>2</Pages>
              <Company>Llama Nexus Labs</Company>
            </Properties>
            """,
        )
        archive.writestr(
            "word/document.xml",
            """
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                        xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
                        xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                        xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"
                        xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram"
                        xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                        xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
                        xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
                        xmlns:v="urn:schemas-microsoft-com:vml">
              <w:body>
                <w:sdt>
                  <w:sdtPr>
                    <w:alias w:val="Approved"/>
                    <w:tag w:val="approved"/>
                    <w:id w:val="101"/>
                    <w14:checkbox><w14:checked w14:val="1"/></w14:checkbox>
                  </w:sdtPr>
                  <w:sdtContent><w:p><w:r><w:t>Approval Gate</w:t></w:r></w:p></w:sdtContent>
                </w:sdt>
                <w:sdt>
                  <w:sdtPr>
                    <w:alias w:val="Launch Owner"/>
                    <w:tag w:val="launch_owner"/>
                    <w:id w:val="102"/>
                    <w:dropDownList>
                      <w:listItem w:displayText="Operations" w:value="ops"/>
                      <w:listItem w:displayText="Security" w:value="sec"/>
                      <w:listItem w:displayText="Support" w:value="support"/>
                    </w:dropDownList>
                  </w:sdtPr>
                  <w:sdtContent><w:p><w:r><w:t>Operations</w:t></w:r></w:p></w:sdtContent>
                </w:sdt>
                <w:sdt>
                  <w:sdtPr>
                    <w:alias w:val="Review Date"/>
                    <w:tag w:val="review_date"/>
                    <w:id w:val="103"/>
                    <w:date w:fullDate="2026-04-15T00:00:00Z">
                      <w:dateFormat w:val="MMMM d, yyyy"/>
                      <w:calendar w:val="gregorian"/>
                    </w:date>
                  </w:sdtPr>
                  <w:sdtContent><w:p><w:r><w:t>April 15, 2026</w:t></w:r></w:p></w:sdtContent>
                </w:sdt>
                <w:p><w:r><w:drawing>
                  <wp:inline>
                    <wp:docPr id="7" name="architecture.png" title="Architecture Diagram" descr="GraphRAG deployment topology"/>
                    <a:graphic><a:graphicData><a:blip r:embed="rIdImage1"/></a:graphicData></a:graphic>
                  </wp:inline>
                </w:drawing></w:r></w:p>
                <w:p><w:r><w:drawing>
                  <wp:inline>
                    <wp:docPr id="9" name="Launch Metrics Chart" title="Launch Metrics" descr="Quarterly launch readiness metrics"/>
                    <a:graphic>
                      <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart">
                        <c:chart r:id="rIdChart1"/>
                      </a:graphicData>
                    </a:graphic>
                  </wp:inline>
                </w:drawing></w:r></w:p>
                <w:p><w:r><w:drawing>
                  <wp:inline>
                    <wp:docPr id="10" name="Launch Flow Diagram" title="Launch Flow" descr="SmartArt launch dependency flow"/>
                    <a:graphic>
                      <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/diagram">
                        <dgm:relIds r:dm="rIdDiagram1"/>
                      </a:graphicData>
                    </a:graphic>
                  </wp:inline>
                </w:drawing></w:r></w:p>
                <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:rPr><w:b/></w:rPr><w:t>Implementation Plan</w:t></w:r></w:p>
                <w:p><w:r><w:drawing>
                  <wp:inline>
                    <wp:docPr id="8" name="Risk Callout Box" title="Risk Callout" descr="Text box captures launch blockers"/>
                    <a:graphic>
                      <a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">
                        <wps:wsp>
                          <wps:txbx>
                            <w:txbxContent>
                              <w:p><w:r><w:rPr><w:b/></w:rPr><w:t>Risk callout: launch blockers require graph trace</w:t></w:r></w:p>
                            </w:txbxContent>
                          </wps:txbx>
                        </wps:wsp>
                      </a:graphicData>
                    </a:graphic>
                  </wp:inline>
                </w:drawing></w:r></w:p>
                <w:p><w:r><w:pict>
                  <v:shape id="LegacyDecisionBox" title="Legacy Decision Box" alt="Legacy text box preserves source links">
                    <v:textbox>
                      <w:txbxContent>
                        <w:p><w:r><w:t>Legacy decision note: preserve source links</w:t></w:r></w:p>
                      </w:txbxContent>
                    </v:textbox>
                  </v:shape>
                </w:pict></w:r></w:p>
                <w:p><w:hyperlink r:id="rIdLink1"><w:r><w:t>GraphRAG design brief</w:t></w:r></w:hyperlink></w:p>
                <w:p><w:r><w:t>Architecture note</w:t></w:r><w:r><w:commentReference w:id="0"/></w:r></w:p>
                <w:p><w:ins w:id="11" w:author="Grace Hopper" w:date="2026-02-01T10:00:00Z"><w:r><w:t>Inserted launch checklist</w:t></w:r></w:ins></w:p>
                <w:p><w:del w:id="12" w:author="Alan Turing" w:date="2026-02-02T11:30:00Z"><w:r><w:delText>Deprecated manual review</w:delText></w:r></w:del></w:p>
                <w:p><w:r><w:t>Deployment evidence</w:t></w:r><w:r><w:footnoteReference w:id="2"/></w:r><w:r><w:endnoteReference w:id="3"/></w:r></w:p>
                <w:tbl>
                  <w:tr><w:tc><w:p><w:r><w:t>Component</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Owner</w:t></w:r></w:p></w:tc></w:tr>
                  <w:tr><w:tc><w:p><w:r><w:t>Extractor</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Llama Nexus</w:t></w:r></w:p></w:tc></w:tr>
                  <w:tr><w:tc>
                    <w:tcPr>
                      <w:tcStyle w:val="GovernanceCell"/>
                      <w:tcW w:w="4320" w:type="dxa"/>
                      <w:gridSpan w:val="2"/>
                      <w:vAlign w:val="center"/>
                      <w:textDirection w:val="tbRl"/>
                      <w:shd w:val="pct20" w:fill="FFF2CC"/>
                      <w:tcMar>
                        <w:left w:w="144" w:type="dxa"/>
                        <w:right w:w="288" w:type="dxa"/>
                        <w:top w:w="72" w:type="dxa"/>
                        <w:bottom w:w="72" w:type="dxa"/>
                      </w:tcMar>
                      <w:tcBorders>
                        <w:top w:val="single" w:sz="8" w:color="4472C4"/>
                        <w:bottom w:val="single" w:sz="12" w:color="70AD47"/>
                      </w:tcBorders>
                    </w:tcPr>
                    <w:p><w:r><w:rPr><w:b/><w:i/></w:rPr><w:t>Governance checkpoint</w:t></w:r></w:p>
                  </w:tc></w:tr>
                  <w:tr><w:tc><w:tcPr><w:vMerge w:val="restart"/></w:tcPr><w:p><w:r><w:t>Risk Register</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Open</w:t></w:r></w:p></w:tc></w:tr>
                  <w:tr><w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p/></w:tc><w:tc><w:p><w:r><w:t>Mitigated</w:t></w:r></w:p></w:tc></w:tr>
                </w:tbl>
              </w:body>
            </w:document>
            """,
        )
        archive.writestr(
            "word/_rels/document.xml.rels",
            """
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rIdImage1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/architecture.png"/>
              <Relationship Id="rIdChart1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" Target="charts/chart1.xml"/>
              <Relationship Id="rIdDiagram1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="diagrams/data1.xml"/>
              <Relationship Id="rIdLink1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.com/graphrag-design" TargetMode="External"/>
              <Relationship Id="rIdHeader1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>
              <Relationship Id="rIdFooter1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>
            </Relationships>
            """,
        )
        archive.writestr(
            "word/comments.xml",
            """
            <w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:comment w:id="0" w:author="Ada Lovelace" w:initials="AL">
                <w:p><w:r><w:t>Review the GraphRAG extraction boundary.</w:t></w:r></w:p>
              </w:comment>
            </w:comments>
            """,
        )
        archive.writestr(
            "word/footnotes.xml",
            """
            <w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:footnote w:id="2"><w:p><w:r><w:t>Footnote captures deployment caveats.</w:t></w:r></w:p></w:footnote>
            </w:footnotes>
            """,
        )
        archive.writestr(
            "word/endnotes.xml",
            """
            <w:endnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:endnote w:id="3"><w:p><w:r><w:t>Endnote lists governance owners.</w:t></w:r></w:p></w:endnote>
            </w:endnotes>
            """,
        )
        archive.writestr(
            "word/header1.xml",
            """
            <w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:p><w:r><w:t>Confidential Architecture Header</w:t></w:r></w:p>
            </w:hdr>
            """,
        )
        archive.writestr(
            "word/footer1.xml",
            """
            <w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:p><w:r><w:t>Footer retention owner: Knowledge Platform</w:t></w:r></w:p>
            </w:ftr>
            """,
        )
        archive.writestr("word/media/architecture.png", _tiny_png_bytes())
        archive.writestr(
            "word/charts/chart1.xml",
            """
            <c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"
                          xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
              <c:chart>
                <c:title><c:tx><c:rich><a:p><a:r><a:t>Launch Metrics</a:t></a:r></a:p></c:rich></c:tx></c:title>
                <c:plotArea>
                  <c:barChart>
                    <c:ser>
                      <c:idx val="0"/>
                      <c:order val="0"/>
                      <c:tx><c:v>Readiness</c:v></c:tx>
                      <c:spPr><a:solidFill><a:srgbClr val="4472C4"/></a:solidFill></c:spPr>
                      <c:cat><c:strLit><c:pt idx="0"><c:v>Q1</c:v></c:pt><c:pt idx="1"><c:v>Q2</c:v></c:pt></c:strLit></c:cat>
                      <c:val><c:numLit><c:pt idx="0"><c:v>72</c:v></c:pt><c:pt idx="1"><c:v>88</c:v></c:pt></c:numLit></c:val>
                    </c:ser>
                  </c:barChart>
                  <c:catAx><c:axId val="10"/><c:axPos val="b"/><c:title><c:tx><c:rich><a:p><a:r><a:t>Quarter</a:t></a:r></a:p></c:rich></c:tx></c:title></c:catAx>
                  <c:valAx><c:axId val="20"/><c:axPos val="l"/><c:title><c:tx><c:rich><a:p><a:r><a:t>Score</a:t></a:r></a:p></c:rich></c:tx></c:title></c:valAx>
                </c:plotArea>
                <c:legend><c:legendPos val="r"/></c:legend>
              </c:chart>
            </c:chartSpace>
            """,
        )
        archive.writestr(
            "word/diagrams/data1.xml",
            """
            <dgm:dataModel xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram"
                           xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
              <dgm:ptLst>
                <dgm:pt modelId="release"><dgm:t><a:p><a:r><a:t>Release Gate</a:t></a:r></a:p></dgm:t></dgm:pt>
                <dgm:pt modelId="dependency"><dgm:t><a:p><a:r><a:t>Dependency Review</a:t></a:r></a:p></dgm:t></dgm:pt>
                <dgm:pt modelId="handoff"><dgm:t><a:p><a:r><a:t>Owner Handoff</a:t></a:r></a:p></dgm:t></dgm:pt>
              </dgm:ptLst>
              <dgm:cxnLst>
                <dgm:cxn modelId="dc1" type="parOf" srcId="release" destId="dependency"/>
                <dgm:cxn modelId="dc2" type="parOf" srcId="release" destId="handoff"/>
              </dgm:cxnLst>
            </dgm:dataModel>
            """,
        )
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_search_proxy_uses_form_fields():
    search = load_graphrag_route_module("search")
    mock_get_client = Mock()
    mock_get_client.return_value.post = AsyncMock(return_value=_mock_response({"results": []}))
    request = Mock(spec=Request)
    request.json = AsyncMock(return_value={"query": "what is GraphRAG?", "top_k": 7})

    with (
        patch.object(search, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(search, "get_http_client", mock_get_client),
    ):
        await search.graphrag_search(request)

    mock_get_client.return_value.post.assert_awaited_once()
    _, kwargs = mock_get_client.return_value.post.await_args
    assert kwargs["data"] == {"query": "what is GraphRAG?", "top_k": "7"}


def _seed_local_pipeline_search_run():
    from modules.graphrag.pipeline_runs import get_pipeline_run_store

    connector_label = "Product Revenue -> Regional Demand: drives"
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_search_local_001",
        "status": "preview_only",
        "updated_at": "2026-02-01T10:00:00Z",
        "document": {"filename": "strategy.pptx", "domain": "slides"},
        "local_graph": {
            "filename": "strategy.pptx",
            "domain": "slides",
            "node_count": 3,
            "edge_count": 2,
            "nodes": [
                {"id": "deck", "name": "Strategy Deck", "type": "document"},
                {"id": "slide", "name": "Slide 1: Revenue Architecture", "type": "slide"},
                {"id": "connector", "name": connector_label, "type": "connector"},
            ],
            "edges": [
                {
                    "source": "Strategy Deck",
                    "target": "Slide 1: Revenue Architecture",
                    "label": "has_slide",
                    "type": "has_slide",
                    "weight": 0.72,
                    "evidence": "Revenue Architecture",
                    "metadata": {"slide": 1, "slide_title": "Revenue Architecture", "element_type": "slide_title"},
                },
                {
                    "source": "Slide 1: Revenue Architecture",
                    "target": connector_label,
                    "label": "contains_connector",
                    "type": "contains_connector",
                    "weight": 0.88,
                    "evidence": connector_label,
                    "metadata": {
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "connector_title": "drives",
                        "connector_description": "Product revenue drives regional demand",
                        "source_label": "Product Revenue",
                        "target_label": "Regional Demand",
                        "layout_bbox": "x=3.00in y=2.40in w=1.00in h=0.30in",
                        "layout_zone": "middle_center",
                        "element_type": "shape_connector",
                    },
                },
            ],
        },
        "local_context": {
            "filename": "strategy.pptx",
            "domain": "slides",
            "chunk_count": 1,
            "cached_chunk_count": 1,
            "chunks": [
                {
                    "id": "chunk_0000",
                    "chunk_index": 0,
                    "content": (
                        "[shape_connector] slide 1 connector 1: Product Revenue -> Regional Demand: drives\n"
                        "[slide_text] slide 1: Regional demand depends on product revenue and the risk map."
                    ),
                    "metadata": {
                        "provenance": [
                            "slide=1; slide_title=Revenue Architecture; layout_bbox=x=3.00in y=2.40in w=1.00in h=0.30in; layout_zone=middle_center"
                        ],
                    },
                },
            ],
        },
    })
    return connector_label


@pytest.mark.asyncio
async def test_search_uses_local_pipeline_graph_when_remote_url_missing(monkeypatch):
    from modules.graphrag.config import reset_config

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    connector_label = _seed_local_pipeline_search_run()

    search = load_graphrag_route_module("search")
    request = Mock(spec=Request)
    request.json = AsyncMock(return_value={
        "query": "Which slide contains the arrow connecting Product Revenue to Regional Demand?",
        "domain": "slides",
        "pipeline_run_ids": ["graphrag_search_local_001"],
        "source": "Strategy Deck",
        "target": connector_label,
        "top_k": 6,
    })

    result = await search.graphrag_search(request)

    assert result["source"] == "local_pipeline"
    assert result["search_method_used"] == "local_pipeline_graph"
    assert result["local_pipeline"]["ready"] is True
    assert result["local_pipeline"]["matched_path_count"] >= 1
    assert result["local_pipeline"]["matched_chunk_count"] >= 1
    assert result["graph_overview"]["relationship_count"] == 2
    assert any(item["result_type"] == "local_pipeline_graph_path" for item in result["results"])
    assert any(item["result_type"] == "local_pipeline_chunk" for item in result["results"])
    assert any("Slide 1: Revenue Architecture" in item["content"] for item in result["results"])
    assert "GRAPHRAG_URL is not configured" in result["warnings"][0]


@pytest.mark.asyncio
async def test_search_falls_back_to_local_pipeline_on_remote_error():
    search = load_graphrag_route_module("search")
    _seed_local_pipeline_search_run()
    mock_get_client = Mock()
    mock_get_client.return_value.post = AsyncMock(side_effect=httpx.ConnectError("offline"))
    request = Mock(spec=Request)
    request.json = AsyncMock(return_value={
        "query": "Find layout bbox for Product Revenue to Regional Demand connector",
        "domain": "slides",
        "top_k": 5,
    })

    with patch.object(search, "get_http_client", mock_get_client):
        result = await search.graphrag_search(request)

    assert result["source"] == "local_pipeline"
    assert result["local_pipeline"]["fallback_reason"] == "offline"
    assert any(
        item["result_type"] == "local_pipeline_graph_edge"
        and item["metadata"].get("layout_bbox") == "x=3.00in y=2.40in w=1.00in h=0.30in"
        for item in result["results"]
    )


@pytest.mark.asyncio
async def test_advanced_and_intelligent_search_support_local_pipeline_mode(monkeypatch):
    from modules.graphrag.config import reset_config

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    _seed_local_pipeline_search_run()
    search = load_graphrag_route_module("search")

    advanced_request = Mock(spec=Request)
    advanced_request.json = AsyncMock(return_value={
        "query": "layout bbox for Product Revenue connector",
        "domain": "slides",
        "search_type": "local_pipeline",
    })
    intelligent_request = Mock(spec=Request)
    intelligent_request.json = AsyncMock(return_value={
        "query": "Which slide contains Product Revenue connector?",
        "domain": "slides",
        "search_type": "local_pipeline",
    })

    advanced = await search.graphrag_advanced_search(advanced_request)
    intelligent = await search.intelligent_search(intelligent_request)

    assert advanced["source"] == "local_pipeline"
    assert advanced["local_pipeline"]["matched_graph_edge_count"] >= 1
    assert intelligent["source"] == "local_pipeline"
    assert any(item["result_type"].startswith("local_pipeline_") for item in intelligent["results"])


@pytest.mark.asyncio
async def test_multi_hop_reasoning_uses_local_pipeline_when_remote_url_missing(monkeypatch):
    from modules.graphrag.config import reset_config

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    connector_label = _seed_local_pipeline_search_run()
    search = load_graphrag_route_module("search")
    request = Mock(spec=Request)
    request.json = AsyncMock(return_value={
        "query": "Find connection between Strategy Deck and Product Revenue connector",
        "domain": "slides",
        "pipeline_run_ids": ["graphrag_search_local_001"],
        "source": "Strategy Deck",
        "target": connector_label,
        "max_hops": 5,
    })

    result = await search.graphrag_multi_hop_reasoning(request)

    assert result["reasoning_source"] == "local_pipeline"
    assert result["local_pipeline"]["ready"] is True
    assert result["local_pipeline"]["matched_path_count"] >= 1
    assert result["reasoning_path"][0]["source"] == "Strategy Deck"
    assert any("Slide 1: Revenue Architecture" in path["nodes"] for path in result["paths"])
    assert result["bridge_entities"][0]["name"] == "Slide 1: Revenue Architecture"
    assert "GRAPHRAG_URL is not configured" in result["warnings"][0]


@pytest.mark.asyncio
async def test_explain_relationship_falls_back_to_local_pipeline_on_remote_error(monkeypatch):
    from modules.graphrag.config import reset_config

    monkeypatch.setenv("GRAPHRAG_URL", "http://graphrag.test")
    reset_config()
    connector_label = _seed_local_pipeline_search_run()
    search = load_graphrag_route_module("search")
    mock_get_client = Mock()
    mock_get_client.return_value.post = AsyncMock(side_effect=httpx.ConnectError("offline"))
    request = Mock(spec=Request)
    request.json = AsyncMock(return_value={
        "source": "Slide 1: Revenue Architecture",
        "target": connector_label,
        "domain": "slides",
        "pipeline_run_ids": ["graphrag_search_local_001"],
    })

    with patch.object(search, "get_http_client", mock_get_client):
        result = await search.explain_relationship(request)

    assert result["reasoning_source"] == "local_pipeline"
    assert result["source"] == "Slide 1: Revenue Architecture"
    assert result["target"] == connector_label
    assert result["local_pipeline"]["fallback_reason"] == "offline"
    assert result["local_pipeline"]["matched_relationship_count"] >= 1
    assert result["confidence"] > 0
    assert any(
        evidence["metadata"].get("layout_bbox") == "x=3.00in y=2.40in w=1.00in h=0.30in"
        for evidence in result["evidence"]
        if isinstance(evidence.get("metadata"), dict)
    )


@pytest.mark.asyncio
async def test_remaining_reasoning_routes_support_local_pipeline_mode(monkeypatch):
    from modules.graphrag.config import reset_config

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    connector_label = _seed_local_pipeline_search_run()
    search = load_graphrag_route_module("search")

    causal_request = Mock(spec=Request)
    causal_request.json = AsyncMock(return_value={
        "query": "How does Strategy Deck influence Product Revenue to Regional Demand?",
        "domain": "slides",
        "source": "Strategy Deck",
        "target": connector_label,
        "pipeline_run_ids": ["graphrag_search_local_001"],
    })
    comparative_request = Mock(spec=Request)
    comparative_request.json = AsyncMock(return_value={
        "query": f"Compare Strategy Deck and {connector_label}",
        "domain": "slides",
        "source": "Strategy Deck",
        "target": connector_label,
        "pipeline_run_ids": ["graphrag_search_local_001"],
    })
    advanced_request = Mock(spec=Request)
    advanced_request.json = AsyncMock(return_value={
        "query": f"Find connection between Strategy Deck and {connector_label}",
        "domain": "slides",
        "source": "Strategy Deck",
        "target": connector_label,
        "pipeline_run_ids": ["graphrag_search_local_001"],
    })

    causal = await search.graphrag_causal_reasoning(causal_request)
    comparative = await search.graphrag_comparative_reasoning(comparative_request)
    advanced = await search.graphrag_advanced_reasoning(advanced_request)

    for result, task in (
        (causal, "causal"),
        (comparative, "comparative"),
        (advanced, "advanced"),
    ):
        assert result["reasoning_source"] == "local_pipeline"
        assert result["reasoning_task"] == task
        assert result["local_pipeline"]["matched_path_count"] >= 1
        assert result["warnings"]


@pytest.mark.asyncio
async def test_query_intent_proxy_uses_post_params():
    search = load_graphrag_route_module("search")
    mock_get_client = Mock()
    mock_get_client.return_value.post = AsyncMock(return_value=_mock_response({"intent_type": "factual"}))

    with (
        patch.object(search, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(search, "get_http_client", mock_get_client),
    ):
        result = await search.analyze_query_intent("compare llama nexus and graphrag")

    assert result["intent_type"] == "factual"
    mock_get_client.return_value.post.assert_awaited_once_with(
        "http://graphrag.test/api/analyze-query-intent",
        params={"query": "compare llama nexus and graphrag"},
    )


@pytest.mark.asyncio
async def test_query_complexity_proxy_uses_typed_backend_route():
    search = load_graphrag_route_module("search")
    mock_get_client = Mock()
    mock_get_client.return_value.post = AsyncMock(return_value=_mock_response({
        "primary_reasoning": "comparative",
        "requires_multi_hop": False,
    }))
    request = Mock(spec=Request)
    request.json = AsyncMock(return_value={"query": "compare A and B"})

    with (
        patch.object(search, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(search, "get_http_client", mock_get_client),
    ):
        result = await search.analyze_query_complexity(request)

    assert result["primary_reasoning"] == "comparative"
    mock_get_client.return_value.post.assert_awaited_once_with(
        "http://graphrag.test/api/query-complexity-analysis",
        params={"query": "compare A and B"},
    )


@pytest.mark.asyncio
async def test_upload_proxy_uses_upstream_files_field():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.post = AsyncMock(
        return_value=_mock_response({"chunks": [], "total_chunks": 0})
    )

    upload = Mock(spec=UploadFile)
    upload.filename = "notes.txt"
    upload.content_type = "text/plain"
    upload.read = AsyncMock(return_value=b"hello graph")

    with (
        patch.object(ingest, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(ingest, "get_http_client", mock_get_client),
    ):
        result = await ingest.graphrag_upload_document(
            upload,
            domain="docs",
            build_knowledge_graph=True,
        )

    assert result["status"] == "success"
    _, kwargs = mock_get_client.return_value.post.await_args
    assert "files" in kwargs["files"]
    assert kwargs["files"]["files"][0] == "notes.txt"
    assert kwargs["data"]["domain"] == "docs"
    assert kwargs["data"]["build_knowledge_graph"] == "true"


@pytest.mark.asyncio
async def test_json_ingest_materializes_text_as_upstream_file():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.post = AsyncMock(
        return_value=_mock_response({"results": {"note.md": {"total_chunks": 1}}, "total_files": 1})
    )
    request = Mock(spec=Request)
    request.json = AsyncMock(return_value={
        "text": "# GraphRAG\nContext.",
        "filename": "note.md",
        "domain": "docs",
        "content_type": "text/markdown",
        "build_knowledge_graph": False,
    })

    with (
        patch.object(ingest, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(ingest, "get_http_client", mock_get_client),
    ):
        result = await ingest.graphrag_ingest_document(request)

    assert result["status"] == "success"
    assert result["filename"] == "note.md"
    _, kwargs = mock_get_client.return_value.post.await_args
    assert kwargs["files"]["files"] == ("note.md", b"# GraphRAG\nContext.", "text/markdown")
    assert kwargs["data"] == {
        "domain": "docs",
        "use_semantic_chunking": "true",
        "build_knowledge_graph": "false",
    }


@pytest.mark.asyncio
async def test_batch_upload_uses_single_repeated_files_call():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.post = AsyncMock(
        return_value=_mock_response({"results": {"a.txt": {}, "b.txt": {}}, "total_files": 2})
    )

    upload_a = Mock(spec=UploadFile)
    upload_a.filename = "a.txt"
    upload_a.content_type = "text/plain"
    upload_a.read = AsyncMock(return_value=b"a")
    upload_b = Mock(spec=UploadFile)
    upload_b.filename = "b.txt"
    upload_b.content_type = "text/plain"
    upload_b.read = AsyncMock(return_value=b"b")

    with (
        patch.object(ingest, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(ingest, "get_http_client", mock_get_client),
    ):
        result = await ingest.graphrag_upload_batch(
            [upload_a, upload_b],
            domain="docs",
            build_knowledge_graph=True,
        )

    assert result["total_files"] == 2
    mock_get_client.return_value.post.assert_awaited_once()
    _, kwargs = mock_get_client.return_value.post.await_args
    assert [field for field, _ in kwargs["files"]] == ["files", "files"]
    assert kwargs["files"][0][1] == ("a.txt", b"a", "text/plain")
    assert kwargs["files"][1][1] == ("b.txt", b"b", "text/plain")
    assert kwargs["data"]["domain"] == "docs"
    assert kwargs["data"]["build_knowledge_graph"] == "true"


@pytest.mark.asyncio
async def test_pipeline_preview_returns_stage_samples_and_remembered_run():
    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload(
        "research.md",
        b"# GraphRAG Strategy\n\n| Entity | Role |\n| Llama Nexus | Orchestrator |\n\nAda Lovelace maps GraphRAG relationships.",
        "text/markdown",
    )

    result = await ingest.graphrag_ingest_pipeline_preview(
        upload,
        domain="research",
        use_semantic_chunking=True,
        build_knowledge_graph=True,
        sample_limit=4,
    )

    stage_names = [stage["name"] for stage in result["stages"]]
    assert stage_names == [
        "received",
        "structure_extraction",
        "chunking",
        "graph_candidates",
        "local_graph_artifact",
        "local_graph_reasoning_index",
        "embedding_plan",
        "document_profile",
        "local_context_cache",
        "graph_interaction_plan",
        "structured_ingest_document",
    ]
    assert result["document"]["filename"] == "research.md"
    assert result["document_profile"]["readiness_score"] >= 70
    assert "tables" in result["document_profile"]["detected_capabilities"]
    profile_stage = next(stage for stage in result["stages"] if stage["name"] == "document_profile")
    assert profile_stage["sample_outputs"]["recommended_payload"] == "structured_text"
    assert profile_stage["sample_outputs"]["evidence_profile"]["quick_counts"]["tables"] >= 1
    local_graph_stage = next(stage for stage in result["stages"] if stage["name"] == "local_graph_artifact")
    assert local_graph_stage["counts"]["nodes"] >= 1
    assert local_graph_stage["sample_outputs"]["nodes"]
    assert local_graph_stage["sample_outputs"]["edges"]
    local_context_stage = next(stage for stage in result["stages"] if stage["name"] == "local_context_cache")
    assert local_context_stage["counts"]["cached_chunks"] >= 1
    assert local_context_stage["sample_outputs"]["chunks"]
    reasoning_stage = next(stage for stage in result["stages"] if stage["name"] == "local_graph_reasoning_index")
    assert reasoning_stage["counts"]["graph_edges"] == local_graph_stage["counts"]["edges"]
    assert "bridge_entities" in reasoning_stage["sample_outputs"]
    assert "path_candidates" in reasoning_stage["sample_outputs"]
    interaction_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_interaction_plan")
    assert interaction_stage["counts"]["suggested_questions"] >= 1
    assert interaction_stage["sample_outputs"]["suggested_questions"][0]["kind"] == "summary"
    assert interaction_stage["sample_outputs"]["handoff_lanes"]
    assert interaction_stage["sample_outputs"]["starter_prompts"]
    assert interaction_stage["sample_outputs"]["retrieval_rehearsals"]
    assert interaction_stage["counts"]["retrieval_rehearsals"] == result["interaction_handoff"]["counts"]["retrieval_rehearsals"]
    assert interaction_stage["sample_outputs"]["recommended_mode"] in {
        "graph_path_reasoning",
        "chunk_path_retrieval",
        "graph_augmented_retrieval",
        "relationship_exploration",
        "source_trace_summary",
    }
    assert result["interaction_suggestions"]["question_count"] >= 1
    assert result["interaction_handoff"]["ready"] is True
    assert result["interaction_handoff"]["chat_context_endpoint"] == "/api/v1/graphrag/chat-context-v2"
    assert result["interaction_handoff"]["counts"]["starter_prompts"] >= 1
    assert result["interaction_handoff"]["counts"]["retrieval_rehearsals"] >= 1
    assert any(
        rehearsal["status"] == "ready"
        and (rehearsal["matched_chunk_count"] >= 1 or rehearsal["matched_anchor_count"] >= 1)
        for rehearsal in result["interaction_handoff"]["retrieval_rehearsals"]
    )
    assert any(lane["id"] == "chat" and lane["ready"] for lane in result["interaction_handoff"]["lanes"])
    assert result["summary"]["interaction_question_count"] == result["interaction_suggestions"]["question_count"]
    assert result["summary"]["interaction_handoff"]["starter_prompt_count"] == result["interaction_handoff"]["counts"]["starter_prompts"]
    assert result["summary"]["interaction_handoff"]["retrieval_rehearsal_count"] == result["interaction_handoff"]["counts"]["retrieval_rehearsals"]
    assert result["summary"]["local_graph_reasoning"]["bridge_entity_count"] == result["local_graph_reasoning"]["bridge_entity_count"]
    assert result["summary"]["chunk_count"] >= 1
    assert result["samples"]["elements"]
    assert result["samples"]["chunks"]
    assert result["samples"]["entities"]
    assert result["structured_ingest"]["filename"] == "research.structured.md"
    assert result["structured_ingest"]["char_count"] > 0
    assert result["structured_ingest"]["artifact_available"] is True
    assert result["structured_ingest_artifact"]["filename"] == "research.structured.md"
    assert "# Structured GraphRAG Ingest: research.md" in result["structured_ingest_artifact"]["content"]
    assert result["document"]["source_artifact_available"] is True
    assert result["source_document_artifact"]["filename"] == "research.md"
    assert result["source_document_artifact"]["content_type"] == "text/markdown"
    assert result["source_document_artifact"]["size_bytes"] == len(await upload.read())

    remembered = await ingest.graphrag_ingest_pipeline_status(result["run_id"])
    assert remembered["run_id"] == result["run_id"]
    manifest = await ingest.graphrag_ingest_pipeline_artifacts(result["run_id"])
    artifacts = {artifact["kind"]: artifact for artifact in manifest["artifacts"]}
    assert artifacts["source_document"]["available"] is True
    assert artifacts["source_document"]["content_type"] == "text/markdown"
    assert artifacts["source_document"]["endpoint"].endswith("/artifacts/source_document")
    assert artifacts["structured_ingest"]["available"] is True
    assert artifacts["local_graph"]["available"] is True
    assert artifacts["interaction_handoff"]["available"] is True
    assert artifacts["structured_ingest"]["endpoint"].endswith("/artifacts/structured_ingest")
    structured_response = await ingest.graphrag_ingest_pipeline_artifact(result["run_id"], "structured")
    assert structured_response.media_type == "text/markdown; charset=utf-8"
    assert "# Structured GraphRAG Ingest: research.md" in structured_response.body.decode("utf-8")
    source_response = await ingest.graphrag_ingest_pipeline_artifact(result["run_id"], "original")
    assert source_response.media_type == "text/markdown"
    assert b"# GraphRAG Strategy" in source_response.body
    graph_artifact = await ingest.graphrag_ingest_pipeline_artifact(result["run_id"], "local_graph")
    assert graph_artifact["edge_count"] >= 1
    downloaded = await ingest.graphrag_ingest_pipeline_artifact(result["run_id"], "manifest", download=True)
    assert downloaded.media_type == "application/json"
    assert b'"artifact_count"' in downloaded.body


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_markdown_frontmatter_links_tasks_code_and_mermaid():
    ingest = load_graphrag_route_module("ingest")
    markdown = """---
title: GraphRAG Strategy
author: Ada Lovelace
keywords:
  - graph
  - extraction
---
# GraphRAG Strategy

- [x] Extract chart evidence
- [ ] Validate chat handoff
See [risk register](https://example.com/risk-register).

```mermaid
graph TD
  Nexus[Llama Nexus] -->|orchestrates| GraphRAG[GraphRAG]
  GraphRAG --> Evidence[Evidence Graph]
```

```python
def build_kg():
    return "relationships"
```

| Entity | Role |
| --- | --- |
| Llama Nexus | Orchestrator |

Ada Lovelace maps GraphRAG relationships.
""".encode()
    upload = _mock_upload("strategy.md", markdown, "text/markdown")

    result = await ingest.graphrag_ingest_pipeline_preview(
        upload,
        domain="research",
        use_semantic_chunking=True,
        build_knowledge_graph=True,
        sample_limit=120,
    )

    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["document_metadata"] >= 3
    assert by_type["form_field"] == 2
    assert by_type["link"] >= 1
    assert by_type["code_block"] == 2
    assert by_type["diagram_node"] >= 3
    assert by_type["diagram_relationship"] >= 2
    assert by_type["table_header"] == 1
    assert by_type["table_row"] == 1

    capabilities = set(result["document_profile"]["detected_capabilities"])
    assert {"metadata", "forms", "links", "code_blocks", "diagrams", "tables"} <= capabilities
    quick_counts = result["document_profile"]["evidence_profile"]["quick_counts"]
    assert quick_counts["code_blocks"] == 2
    assert quick_counts["forms"] == 2
    assert quick_counts["links"] >= 1
    assert quick_counts["diagrams"] >= 5
    assert quick_counts["tables"] == 1

    graph_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_candidates")
    categories = graph_stage["sample_outputs"]["relationship_sample_categories"]
    assert categories["code_block"] >= 2
    assert categories["diagram"] >= 2
    assert categories["form_field"] >= 2
    assert categories["link"] >= 1
    assert categories["table_fact"] >= 1

    relationships = result["samples"]["relationships"]
    assert any(
        relationship["source"] == "Llama Nexus"
        and relationship["relation"] == "orchestrates"
        and relationship["target"] == "GraphRAG"
        for relationship in relationships
    )
    assert any(
        relationship["relation"] == "contains_code_block"
        and relationship["metadata"].get("markdown_code_language") == "python"
        and relationship["metadata"].get("container_label") == "GraphRAG Strategy"
        for relationship in relationships
    )
    assert any(
        relationship["relation"] == "uses_language"
        and relationship["target"] == "python"
        for relationship in relationships
    )
    assert any(
        relationship["relation"] == "has_field"
        and relationship["target"] == "Extract chart evidence"
        and relationship["metadata"].get("form_kind") == "markdown_task_list"
        and relationship["metadata"].get("form_field_checked") is True
        for relationship in relationships
    )
    assert any(
        relationship["relation"] == "links_to"
        and relationship["target"] == "https://example.com/risk-register"
        and relationship["metadata"].get("link_kind") == "markdown_link"
        for relationship in relationships
    )

    local_edges = result["local_graph"]["edges"]
    assert any(
        edge["source"] == "Llama Nexus"
        and edge["label"] == "orchestrates"
        and edge["target"] == "GraphRAG"
        for edge in local_edges
    )

    embedding_stage = next(stage for stage in result["stages"] if stage["name"] == "embedding_plan")
    evidence_families = embedding_stage["sample_outputs"]["embedding_evidence_family_counts"]
    assert evidence_families["code_block"] >= 1
    assert evidence_families["diagram"] >= 1

    artifact = result["structured_ingest_artifact"]["content"]
    assert "markdown_code_language=python" in artifact
    assert "link_kind=markdown_link" in artifact
    assert "diagram_part=markdown mermaid block" in artifact


@pytest.mark.asyncio
async def test_pipeline_run_listing_returns_recent_summaries_and_filters(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHRAG_PIPELINE_RUNS_FILE", str(tmp_path / "pipeline_runs.json"))
    ingest = load_graphrag_route_module("ingest")
    _reset_pipeline_run_cache(ingest)
    first = await ingest.graphrag_ingest_pipeline_preview(
        _mock_upload("research.md", b"# Research\nGraphRAG evidence.", "text/markdown"),
        domain="research",
        sample_limit=2,
    )
    second = await ingest.graphrag_ingest_pipeline_preview(
        _mock_upload("strategy.pptx", _minimal_pptx_bytes(), "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        domain="slides",
        sample_limit=2,
    )

    listed = await ingest.graphrag_list_ingestion_pipeline_runs(limit=10)

    assert listed["returned"] == 2
    assert listed["runs"][0]["run_id"] == second["run_id"]
    assert listed["runs"][1]["run_id"] == first["run_id"]
    assert "samples" not in listed["runs"][0]
    assert listed["runs"][0]["summary"]["chunk_count"] >= 1
    assert listed["runs"][0]["document_profile"]["complexity"] in {"structured", "rich", "complex"}
    assert "tables" in listed["runs"][0]["document_profile"]["detected_capabilities"]
    assert "quick_counts" in listed["runs"][0]["document_profile"]["evidence_profile"]
    assert listed["runs"][0]["structured_ingest"]["uses_structured_text"] is True
    assert listed["runs"][0]["structured_ingest"]["artifact_available"] is True
    assert listed["runs"][0]["artifacts"]["artifact_count"] >= 5
    assert listed["runs"][0]["interaction_suggestions"]["question_count"] >= 1
    assert listed["runs"][0]["interaction_handoff"]["ready"] is True
    assert listed["runs"][0]["interaction_handoff"]["starter_prompt_count"] >= 1
    assert listed["runs"][0]["interaction_handoff"]["retrieval_rehearsal_count"] >= 1
    assert listed["runs"][0]["interaction_handoff"]["rehearsed_chunk_match_count"] >= 1
    assert listed["runs"][0]["summary"]["interaction_handoff_ready"] is True
    resume_handoff = listed["runs"][0]["resume_handoff"]
    assert resume_handoff["ready"] is True
    assert resume_handoff["run_id"] == second["run_id"]
    assert resume_handoff["chat_context_endpoint"] == "/api/v1/graphrag/chat-context-v2"
    assert resume_handoff["chat_context_request"]["pipeline_run_ids"] == [second["run_id"]]
    assert resume_handoff["chat_context_request"]["use_local_pipeline_context"] is True
    assert resume_handoff["starter_prompts"][0]["prompt"]
    assert "exactly 4 bullets" in resume_handoff["starter_prompts"][0]["prompt"]
    assert "under 180 words" in resume_handoff["starter_prompts"][0]["prompt"]
    assert "at most 4 direct relationships" in resume_handoff["starter_prompts"][0]["prompt"]
    assert resume_handoff["suggested_questions"][0]["prompt"] == resume_handoff["starter_prompts"][0]["prompt"]
    assert resume_handoff["route_state"]["graphragHandoff"]["runId"] == second["run_id"]
    assert resume_handoff["route_state"]["graphragHandoff"]["prompt"] == resume_handoff["starter_prompts"][0]["prompt"]
    assert resume_handoff["route_state"]["graphragHandoff"]["chatContextEndpoint"] == "/api/v1/graphrag/chat-context-v2"
    assert resume_handoff["route_state"]["graphragHandoff"]["chatContextRequest"]["pipeline_run_ids"] == [second["run_id"]]
    assert resume_handoff["route_state"]["graphragHandoff"]["chatContextRequest"]["use_local_pipeline_context"] is True
    assert resume_handoff["route_state"]["graphragHandoff"]["suggestedQuestions"][0]["prompt"]
    assert resume_handoff["artifact_manifest_endpoint"].endswith(f"/{second['run_id']}/artifacts")
    assert any(artifact["kind"] == "structured_ingest" for artifact in resume_handoff["available_artifacts"])
    assert any(action["id"] == "ask_chat" for action in resume_handoff["actions"])
    assert listed["runs"][0]["stages"][-1]["name"] == "structured_ingest_document"

    slides = await ingest.graphrag_list_ingestion_pipeline_runs(limit=10, domain="slides")
    assert slides["returned"] == 1
    assert slides["runs"][0]["document"]["filename"] == "strategy.pptx"

    complete = await ingest.graphrag_list_ingestion_pipeline_runs(limit=10, status="complete")
    assert {run["run_id"] for run in complete["runs"]} == {first["run_id"], second["run_id"]}

    _reset_pipeline_run_cache(ingest)
    remembered_after_restart = await ingest.graphrag_ingest_pipeline_status(second["run_id"])
    assert remembered_after_restart["document"]["filename"] == "strategy.pptx"
    assert remembered_after_restart["artifacts"]["artifact_count"] >= 5
    assert remembered_after_restart["resume_handoff"]["ready"] is True
    assert remembered_after_restart["resume_handoff"]["chat_context_request"]["pipeline_run_ids"] == [second["run_id"]]
    assert remembered_after_restart["resume_handoff"]["route_state"]["graphragHandoff"]["runId"] == second["run_id"]
    source_after_restart = await ingest.graphrag_ingest_pipeline_artifact(second["run_id"], "source_document")
    assert source_after_restart.media_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    assert source_after_restart.body.startswith(b"PK")
    structured_after_restart = await ingest.graphrag_ingest_pipeline_artifact(second["run_id"], "structured")
    assert structured_after_restart.media_type == "text/markdown; charset=utf-8"
    assert "# Structured GraphRAG Ingest: strategy.pptx" in structured_after_restart.body.decode("utf-8")

    listed_after_restart = await ingest.graphrag_list_ingestion_pipeline_runs(limit=10)
    assert listed_after_restart["returned"] == 2
    assert listed_after_restart["runs"][0]["run_id"] == second["run_id"]


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_pdf_styles_links_forms_and_images():
    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload(
        "annual-strategy.pdf",
        _minimal_pdf_bytes(),
        "application/pdf",
    )

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="pdfs", sample_limit=80)

    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["heading"] >= 1
    assert by_type["paragraph"] >= 1
    assert by_type["link"] >= 3
    assert by_type["form_field"] >= 1
    assert by_type["image"] >= 1
    assert by_type["document_metadata"] >= 6
    assert by_type["outline_item"] >= 1
    assert by_type["annotation"] >= 1
    assert by_type["table_header"] >= 1
    assert by_type["table_row"] >= 1
    assert by_type["embedded_object"] >= 1
    chunking_stage = next(stage for stage in result["stages"] if stage["name"] == "chunking")
    assert chunking_stage["counts"]["semantic_scopes"] >= 3
    assert chunking_stage["counts"]["by_scope_type"]["page"] >= 1
    assert chunking_stage["counts"]["by_scope_type"]["attachment"] >= 1
    assert any(
        chunk["metadata"].get("chunk_scope_type") == "page"
        and chunk["metadata"].get("chunk_scope_label") == "page 1"
        and "GraphRAG connects PDF evidence to answers." in chunk["text"]
        for chunk in chunking_stage["sample_outputs"]["chunks"]
    )
    assert any(
        chunk["metadata"].get("chunk_scope_type") == "attachment"
        and chunk["metadata"].get("chunk_scope_label") == "attachment source-data.csv"
        for chunk in chunking_stage["sample_outputs"]["chunks"]
    )
    assert any(
        chunk["metadata"].get("chunk_scope_type") == "table"
        and "attachment source-data.csv table" in chunk["metadata"].get("chunk_scope_label", "")
        and "Artifact: Source Data" in chunk["text"]
        and "GraphRAG connects PDF evidence to answers." not in chunk["text"]
        for chunk in chunking_stage["sample_outputs"]["chunks"]
    )
    assert "embedded_objects" in result["document_profile"]["detected_capabilities"]
    assert "layout" in result["document_profile"]["detected_capabilities"]
    evidence_profile = result["document_profile"]["evidence_profile"]
    assert evidence_profile["quick_counts"]["attachments"] == 1
    assert evidence_profile["quick_counts"]["layout_elements"] >= 4
    assert evidence_profile["quick_counts"]["tables"] >= 2
    assert evidence_profile["graph_evidence"]["attachment_relationship_count"] >= 2
    assert evidence_profile["graph_evidence"]["table_fact_count"] >= 2
    assert any(
        artifact["kind"] == "attachment"
        and artifact["filename"] == "source-data.csv"
        and artifact["parsed_child_elements"] >= 2
        for artifact in evidence_profile["source_artifacts"]
    )
    assert any(
        element["type"] == "heading"
        and element["text"] == "Annual Strategy"
        and element["metadata"].get("runs", [{}])[0].get("size", 0) >= 18
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "document_metadata"
        and element["metadata"].get("metadata_field") == "author"
        and element["metadata"].get("metadata_value") == "Ada Lovelace"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "outline_item"
        and element["text"] == "Annual Strategy"
        and element["metadata"].get("outline_level") == 1
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "annotation"
        and element["metadata"].get("annotation_author") == "Grace Hopper"
        and element["metadata"].get("annotation_subject") == "Review Note"
        and element["metadata"].get("layout_bbox")
        and element["metadata"].get("layout_zone")
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "link"
        and element["text"] == "https://example.com/graphrag"
        and element["metadata"].get("layout_bbox") == "x=1.00in y=1.72in w=2.33in h=0.28in"
        and element["metadata"].get("layout_zone") == "top_left"
        and element["metadata"].get("layout_x_points") == 72
        and element["metadata"].get("page_width_inches")
        for element in result["samples"]["elements"]
    )
    pdf_link_uris = {
        element["metadata"].get("uri")
        for element in result["samples"]["elements"]
        if element["type"] == "link"
    }
    assert {
        "https://example.com/graphrag",
        "https://example.com/evidence-index",
        "https://example.com/pdf-source",
    }.issubset(pdf_link_uris)
    assert any(
        element["type"] == "form_field"
        and element["metadata"].get("label") == "Approved By"
        and element["metadata"].get("value") == "Ada Lovelace"
        and element["metadata"].get("layout_bbox") == "x=1.00in y=2.22in w=2.61in h=0.31in"
        and element["metadata"].get("layout_zone") == "top_left"
        and element["metadata"].get("layout_center_x_points") == 166
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "image"
        and element["metadata"].get("layout_bbox") == "x=1.00in y=2.78in w=0.67in h=0.67in"
        and element["metadata"].get("layout_zone") == "top_left"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "table_row"
        and "Metric: Evidence" in element["text"]
        and "Owner: Ada Lovelace" in element["text"]
        and element["metadata"].get("table_source") == "pdf_find_tables"
        and element["metadata"].get("table_row_count") == 2
        and element["metadata"].get("table_column_count") == 3
        and element["metadata"].get("table_bbox")
        and element["metadata"].get("layout_bbox")
        and element["metadata"].get("layout_zone")
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "embedded_object"
        and element["metadata"].get("media_kind") == "pdf_attachment"
        and element["metadata"].get("embedded_file_filename") == "source-data.csv"
        and element["metadata"].get("embedded_file_description") == "Source data table"
        and element["metadata"].get("embedded_file_mime") == "text/csv"
        and element["metadata"].get("embedded_file_size_bytes", 0) > 0
        and element["metadata"].get("embedded_file_sha256")
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "table_row"
        and "Artifact: Source Data" in element["text"]
        and "Curator: Grace Hopper" in element["text"]
        and element["metadata"].get("embedded_file_filename") == "source-data.csv"
        and element["metadata"].get("embedded_file_extraction_method") == "csv_table"
        for element in result["samples"]["elements"]
    )

    relationships = result["samples"]["relationships"]
    assert any(
        relationship["source"] == "annual-strategy.pdf"
        and relationship["relation"] == "created_by"
        and relationship["target"] == "Ada Lovelace"
        and relationship["metadata"].get("metadata_part") == "pdf:metadata"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "annual-strategy.pdf"
        and relationship["relation"] == "has_keyword"
        and relationship["target"] == "GraphRAG"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "annual-strategy.pdf"
        and relationship["relation"] == "has_outline_item"
        and relationship["target"] == "Annual Strategy"
        and relationship["metadata"].get("outline_level") == 1
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "annual-strategy.pdf"
        and relationship["relation"] == "has_annotation"
        and "Review graph-grounded evidence." in relationship["target"]
        and relationship["metadata"].get("annotation_author") == "Grace Hopper"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Grace Hopper"
        and relationship["relation"] == "annotated"
        and relationship["target"] == "annual-strategy.pdf"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "annual-strategy.pdf"
        and relationship["relation"] == "links_to"
        and relationship["target"] == "https://example.com/graphrag"
        and relationship["metadata"].get("layout_bbox") == "x=1.00in y=1.72in w=2.33in h=0.28in"
        and relationship["metadata"].get("layout_zone") == "top_left"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "annual-strategy.pdf"
        and relationship["relation"] == "has_field"
        and relationship["target"] == "Approved By"
        and relationship["metadata"].get("layout_bbox") == "x=1.00in y=2.22in w=2.61in h=0.31in"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Approved By"
        and relationship["relation"] == "has_value"
        and relationship["target"] == "Ada Lovelace"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "annual-strategy.pdf"
        and relationship["relation"] == "has_page"
        and relationship["target"] == "annual-strategy.pdf page 1"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "annual-strategy.pdf page 1"
        and relationship["relation"] == "contains_heading"
        and relationship["target"] == "Annual Strategy"
        and relationship["metadata"].get("container_type") == "page"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "annual-strategy.pdf page 1"
        and relationship["relation"] == "has_section"
        and relationship["target"] == "Annual Strategy"
        and relationship["metadata"].get("section_level") == 1
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Annual Strategy"
        and relationship["relation"] == "contains_text"
        and relationship["target"] == "GraphRAG connects PDF evidence to answers."
        and relationship["metadata"].get("section_label") == "Annual Strategy"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "annual-strategy.pdf page 1"
        and relationship["relation"] == "contains_outline_item"
        and relationship["target"] == "Annual Strategy"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Annual Strategy"
        and relationship["relation"] == "contains_annotation"
        and "Review graph-grounded evidence." in relationship["target"]
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Evidence"
        and relationship["relation"] == "owner"
        and relationship["target"] == "Ada Lovelace"
        and relationship["metadata"].get("table_source") == "pdf_find_tables"
        and relationship["metadata"].get("table_row_count") == 2
        and relationship["metadata"].get("layout_zone")
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Evidence"
        and relationship["relation"] == "status"
        and relationship["target"] == "Approved"
        and relationship["metadata"].get("table_column_count") == 3
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "annual-strategy.pdf page 1"
        and relationship["relation"] == "contains_table"
        and relationship["target"] == "annual-strategy.pdf page 1 table 1"
        and relationship["metadata"].get("table_source") == "pdf_find_tables"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "annual-strategy.pdf"
        and relationship["relation"] == "contains_embedded_object"
        and relationship["target"] == "source-data.csv"
        and relationship["metadata"].get("media_kind") == "pdf_attachment"
        and relationship["metadata"].get("embedded_file_filename") == "source-data.csv"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Source Data"
        and relationship["relation"] == "curator"
        and relationship["target"] == "Grace Hopper"
        and relationship["metadata"].get("embedded_file_filename") == "source-data.csv"
        and relationship["metadata"].get("embedded_file_extraction_method") == "csv_table"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "annual-strategy.pdf"
        and relationship["relation"] == "has_attachment"
        and relationship["target"] == "annual-strategy.pdf attachment source-data.csv"
        and relationship["metadata"].get("container_type") == "attachment"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "annual-strategy.pdf attachment source-data.csv"
        and relationship["relation"] == "contains_table"
        and relationship["target"] == "annual-strategy.pdf attachment source-data.csv table 1"
        and relationship["metadata"].get("embedded_file_filename") == "source-data.csv"
        for relationship in relationships
    )
    interaction_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_interaction_plan")
    suggested_labels = {
        question["label"]
        for question in interaction_stage["sample_outputs"]["suggested_questions"]
    }
    assert "Trace source artifacts" in suggested_labels
    assert "Inspect layout evidence" in suggested_labels
    assert "Review markup" in suggested_labels
    assert "Inspect form fields" in suggested_labels
    assert any(card["id"] == "layout" for card in result["interaction_handoff"]["capability_cards"])
    assert "layout" in result["interaction_handoff"]["citation_scopes"]
    assert "layout" in result["interaction_handoff"]["answer_guidance"]

    embedding_text = "\n".join(chunk["text"] for chunk in result["samples"]["chunks"])
    embedding_provenance = [
        provenance
        for chunk in result["samples"]["chunks"]
        for provenance in chunk.get("metadata", {}).get("provenance", [])
    ]
    assert "embedded_file_filename=source-data.csv" in embedding_text
    assert "Artifact: Source Data; Curator: Grace Hopper" in embedding_text
    assert any("embedded_file_filename=source-data.csv" in provenance for provenance in embedding_provenance)


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_scanned_pdf_page_ocr(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_ENABLE_IMAGE_OCR", "true")
    from modules.graphrag import document_pipeline

    monkeypatch.setattr(
        document_pipeline,
        "_optional_image_ocr",
        lambda content, *, limit=500: "Scanned Invoice ACME Corp Total Due 42",
    )

    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload(
        "scanned-invoice.pdf",
        _image_only_pdf_bytes(),
        "application/pdf",
    )

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="pdfs", sample_limit=80)

    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["image"] >= 1
    assert by_type["page_ocr_text"] >= 1
    assert "ocr" in result["document_profile"]["detected_capabilities"]
    assert not any(
        "Document appears image-only" in warning
        for warning in result["document_profile"]["quality_warnings"]
    )
    assert any(
        element["type"] == "page_ocr_text"
        and element["text"] == "Scanned Invoice ACME Corp Total Due 42"
        and element["metadata"].get("ocr_source") == "pdf_page_render"
        and element["metadata"].get("ocr_engine") == "pytesseract"
        and element["metadata"].get("image_mime") == "image/png"
        and element["metadata"].get("layout_bbox") == "x=0.00in y=0.00in w=8.26in h=11.69in"
        and element["metadata"].get("layout_zone") == "middle_center"
        for element in result["samples"]["elements"]
    )
    assert any(
        chunk["text"].find("[page_ocr_text] page 1 OCR") >= 0
        and "Scanned Invoice ACME Corp Total Due 42" in chunk["text"]
        for chunk in result["samples"]["chunks"]
    )
    assert any(
        relationship["source"] == "scanned-invoice.pdf page 1"
        and relationship["relation"] == "contains_text"
        and "Scanned Invoice ACME Corp" in relationship["target"]
        and relationship["metadata"].get("ocr_source") == "pdf_page_render"
        for relationship in result["samples"]["relationships"]
    )
    interaction_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_interaction_plan")
    assert any(
        question["label"] == "Review OCR evidence"
        and "OCR-derived evidence" in question["prompt"]
        for question in interaction_stage["sample_outputs"]["suggested_questions"]
    )


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_standalone_image_with_ocr(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_ENABLE_IMAGE_OCR", "true")
    from modules.graphrag import document_pipeline

    monkeypatch.setattr(
        document_pipeline,
        "_optional_image_ocr",
        lambda content, *, limit=500: "Screenshot ACME Launch Plan Owner Grace Hopper",
    )

    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload("launch-plan.png", _minimal_png_bytes(), "image/png")

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="images", sample_limit=80)

    assert result["summary"]["extraction_method"] == "image_file"
    assert result["document_profile"]["recommended_payload"] == "structured_text"
    assert "images" in result["document_profile"]["detected_capabilities"]
    assert "ocr" in result["document_profile"]["detected_capabilities"]

    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["image"] == 1
    assert by_type["image_ocr_text"] == 1
    assert any(
        element["type"] == "image"
        and element["metadata"].get("image_mime") == "image/png"
        and element["metadata"].get("image_width") == 1
        and element["metadata"].get("ocr_source") == "image_file"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "image_ocr_text"
        and element["text"] == "Screenshot ACME Launch Plan Owner Grace Hopper"
        and element["metadata"].get("ocr_source") == "image_file"
        for element in result["samples"]["elements"]
    )
    assert any(
        chunk["text"].find("[image_ocr_text] image OCR") >= 0
        and "Screenshot ACME Launch Plan Owner Grace Hopper" in chunk["text"]
        for chunk in result["samples"]["chunks"]
    )
    assert any(
        relationship["source"] == "launch-plan.png"
        and relationship["relation"] == "contains_text"
        and relationship["target"] == "Screenshot ACME Launch Plan Owner Grace Hopper"
        and relationship["metadata"].get("ocr_source") == "image_file"
        for relationship in result["samples"]["relationships"]
    )


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_pptx_packaged_image_ocr(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_ENABLE_IMAGE_OCR", "true")
    from modules.graphrag import document_pipeline

    monkeypatch.setattr(
        document_pipeline,
        "_optional_image_ocr",
        lambda content, *, limit=500: "Screenshot Risk Map shows EMEA Launch Dependency",
    )

    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload(
        "strategy.pptx",
        _minimal_pptx_bytes(),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="slides", sample_limit=80)

    assert "ocr" in result["document_profile"]["detected_capabilities"]
    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["image"] >= 2
    assert by_type["image_ocr_text"] >= 1
    assert any(
        element["type"] == "image_ocr_text"
        and element["text"] == "Screenshot Risk Map shows EMEA Launch Dependency"
        and element["metadata"].get("ocr_source") == "pptx_image"
        and element["metadata"].get("image_part") == "ppt/media/risk-map.png"
        and element["metadata"].get("slide_title") == "Revenue Architecture"
        and element["metadata"].get("layout_zone") == "top_right"
        for element in result["samples"]["elements"]
    )
    assert any(
        chunk["text"].find("[image_ocr_text] slide 1 image 1 OCR") >= 0
        and "Screenshot Risk Map shows EMEA Launch Dependency" in chunk["text"]
        for chunk in result["samples"]["chunks"]
    )
    graph_candidates_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_candidates")
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["ocr"] >= 1
    assert any(
        relationship["relation"] == "image_contains_text"
        and relationship["source"] == "Risk Map"
        and relationship["target"] == "Screenshot Risk Map shows EMEA Launch Dependency"
        and relationship["metadata"].get("ocr_source") == "pptx_image"
        and relationship["metadata"].get("image_part") == "ppt/media/risk-map.png"
        for relationship in result["samples"]["relationships"]
    )
    interaction_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_interaction_plan")
    assert any(
        question["label"] == "Review OCR evidence"
        for question in interaction_stage["sample_outputs"]["suggested_questions"]
    )
    embedding_stage = next(stage for stage in result["stages"] if stage["name"] == "embedding_plan")
    embedding_samples = [
        *embedding_stage["sample_outputs"]["embedding_inputs"],
        *embedding_stage["sample_outputs"]["representative_embedding_inputs"],
    ]
    embedding_text = "\n".join(item["text"] for item in embedding_samples)
    embedding_provenance = [
        provenance
        for item in embedding_samples
        for provenance in item["metadata"].get("provenance", [])
    ]
    assert "Screenshot Risk Map shows EMEA Launch Dependency" in embedding_text
    assert any("ocr_source=pptx_image" in provenance for provenance in embedding_provenance)


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_zip_bundle_entries_and_child_facts():
    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload("evidence-bundle.zip", _minimal_zip_bundle_bytes(), "application/zip")

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="bundles", sample_limit=80)

    assert result["summary"]["extraction_method"] == "archive_bundle"
    assert result["document_profile"]["recommended_payload"] == "structured_text"
    assert "archives" in result["document_profile"]["detected_capabilities"]
    assert "tables" in result["document_profile"]["detected_capabilities"]
    assert "images" in result["document_profile"]["detected_capabilities"]
    evidence_profile = result["document_profile"]["evidence_profile"]
    assert evidence_profile["quick_counts"]["archive_entries"] == 4
    assert evidence_profile["quick_counts"]["tables"] >= 1
    assert evidence_profile["quick_counts"]["table_elements"] >= 2
    assert evidence_profile["quick_counts"]["images"] >= 1
    assert evidence_profile["graph_evidence"]["archive_relationship_count"] >= 4
    assert evidence_profile["graph_evidence"]["table_fact_count"] >= 1
    assert any(
        artifact["kind"] == "archive_entry"
        and artifact["path"] == "data/owners.csv"
        and artifact["parsed_child_elements"] >= 2
        for artifact in evidence_profile["source_artifacts"]
    )

    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["archive_entry"] == 4
    assert by_type["table_header"] >= 1
    assert by_type["table_row"] >= 1
    assert by_type["json_value"] >= 1
    assert by_type["image"] >= 1
    assert any(
        element["type"] == "archive_entry"
        and element["metadata"].get("archive_entry_path") == "data/owners.csv"
        and element["metadata"].get("archive_entry_mime") == "text/csv"
        and element["metadata"].get("archive_entry_size_bytes", 0) > 0
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "table_row"
        and "Artifact: Bundle Evidence" in element["text"]
        and "Owner: Grace Hopper" in element["text"]
        and element["metadata"].get("archive_entry_path") == "data/owners.csv"
        and element["metadata"].get("archive_entry_extraction_method") == "csv_table"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "json_value"
        and "$.project.status: ready" in element["text"]
        and element["metadata"].get("archive_entry_path") == "data/metadata.json"
        and element["metadata"].get("archive_entry_extraction_method") == "json_tree"
        for element in result["samples"]["elements"]
    )

    relationships = result["samples"]["relationships"]
    assert any(
        relationship["source"] == "evidence-bundle.zip"
        and relationship["relation"] == "contains_archive_entry"
        and relationship["target"] == "data/owners.csv"
        and relationship["metadata"].get("archive_entry_path") == "data/owners.csv"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Bundle Evidence"
        and relationship["relation"] == "owner"
        and relationship["target"] == "Grace Hopper"
        and relationship["metadata"].get("archive_entry_path") == "data/owners.csv"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "evidence-bundle.zip"
        and relationship["relation"] == "has_archive_entry"
        and relationship["target"] == "evidence-bundle.zip archive data/owners.csv"
        and relationship["metadata"].get("container_type") == "archive_entry"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "evidence-bundle.zip archive data/owners.csv"
        and relationship["relation"] == "contains_table"
        and relationship["target"] == "evidence-bundle.zip archive data/owners.csv table 2"
        and relationship["metadata"].get("archive_entry_extraction_method") == "csv_table"
        for relationship in relationships
    )

    embedding_text = "\n".join(chunk["text"] for chunk in result["samples"]["chunks"])
    embedding_provenance = [
        provenance
        for chunk in result["samples"]["chunks"]
        for provenance in chunk.get("metadata", {}).get("provenance", [])
    ]
    assert "archive_entry_path=data/owners.csv" in embedding_text
    assert any("archive_entry_path=data/owners.csv" in provenance for provenance in embedding_provenance)


@pytest.mark.asyncio
async def test_pipeline_preview_deep_parses_office_documents_inside_zip_bundle():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("docs/implementation.docx", _minimal_docx_bytes())

    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload("office-evidence.zip", buffer.getvalue(), "application/zip")

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="bundles", sample_limit=120)

    assert result["summary"]["extraction_method"] == "archive_bundle"
    assert "archives" in result["document_profile"]["detected_capabilities"]
    assert "tables" in result["document_profile"]["detected_capabilities"]
    assert "forms" in result["document_profile"]["detected_capabilities"]
    assert "revisions" in result["document_profile"]["detected_capabilities"]
    assert "styled_text" in result["document_profile"]["detected_capabilities"]

    evidence_profile = result["document_profile"]["evidence_profile"]
    assert evidence_profile["quick_counts"]["archive_entries"] == 1
    assert evidence_profile["quick_counts"]["tables"] >= 1
    assert evidence_profile["quick_counts"]["forms"] >= 1
    assert evidence_profile["quick_counts"]["styled_text_blocks"] >= 1
    assert evidence_profile["graph_evidence"]["archive_relationship_count"] >= 8
    assert any(
        artifact["kind"] == "archive_entry"
        and artifact["path"] == "docs/implementation.docx"
        and artifact["parsed_child_elements"] >= 8
        for artifact in evidence_profile["source_artifacts"]
    )

    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["archive_entry"] == 1
    assert by_type["document_metadata"] >= 4
    assert by_type["heading"] >= 1
    assert by_type["form_field"] >= 3
    assert by_type["table_row"] >= 1
    assert by_type["revision"] >= 1

    assert any(
        element["type"] == "heading"
        and element["text"] == "Implementation Plan"
        and element["metadata"].get("archive_entry_path") == "docs/implementation.docx"
        and element["metadata"].get("archive_entry_extraction_method") == "docx_xml"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "form_field"
        and element["metadata"].get("archive_entry_path") == "docs/implementation.docx"
        and element["metadata"].get("archive_entry_extraction_method") == "docx_xml"
        for element in result["samples"]["elements"]
    )

    relationships = result["samples"]["relationships"]
    assert any(
        relationship["source"] == "office-evidence.zip"
        and relationship["relation"] == "contains_archive_entry"
        and relationship["target"] == "docs/implementation.docx"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "office-evidence.zip archive docs/implementation.docx"
        and relationship["relation"] == "contains_heading"
        and relationship["target"] == "Implementation Plan"
        and relationship["metadata"].get("archive_entry_extraction_method") == "docx_xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Extractor"
        and relationship["relation"] == "owner"
        and relationship["target"] == "Llama Nexus"
        and relationship["metadata"].get("archive_entry_path") == "docs/implementation.docx"
        for relationship in relationships
    )

    embedding_text = "\n".join(chunk["text"] for chunk in result["samples"]["chunks"])
    assert "archive_entry_extraction_method=docx_xml" in embedding_text
    assert "Implementation Plan" in embedding_text


@pytest.mark.asyncio
async def test_pipeline_preview_deep_parses_office_documents_embedded_in_pdf():
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Embedded Implementation Evidence", fontsize=14, fontname="helv")
    document.embfile_add(
        "implementation.docx",
        _minimal_docx_bytes(),
        filename="implementation.docx",
        desc="Implementation plan source document",
    )

    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload("embedded-plan.pdf", document.tobytes(), "application/pdf")

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="pdfs", sample_limit=120)

    assert result["summary"]["extraction_method"] == "pdf_blocks"
    assert "embedded_objects" in result["document_profile"]["detected_capabilities"]
    assert "tables" in result["document_profile"]["detected_capabilities"]
    assert "forms" in result["document_profile"]["detected_capabilities"]
    assert "revisions" in result["document_profile"]["detected_capabilities"]

    evidence_profile = result["document_profile"]["evidence_profile"]
    assert evidence_profile["quick_counts"]["attachments"] == 1
    assert evidence_profile["quick_counts"]["tables"] >= 1
    assert evidence_profile["quick_counts"]["forms"] >= 1
    assert evidence_profile["graph_evidence"]["attachment_relationship_count"] >= 8
    assert any(
        artifact["kind"] == "attachment"
        and artifact["filename"] == "implementation.docx"
        and artifact["parsed_child_elements"] >= 8
        for artifact in evidence_profile["source_artifacts"]
    )

    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["embedded_object"] >= 1
    assert by_type["document_metadata"] >= 4
    assert by_type["heading"] >= 1
    assert by_type["form_field"] >= 1
    assert by_type["table_row"] >= 1
    assert by_type["revision"] >= 1

    assert any(
        element["type"] == "heading"
        and element["text"] == "Implementation Plan"
        and element["metadata"].get("embedded_file_filename") == "implementation.docx"
        and element["metadata"].get("embedded_file_extraction_method") == "docx_xml"
        for element in result["samples"]["elements"]
    )

    relationships = result["samples"]["relationships"]
    assert any(
        relationship["source"] == "embedded-plan.pdf"
        and relationship["relation"] == "contains_embedded_object"
        and relationship["target"] == "implementation.docx"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "embedded-plan.pdf attachment implementation.docx"
        and relationship["relation"] == "contains_heading"
        and relationship["target"] == "Implementation Plan"
        and relationship["metadata"].get("embedded_file_extraction_method") == "docx_xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Extractor"
        and relationship["relation"] == "owner"
        and relationship["target"] == "Llama Nexus"
        and relationship["metadata"].get("embedded_file_filename") == "implementation.docx"
        for relationship in relationships
    )

    embedding_text = "\n".join(chunk["text"] for chunk in result["samples"]["chunks"])
    assert "embedded_file_extraction_method=docx_xml" in embedding_text
    assert "Implementation Plan" in embedding_text


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_epub_html_structure_links_and_images():
    pytest.importorskip("bs4")
    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload(
        "field-guide.epub",
        _minimal_epub_bytes(),
        "application/epub+zip",
    )

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="ebooks", sample_limit=20)

    assert result["summary"]["extraction_method"] == "epub_html"
    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["heading"] >= 1
    assert by_type["paragraph"] >= 1
    assert by_type["table_row"] >= 1
    assert by_type["link"] >= 1
    assert by_type["image"] >= 1
    assert any(
        element["type"] == "heading"
        and element["text"] == "GraphRAG Field Guide"
        and element["metadata"].get("chapter_path") == "OEBPS/chapter1.xhtml"
        for element in result["samples"]["elements"]
    )
    assert any(
        relationship["source"] == "field-guide.epub"
        and relationship["relation"] == "links_to"
        and relationship["target"] == "https://example.com/field-guide"
        for relationship in result["samples"]["relationships"]
    )
    assert result["structured_ingest"]["uses_structured_text"] is True


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_email_headers_body_links_and_attachments():
    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload(
        "launch.eml",
        _minimal_eml_bytes(),
        "message/rfc822",
    )

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="messages", sample_limit=80)

    assert result["summary"]["extraction_method"] == "email_message"
    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["document_metadata"] >= 6
    assert by_type["email_body"] >= 1
    assert by_type["embedded_object"] >= 1
    assert by_type["link"] >= 1
    assert by_type["table_header"] >= 1
    assert by_type["table_row"] >= 1

    profile = result["document_profile"]
    assert "messages" in profile["detected_capabilities"]
    assert "links" in profile["detected_capabilities"]
    assert "embedded_objects" in profile["detected_capabilities"]
    assert "tables" in profile["detected_capabilities"]
    assert profile["evidence_profile"]["quick_counts"]["message_body_parts"] >= 1
    assert profile["evidence_profile"]["quick_counts"]["attachments"] >= 1
    assert profile["relationship_type_counts"]["sent_by"] >= 1
    assert profile["relationship_type_counts"]["sent_to"] >= 1
    assert profile["relationship_type_counts"]["cc_to"] >= 1
    assert profile["relationship_type_counts"]["contains_email_body"] >= 1
    assert profile["relationship_type_counts"]["has_attachment"] >= 1

    relationships = [
        *result["samples"]["relationships"],
        *[
            {
                "source": edge.get("source"),
                "relation": edge.get("label"),
                "target": edge.get("target"),
                "metadata": edge.get("metadata", {}),
                "evidence": edge.get("evidence"),
            }
            for edge in result["local_graph"]["edges"]
        ],
    ]
    assert any(
        relationship["source"] == "launch.eml"
        and relationship["relation"] == "sent_by"
        and relationship["target"] == "Ada Lovelace <ada@example.com>"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "launch.eml"
        and relationship["relation"] == "sent_to"
        and relationship["target"] == "Grace Hopper <grace@example.com>"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "launch.eml"
        and relationship["relation"] == "cc_to"
        and relationship["target"] == "Alan Turing <alan@example.com>"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "GraphRAG Launch Evidence"
        and relationship["relation"] == "has_attachment"
        and "launch-evidence.csv" in relationship["target"]
        and relationship["metadata"].get("email_attachment_filename") == "launch-evidence.csv"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "launch.eml"
        and relationship["relation"] == "links_to"
        and relationship["target"] == "https://example.com/runbook"
        and relationship["metadata"].get("email_part_kind") == "plain"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Graph Coverage"
        and relationship["relation"] == "owner"
        and relationship["target"] == "Grace Hopper"
        and relationship["metadata"].get("embedded_file_extraction_method") == "csv_table"
        for relationship in relationships
    )

    interaction_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_interaction_plan")
    assert any(
        question["label"] == "Trace message"
        for question in interaction_stage["sample_outputs"]["suggested_questions"]
    )
    assert any(card["id"] == "messages" for card in result["interaction_handoff"]["capability_cards"])
    assert "message" in result["interaction_handoff"]["citation_scopes"]

    embedding_stage = next(stage for stage in result["stages"] if stage["name"] == "embedding_plan")
    embedding_text = "\n".join(item["text"] for item in embedding_stage["sample_outputs"]["embedding_inputs"])
    representative_embedding_text = "\n".join(
        item["text"]
        for item in embedding_stage["sample_outputs"]["representative_embedding_inputs"]
    )
    assert "email_message_id=<launch-evidence@example.com>" in embedding_text
    assert "email_attachment_filename=launch-evidence.csv" in embedding_text
    assert "launch-evidence.csv" in representative_embedding_text
    assert "https://example.com/runbook" in embedding_text
    assert result["structured_ingest"]["uses_structured_text"] is True


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_opendocument_text_and_table_relationships():
    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload(
        "deployment.odt",
        _minimal_odt_bytes(),
        "application/vnd.oasis.opendocument.text",
    )

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="docs", sample_limit=80)

    assert result["summary"]["extraction_method"] == "odt_xml"
    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["document_metadata"] >= 5
    assert by_type["heading"] >= 1
    assert by_type["paragraph"] >= 1
    assert by_type["link"] >= 1
    assert by_type["comment"] >= 2
    assert by_type["revision"] >= 2
    assert by_type["image"] >= 1
    assert by_type["table_header"] >= 1
    assert by_type["table_row"] >= 1
    assert "revisions" in result["document_profile"]["detected_capabilities"]
    assert "styled_text" in result["document_profile"]["detected_capabilities"]
    assert result["document_profile"]["evidence_profile"]["quick_counts"]["styled_text_blocks"] >= 1
    graph_candidates_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_candidates")
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["revision"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["styled_text"] >= 1
    interaction_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_interaction_plan")
    emphasis_questions = [
        question
        for question in interaction_stage["sample_outputs"]["suggested_questions"]
        if question.get("label") == "Inspect emphasized text"
    ]
    assert emphasis_questions
    assert emphasis_questions[0]["featured_emphasis_term"] == "Graph handoff routing"
    assert "Graph handoff routing" in emphasis_questions[0]["prompt"]
    assert "style summary \"bold\"" in emphasis_questions[0]["prompt"]
    assert any(
        anchor["kind"] == "styled_text"
        and anchor["target"] == "Graph handoff routing"
        and anchor["metadata"].get("style_summary") == "bold"
        for anchor in result["interaction_handoff"]["graph_anchors"]
    )
    assert any(
        rehearsal["prompt_id"] == emphasis_questions[0]["id"]
        and rehearsal["matched_anchor_count"] >= 1
        and any(anchor.get("kind") == "styled_text" for anchor in rehearsal.get("top_anchors", []))
        for rehearsal in result["interaction_handoff"]["retrieval_rehearsals"]
    )
    assert any(
        element["type"] == "document_metadata"
        and element["metadata"].get("metadata_part") == "meta.xml"
        and element["metadata"].get("metadata_field") == "title"
        and element["metadata"].get("metadata_value") == "OpenDocument Deployment Plan"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "link"
        and element["metadata"].get("uri") == "https://example.com/odf-source"
        and element["metadata"].get("link_kind") == "odf_hyperlink"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "paragraph"
        and "Graph handoff routing" in element["text"]
        and element["metadata"].get("section") == "Deployment Plan"
        and element["metadata"].get("style_summary") == "bold"
        and element["metadata"].get("emphasis_terms") == ["Graph handoff routing"]
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "paragraph"
        and element["text"] == "Citation path review"
        and element["metadata"].get("section") == "Deployment Plan"
        and element["metadata"].get("style_summary") == "bold; italic; color #126B5D; bullet levels 0"
        and element["metadata"].get("emphasis_terms") == ["Citation path review"]
        and element["metadata"].get("bullet_levels") == [0]
        for element in result["samples"]["elements"]
    )
    odt_relationships = [
        *result["samples"]["relationships"],
        *[
            {
                "source": edge.get("source"),
                "relation": edge.get("label"),
                "target": edge.get("target"),
                "metadata": edge.get("metadata", {}),
                "evidence": edge.get("evidence"),
            }
            for edge in result["local_graph"]["edges"]
        ],
    ]
    assert any(
        relationship["source"] == "Deployment Plan"
        and relationship["relation"] == "emphasizes"
        and relationship["target"] == "Graph handoff routing"
        and relationship["metadata"].get("styled_text") is True
        and relationship["metadata"].get("style_summary") == "bold"
        for relationship in odt_relationships
    )
    assert any(
        relationship["source"] == "Deployment Plan"
        and relationship["relation"] == "uses_bullet_structure"
        and relationship["target"] == "bullet levels 0"
        and relationship["metadata"].get("bullet_levels") == [0]
        for relationship in odt_relationships
    )
    assert any(
        element["type"] == "comment"
        and element["metadata"].get("author") == "Grace Hopper"
        and element["metadata"].get("comment_kind") == "odf_annotation"
        and element["metadata"].get("comment_anchor_text") == "OpenDocument evidence path"
        and "OpenDocument evidence path" in element["text"]
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "comment"
        and element["metadata"].get("comment_kind") == "odf_annotation_end"
        and element["metadata"].get("comment_id") == "annotation-1"
        and "Annotation range end annotation-1" in element["text"]
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "revision"
        and "Automated graph trace review" in element["text"]
        and element["metadata"].get("revision_type") == "inserted"
        and element["metadata"].get("revision_author") == "Alan Turing"
        and element["metadata"].get("revision_date") == "2026-03-02T10:30:00Z"
        and element["metadata"].get("revision_id") == "ct-insert-1"
        and element["metadata"].get("revision_part") == "content.xml"
        and element["metadata"].get("revision_change_kind") == "odf_tracked_change"
        and element["metadata"].get("revision_anchor_text") == "Automated graph trace review"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "revision"
        and "Legacy file staging checklist" in element["text"]
        and element["metadata"].get("revision_type") == "deleted"
        and element["metadata"].get("revision_author") == "Katherine Johnson"
        and element["metadata"].get("revision_id") == "ct-delete-1"
        and element["metadata"].get("revision_part") == "content.xml"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "image"
        and element["metadata"].get("image_part") == "Pictures/deployment.png"
        and element["metadata"].get("image_mime") == "image/png"
        for element in result["samples"]["elements"]
    )
    assert any(
        relationship["source"] == "deployment.odt"
        and relationship["relation"] == "created_by"
        and relationship["target"] == "Ada Lovelace"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "deployment.odt"
        and relationship["relation"] == "has_keyword"
        and relationship["target"] == "OpenDocument"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "deployment.odt"
        and relationship["relation"] == "links_to"
        and relationship["target"] == "https://example.com/odf-source"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "deployment.odt"
        and relationship["relation"] == "has_comment"
        and "OpenDocument evidence path" in relationship["target"]
        and relationship["metadata"].get("author") == "Grace Hopper"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "deployment.odt"
        and relationship["relation"] == "comments_on"
        and relationship["target"] == "OpenDocument evidence path"
        and relationship["metadata"].get("comment_anchor_text") == "OpenDocument evidence path"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "Grace Hopper"
        and relationship["relation"] == "commented_on"
        and relationship["target"] == "OpenDocument evidence path"
        and relationship["metadata"].get("comment_kind") == "odf_annotation"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "deployment.odt"
        and relationship["relation"] == "has_revision"
        and "Automated graph trace review" in relationship["target"]
        and relationship["metadata"].get("revision_change_kind") == "odf_tracked_change"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "Alan Turing"
        and relationship["relation"] == "inserted_in"
        and relationship["target"] == "deployment.odt"
        and relationship["metadata"].get("revision_id") == "ct-insert-1"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "Katherine Johnson"
        and relationship["relation"] == "deleted_in"
        and relationship["target"] == "deployment.odt"
        and relationship["metadata"].get("revision_id") == "ct-delete-1"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "Grace Hopper"
        and relationship["relation"] == "commented_on"
        and relationship["target"] == "deployment.odt"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "deployment.odt"
        and relationship["relation"] == "contains_image"
        and relationship["target"] == "Deployment Diagram"
        and relationship["metadata"].get("image_part") == "Pictures/deployment.png"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "Uploader"
        and relationship["relation"] == "owner"
        and relationship["target"] == "Ada Lovelace"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "Deployment Plan"
        and relationship["relation"] == "contains_link"
        and relationship["target"] == "https://example.com/odf-source"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "Deployment Plan"
        and relationship["relation"] == "contains_image"
        and relationship["target"] == "Deployment Diagram"
        for relationship in result["samples"]["relationships"]
    )
    embedding_stage = next(stage for stage in result["stages"] if stage["name"] == "embedding_plan")
    embedding_text = "\n".join(item["text"] for item in embedding_stage["sample_outputs"]["embedding_inputs"])
    embedding_provenance = [
        provenance
        for item in embedding_stage["sample_outputs"]["embedding_inputs"]
        for provenance in item["metadata"].get("provenance", [])
    ]
    assert "metadata_part=meta.xml" in embedding_text
    assert "uri=https://example.com/odf-source" in embedding_text
    assert "image_part=Pictures/deployment.png" in embedding_text
    assert "comment_anchor_text=OpenDocument evidence path" in embedding_text
    assert "revision_type=inserted" in embedding_text
    assert "revision_author=Alan Turing" in embedding_text
    assert "revision_id=ct-insert-1" in embedding_text
    assert "revision_change_kind=odf_tracked_change" in embedding_text
    assert "Automated graph trace review" in embedding_text
    assert "style_summary=bold" in embedding_text
    assert 'emphasis_terms=["Graph handoff routing"]' in embedding_text
    assert "bullet_levels=[0]" in embedding_text
    assert any("author=Grace Hopper" in provenance for provenance in embedding_provenance)
    assert any("comment_anchor_text=OpenDocument evidence path" in provenance for provenance in embedding_provenance)
    assert any("revision_author=Katherine Johnson" in provenance for provenance in embedding_provenance)
    assert any("revision_id=ct-delete-1" in provenance for provenance in embedding_provenance)
    assert any("image_part=Pictures/deployment.png" in provenance for provenance in embedding_provenance)
    assert any("style_summary=bold" in provenance for provenance in embedding_provenance)
    assert result["structured_ingest"]["uses_structured_text"] is True


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_odp_presentation_structure():
    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload(
        "launch-deck.odp",
        _minimal_odp_bytes(),
        "application/vnd.oasis.opendocument.presentation",
    )

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="slides", sample_limit=80)

    assert result["summary"]["extraction_method"] == "odp_xml"
    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["document_metadata"] >= 4
    assert by_type["slide_metadata"] >= 2
    assert by_type["slide_title"] >= 2
    assert by_type["slide_text"] >= 2
    assert by_type["table_header"] >= 1
    assert by_type["table_row"] >= 1
    assert by_type["table_cell"] >= 5
    assert by_type["image"] >= 1
    assert by_type["link"] >= 1
    assert by_type["chart_axis"] >= 2
    assert by_type["chart_legend"] >= 1
    assert by_type["chart_series"] >= 1
    assert by_type["chart_point"] >= 2
    assert by_type["form_field"] >= 3
    assert by_type["shape_metadata"] >= 2
    assert by_type["shape_connector"] >= 1

    profile = result["document_profile"]
    assert "presentation_structure" in profile["detected_capabilities"]
    assert "tables" in profile["detected_capabilities"]
    assert "charts" in profile["detected_capabilities"]
    assert "forms" in profile["detected_capabilities"]
    assert "shapes" in profile["detected_capabilities"]
    assert "connectors" in profile["detected_capabilities"]
    assert "images" in profile["detected_capabilities"]
    assert "links" in profile["detected_capabilities"]
    assert "styled_text" in profile["detected_capabilities"]
    assert profile["evidence_profile"]["quick_counts"]["slides"] >= 2
    assert profile["evidence_profile"]["quick_counts"]["hidden_slides"] >= 1
    assert profile["evidence_profile"]["quick_counts"]["charts"] >= 4
    assert profile["evidence_profile"]["quick_counts"]["forms"] >= 3
    assert profile["evidence_profile"]["quick_counts"]["shapes"] >= 2
    assert profile["evidence_profile"]["quick_counts"]["connectors"] >= 1
    assert profile["evidence_profile"]["quick_counts"]["styled_text_blocks"] >= 1
    assert profile["evidence_profile"]["graph_evidence"]["chart_relationship_count"] >= 1
    assert profile["relationship_type_counts"]["contains_chart"] >= 1
    assert profile["relationship_type_counts"]["contains_chart_series"] >= 1
    assert profile["relationship_type_counts"]["contains_chart_point"] >= 1
    assert profile["relationship_type_counts"]["uses_chart_source"] >= 1
    assert profile["relationship_type_counts"]["has_field"] >= 3
    assert profile["relationship_type_counts"]["contains_shape"] >= 2
    assert profile["relationship_type_counts"]["contains_connector"] >= 1
    assert profile["relationship_type_counts"]["requires"] >= 1
    assert profile["relationship_type_counts"]["has_hidden_slide"] >= 1
    assert profile["relationship_type_counts"]["has_slide_visibility"] >= 1
    assert profile["relationship_type_counts"]["uses_slide_part"] >= 1

    graph_candidates_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_candidates")
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["chart_fact"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["form_field"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["connector"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["styled_text"] >= 1

    interaction_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_interaction_plan")
    assert any(
        question["label"] == "Inspect slide structure"
        for question in interaction_stage["sample_outputs"]["suggested_questions"]
    )
    assert any(
        question["label"] == "Audit connectors"
        for question in interaction_stage["sample_outputs"]["suggested_questions"]
    )
    assert any(card["id"] == "presentation_structure" for card in result["interaction_handoff"]["capability_cards"])
    assert any(card["id"] == "forms" for card in result["interaction_handoff"]["capability_cards"])
    assert any(card["id"] == "connectors" for card in result["interaction_handoff"]["capability_cards"])
    assert "slide visibility" in result["interaction_handoff"]["citation_scopes"]
    assert "connector" in result["interaction_handoff"]["citation_scopes"]

    assert any(
        element["type"] == "slide_metadata"
        and element["metadata"].get("slide_title") == "Hidden Assumptions"
        and element["metadata"].get("slide_visibility") == "hidden"
        and element["metadata"].get("slide_part") == "content.xml"
        and element["metadata"].get("format") == "odp"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "slide_text"
        and "regional rollout" in element["text"]
        and element["metadata"].get("slide_title") == "Launch Overview"
        and element["metadata"].get("style_summary") == "bold; color #126B5D"
        and element["metadata"].get("emphasis_terms") == ["regional rollout"]
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "link"
        and element["metadata"].get("uri") == "https://example.com/odp-source"
        and element["metadata"].get("slide_title") == "Launch Overview"
        and element["metadata"].get("slide_visibility") == "visible"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "table_row"
        and element["metadata"].get("table_name") == "Milestones"
        and element["metadata"].get("slide_title") == "Launch Overview"
        and element["metadata"].get("slide_part") == "content.xml"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "table_cell"
        and element["text"] == "Launch Governance"
        and element["metadata"].get("table_name") == "Milestones"
        and element["metadata"].get("slide_title") == "Launch Overview"
        and element["metadata"].get("slide_part") == "content.xml"
        and element["metadata"].get("table_cell_reference") == "R3C1"
        and element["metadata"].get("cell") == "A3"
        and element["metadata"].get("table_cell_grid_span") == 2
        and element["metadata"].get("table_cell_style_name") == "HighlightCell"
        and element["metadata"].get("format") == "odp"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "chart_series"
        and element["metadata"].get("chart_title") == "Launch Risk Burn-down"
        and element["metadata"].get("chart_part") == "Object 1/content.xml"
        and element["metadata"].get("chart_source_part") == "Object 1/content.xml"
        and element["metadata"].get("chart_source_extraction_method") == "odf_chart_table"
        and element["metadata"].get("slide_title") == "Launch Overview"
        and element["metadata"].get("series_name") == "Open Risks"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "chart_point"
        and element["metadata"].get("chart_title") == "Launch Risk Burn-down"
        and element["metadata"].get("series_name") == "Open Risks"
        and element["metadata"].get("category") == "Launch"
        and element["metadata"].get("value") == "3"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "chart_axis"
        and element["metadata"].get("chart_title") == "Launch Risk Burn-down"
        and element["metadata"].get("chart_axis_type") == "value"
        and element["metadata"].get("chart_axis_title") == "Open risks"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "table_row"
        and element["metadata"].get("table_name") == "RiskBurnDown"
        and element["metadata"].get("chart_title") == "Launch Risk Burn-down"
        and element["metadata"].get("chart_source_extraction_method") == "odf_chart_table"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "form_field"
        and element["metadata"].get("form_kind") == "odf_control"
        and element["metadata"].get("field_type") == "checkbox"
        and element["metadata"].get("label") == "Approval Required"
        and element["metadata"].get("value") == "True"
        and element["metadata"].get("form_field_checked") is True
        and element["metadata"].get("control_id") == "approval-required"
        and element["metadata"].get("control_part") == "content.xml"
        and element["metadata"].get("slide_title") == "Launch Overview"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "form_field"
        and element["metadata"].get("field_type") == "listbox"
        and element["metadata"].get("label") == "Launch Region"
        and element["metadata"].get("value") == "North America"
        and element["metadata"].get("form_field_options") == ["North America", "Europe"]
        and element["metadata"].get("form_field_option_values") == ["NA", "EU"]
        and element["metadata"].get("form_field_selected_option") == "North America"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "form_field"
        and element["metadata"].get("field_type") == "text"
        and element["metadata"].get("label") == "Risk Owner"
        and element["metadata"].get("value") == "Ada Lovelace"
        and element["metadata"].get("control_id") == "risk-owner"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "shape_metadata"
        and element["text"] == "Regional Demand: Demand signal for launch approval"
        and element["metadata"].get("shape_id") == "shape-demand"
        and element["metadata"].get("shape_title") == "Regional Demand"
        and element["metadata"].get("shape_description") == "Demand signal for launch approval"
        and element["metadata"].get("layout_bbox") == "x=1.00in y=4.10in w=2.00in h=0.50in"
        and element["metadata"].get("slide_title") == "Launch Overview"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "shape_connector"
        and element["text"] == "Regional Demand -> Approval Gate: requires"
        and element["metadata"].get("connector_id") == "connector-approval"
        and element["metadata"].get("connector_title") == "requires"
        and element["metadata"].get("connector_description") == "Regional demand requires approval gate"
        and element["metadata"].get("source_shape_id") == "shape-demand"
        and element["metadata"].get("target_shape_id") == "shape-approval"
        and element["metadata"].get("source_label") == "Regional Demand"
        and element["metadata"].get("target_label") == "Approval Gate"
        and element["metadata"].get("relation") == "requires"
        and element["metadata"].get("layout_bbox") == "x=3.00in y=4.35in w=0.50in h=0.00in"
        for element in result["samples"]["elements"]
    )

    relationships = [
        *result["samples"]["relationships"],
        *[
            {
                "source": edge.get("source"),
                "relation": edge.get("label"),
                "target": edge.get("target"),
                "metadata": edge.get("metadata", {}),
                "evidence": edge.get("evidence"),
            }
            for edge in result["local_graph"]["edges"]
        ],
    ]
    assert any(
        relationship["source"] == "launch-deck.odp"
        and relationship["relation"] == "has_hidden_slide"
        and relationship["target"] == "Slide 2: Hidden Assumptions"
        and relationship["metadata"].get("slide_visibility") == "hidden"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 2: Hidden Assumptions"
        and relationship["relation"] == "has_slide_visibility"
        and relationship["target"] == "hidden"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 2: Hidden Assumptions"
        and relationship["relation"] == "uses_slide_part"
        and relationship["target"] == "content.xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Launch Overview"
        and relationship["relation"] == "contains_link"
        and relationship["target"] == "Launch source"
        and relationship["metadata"].get("uri") == "https://example.com/odp-source"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Regional Rollout"
        and relationship["relation"] == "owner"
        and relationship["target"] == "Ada Lovelace"
        and relationship["metadata"].get("slide_title") == "Launch Overview"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Launch Overview"
        and relationship["relation"] == "contains_table"
        and relationship["target"] == "R3C1: Launch Governance"
        and relationship["metadata"].get("table_cell_grid_span") == 2
        and relationship["metadata"].get("table_cell_style_name") == "HighlightCell"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Launch Overview"
        and relationship["relation"] == "contains_chart"
        and relationship["target"] == "Launch Risk Burn-down"
        and relationship["metadata"].get("chart_part") == "Object 1/content.xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Launch Risk Burn-down"
        and relationship["relation"] == "contains_chart_series"
        and relationship["target"] == "Open Risks"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Launch Risk Burn-down"
        and relationship["relation"] == "contains_chart_point"
        and relationship["target"] == "Open Risks Launch"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Open Risks"
        and relationship["relation"] == "launch_value"
        and relationship["target"] == "3"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "launch-deck.odp"
        and relationship["relation"] == "uses_chart_source"
        and relationship["target"] == "Object 1/content.xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Launch Overview"
        and relationship["relation"] == "has_field"
        and relationship["target"] == "Approval Required"
        and relationship["metadata"].get("form_kind") == "odf_control"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Launch Region"
        and relationship["relation"] == "has_value"
        and relationship["target"] == "North America"
        and relationship["metadata"].get("form_field_options") == ["North America", "Europe"]
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Risk Owner"
        and relationship["relation"] == "has_value"
        and relationship["target"] == "Ada Lovelace"
        and relationship["metadata"].get("control_id") == "risk-owner"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Launch Overview"
        and relationship["relation"] == "contains_shape"
        and relationship["target"] == "Regional Demand"
        and relationship["metadata"].get("shape_id") == "shape-demand"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Launch Overview"
        and relationship["relation"] == "contains_connector"
        and relationship["target"] == "Regional Demand -> Approval Gate: requires"
        and relationship["metadata"].get("connector_id") == "connector-approval"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Regional Demand"
        and relationship["relation"] == "requires"
        and relationship["target"] == "Approval Gate"
        and relationship["metadata"].get("connector_description") == "Regional demand requires approval gate"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Launch Overview"
        and relationship["relation"] == "emphasizes"
        and relationship["target"] == "regional rollout"
        and relationship["metadata"].get("styled_text") is True
        and relationship["metadata"].get("style_summary") == "bold; color #126B5D"
        for relationship in relationships
    )

    embedding_stage = next(stage for stage in result["stages"] if stage["name"] == "embedding_plan")
    embedding_text = "\n".join(item["text"] for item in embedding_stage["sample_outputs"]["embedding_inputs"])
    embedding_provenance = [
        provenance
        for item in embedding_stage["sample_outputs"]["embedding_inputs"]
        for provenance in item["metadata"].get("provenance", [])
    ]
    assert "slide_title=Hidden Assumptions" in embedding_text
    assert "slide_visibility=hidden" in embedding_text
    assert "slide_part=content.xml" in embedding_text
    assert "Confidential ODP assumptions support regional demand." in embedding_text
    assert "uri=https://example.com/odp-source" in embedding_text
    assert "table_name=Milestones" in embedding_text
    assert "table_cell_reference=R3C1" in embedding_text
    assert "table_cell_grid_span=2" in embedding_text
    assert "table_cell_style_name=HighlightCell" in embedding_text
    assert "chart_title=Launch Risk Burn-down" in embedding_text
    assert "chart_part=Object 1/content.xml" in embedding_text
    assert "chart_source_part=Object 1/content.xml" in embedding_text
    assert "chart_source_extraction_method=odf_chart_table" in embedding_text
    assert "Open Risks; Launch: 3" in embedding_text
    assert "form_kind=odf_control" in embedding_text
    assert "field_type=listbox" in embedding_text
    assert "control_id=launch-region" in embedding_text
    assert "Launch Region: North America; options: North America, Europe" in embedding_text
    assert "shape_id=shape-demand" in embedding_text
    assert "shape_title=Regional Demand" in embedding_text
    assert "connector_id=connector-approval" in embedding_text
    assert "connector_title=requires" in embedding_text
    assert "connector_description=Regional demand requires approval gate" in embedding_text
    assert "style_summary=bold; color #126B5D" in embedding_text
    assert 'emphasis_terms=["regional rollout"]' in embedding_text
    assert any("slide_visibility=hidden" in provenance for provenance in embedding_provenance)
    assert any("slide_part=content.xml" in provenance for provenance in embedding_provenance)
    assert any("chart_source_part=Object 1/content.xml" in provenance for provenance in embedding_provenance)
    assert any("form_kind=odf_control" in provenance for provenance in embedding_provenance)
    assert any("control_id=risk-owner" in provenance for provenance in embedding_provenance)
    assert any("shape_id=shape-demand" in provenance for provenance in embedding_provenance)
    assert any("connector_id=connector-approval" in provenance for provenance in embedding_provenance)
    assert any("style_summary=bold; color #126B5D" in provenance for provenance in embedding_provenance)
    assert result["structured_ingest"]["uses_structured_text"] is True


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_ods_formulas_and_table_relationships():
    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload(
        "quality-scorecard.ods",
        _minimal_ods_bytes(),
        "application/vnd.oasis.opendocument.spreadsheet",
    )

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="sheets", sample_limit=120)

    assert result["summary"]["extraction_method"] == "ods_xml"
    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["document_metadata"] >= 3
    assert by_type["table_header"] >= 1
    assert by_type["table_row"] >= 1
    assert by_type["table_cell"] >= 6
    assert by_type["comment"] >= 1
    assert "comments" in result["document_profile"]["detected_capabilities"]
    graph_candidates_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_candidates")
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["comment"] >= 1
    assert any(
        element["type"] == "table_row"
        and element["metadata"].get("table_name") == "Scorecard"
        and element["metadata"].get("format") == "ods"
        and element["metadata"].get("formulas", [{}])[0].get("cell") == "C2"
        and element["metadata"].get("formulas", [{}])[0].get("formula") == "of:=SUM([.C3];[.C4])"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "table_cell"
        and element["text"] == "18"
        and element["metadata"].get("table_name") == "Scorecard"
        and element["metadata"].get("cell") == "C2"
        and element["metadata"].get("table_cell_reference") == "R2C3"
        and element["metadata"].get("cell_value_type") == "float"
        and element["metadata"].get("formulas", [{}])[0].get("formula") == "of:=SUM([.C3];[.C4])"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "comment"
        and element["metadata"].get("comment_kind") == "odf_annotation"
        and element["metadata"].get("author") == "Grace Hopper"
        and element["metadata"].get("comment_anchor_sheet") == "Scorecard"
        and element["metadata"].get("comment_anchor_cell") == "B2"
        and element["metadata"].get("comment_anchor_header") == "Owner"
        and element["metadata"].get("comment_anchor_value") == "Ada Lovelace"
        and element["metadata"].get("comment_anchor_text") == "Owner; Ada Lovelace"
        for element in result["samples"]["elements"]
    )

    relationships = result["samples"]["relationships"]
    assert any(
        relationship["source"] == "quality-scorecard.ods"
        and relationship["relation"] == "created_by"
        and relationship["target"] == "Ada Lovelace"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "GraphRAG"
        and relationship["relation"] == "owner"
        and relationship["target"] == "Ada Lovelace"
        and relationship["metadata"].get("formulas", [{}])[0].get("cell") == "C2"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "GraphRAG"
        and relationship["relation"] == "score"
        and relationship["target"] == "18"
        and relationship["metadata"].get("formulas", [{}])[0].get("formula") == "of:=SUM([.C3];[.C4])"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!B2"
        and relationship["relation"] == "has_comment"
        and "Confirm owner before graph publication" in relationship["target"]
        and relationship["metadata"].get("comment_anchor_cell") == "B2"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!B2"
        and relationship["relation"] == "comments_on"
        and relationship["target"] == "Owner; Ada Lovelace"
        and relationship["metadata"].get("comment_anchor_header") == "Owner"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Grace Hopper"
        and relationship["relation"] == "commented_on"
        and relationship["target"] == "Owner; Ada Lovelace"
        and relationship["metadata"].get("comment_anchor_sheet") == "Scorecard"
        for relationship in relationships
    )

    embedding_stage = next(stage for stage in result["stages"] if stage["name"] == "embedding_plan")
    embedding_text = "\n".join(item["text"] for item in embedding_stage["sample_outputs"]["embedding_inputs"])
    embedding_provenance = [
        provenance
        for item in embedding_stage["sample_outputs"]["embedding_inputs"]
        for provenance in item["metadata"].get("provenance", [])
    ]
    assert "table_name=Scorecard" in embedding_text
    assert "table_cell_reference=R2C3" in embedding_text
    assert "cell_value_type=float" in embedding_text
    assert "of:=SUM([.C3];[.C4])" in embedding_text
    assert "comment_anchor_text=Owner; Ada Lovelace" in embedding_text
    assert "comment_anchor_cell=B2" in embedding_text
    assert any("cell" in provenance and "C2" in provenance for provenance in embedding_provenance)
    assert any("comment_anchor_header=Owner" in provenance for provenance in embedding_provenance)
    assert any("comment_anchor_value=Ada Lovelace" in provenance for provenance in embedding_provenance)
    assert result["structured_ingest"]["uses_structured_text"] is True


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_powerpoint_tables_and_charts():
    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload(
        "strategy.pptx",
        _minimal_pptx_bytes(),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="slides", sample_limit=80)

    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["document_metadata"] >= 9
    assert by_type["slide_metadata"] >= 2
    assert by_type["slide_title"] >= 1
    assert by_type["slide_text"] >= 3
    assert by_type["table_header"] >= 3
    assert by_type["table_row"] >= 3
    assert by_type["table_cell"] >= 5
    assert by_type["chart_series"] >= 1
    assert by_type["chart_point"] >= 2
    assert by_type["chart_axis"] >= 2
    assert by_type["chart_legend"] >= 1
    assert by_type["chart_data_label"] >= 1
    assert by_type["chart_trendline"] >= 1
    assert by_type["image"] >= 2
    assert by_type["link"] >= 1
    assert by_type["comment"] >= 1
    assert by_type["speaker_notes"] >= 1
    assert by_type["media"] >= 2
    assert by_type["embedded_object"] >= 2
    assert by_type["form_field"] >= 2
    assert by_type["diagram_node"] >= 3
    assert by_type["diagram_relationship"] >= 2
    assert by_type["shape_metadata"] >= 1
    assert by_type["shape_connector"] >= 2
    chunking_stage = next(stage for stage in result["stages"] if stage["name"] == "chunking")
    assert chunking_stage["counts"]["semantic_scopes"] >= 4
    assert chunking_stage["counts"]["by_scope_type"]["slide"] >= 2
    assert chunking_stage["counts"]["by_scope_type"]["chart_source"] >= 1
    assert chunking_stage["counts"]["by_scope_type"]["attachment"] >= 1
    assert any(
        chunk["metadata"].get("chunk_scope_type") == "slide"
        and chunk["metadata"].get("chunk_scope_label") == "Slide 1: Revenue Architecture"
        for chunk in chunking_stage["sample_outputs"]["chunks"]
    )
    assert any(
        chunk["metadata"].get("chunk_scope_type") == "slide"
        and chunk["metadata"].get("chunk_scope_label") == "Slide 2: Hidden Assumptions"
        and "slide_visibility=hidden" in chunk["text"]
        for chunk in chunking_stage["sample_outputs"]["chunks"]
    )
    assert any(
        chunk["metadata"].get("chunk_scope_type") == "chart_source"
        and "ppt/embeddings/chartWorkbook.xlsx" in chunk["metadata"].get("chunk_scope_label", "")
        for chunk in chunking_stage["sample_outputs"]["chunks"]
    )
    assert any(
        "ppt/embeddings/chartWorkbook.xlsx" in " ".join([
            str(chunk["metadata"].get("chunk_scope_label", "")),
            " ".join(str(item) for item in chunk["metadata"].get("provenance", [])),
        ])
        and "Forecast Risk" in chunk["metadata"].get("evidence_preview", chunk["text"])
        for chunk in chunking_stage["sample_outputs"]["chunks"]
    )
    assert any(
        chunk["metadata"].get("chunk_scope_type") == "attachment"
        and chunk["metadata"].get("chunk_scope_label") == "attachment operations.xlsx"
        for chunk in chunking_stage["sample_outputs"]["chunks"]
    )
    assert any(
        chunk["metadata"].get("chunk_scope_type") == "table"
        and "attachment operations.xlsx table" in chunk["metadata"].get("chunk_scope_label", "")
        and "Operational Workbook" in chunk["metadata"].get("evidence_preview", chunk["text"])
        for chunk in chunking_stage["sample_outputs"]["chunks"]
    )
    graph_candidates_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_candidates")
    relationship_cards = graph_candidates_stage["sample_outputs"]["relationship_evidence_cards"]
    relationship_card_categories = {card["category"] for card in relationship_cards}
    assert {"connector", "diagram", "presentation_structure", "speaker_notes", "chart_fact", "table_fact", "form_field", "link", "styled_text", "layout"}.issubset(relationship_card_categories)
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["connector"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["diagram"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["presentation_structure"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["speaker_notes"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["chart_fact"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["table_fact"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["link"] >= 1
    assert any(
        card["category"] == "connector"
        and "Product Revenue --[drives]--> Regional Demand" in card["title"]
        for card in relationship_cards
    )
    assert any(
        card["category"] == "chart_fact"
        for card in relationship_cards
    )
    assert any(
        card["category"] == "diagram"
        and "diagram_part=ppt/diagrams/data1.xml" in card.get("provenance", "")
        for card in relationship_cards
    )
    assert any(
        card["category"] == "presentation_structure"
        and "Slide 2: Hidden Assumptions" in card["title"]
        and "slide_visibility=hidden" in card.get("provenance", "")
        for card in relationship_cards
    )
    assert any(
        card["category"] == "speaker_notes"
        and "graph-grounded launch path" in " ".join([card.get("title", ""), card.get("evidence", "")])
        and "notes_relationship_id=rIdNotes1" in card.get("provenance", "")
        and "slide_part=ppt/slides/slide1.xml" in card.get("provenance", "")
        for card in relationship_cards
    )
    assert any(
        card["category"] == "table_fact"
        and card.get("title")
        and ("table=" in card.get("provenance", "") or "table_name=" in card.get("provenance", ""))
        for card in relationship_cards
    )
    assert any(
        card["category"] == "link"
        and "Source Brief" in " ".join([card.get("title", ""), card.get("evidence_preview", "")])
        and "https://example.com/source-brief" in " ".join([card.get("title", ""), card.get("evidence_preview", "")])
        for card in relationship_cards
    )
    profile_stage = next(stage for stage in result["stages"] if stage["name"] == "document_profile")
    assert "forms" in profile_stage["sample_outputs"]["detected_capabilities"]
    assert "layout" in profile_stage["sample_outputs"]["detected_capabilities"]
    assert "presentation_structure" in profile_stage["sample_outputs"]["detected_capabilities"]
    assert "shapes" in profile_stage["sample_outputs"]["detected_capabilities"]
    assert "connectors" in profile_stage["sample_outputs"]["detected_capabilities"]
    assert "styled_text" in profile_stage["sample_outputs"]["detected_capabilities"]
    assert profile_stage["sample_outputs"]["evidence_profile"]["quick_counts"]["shapes"] >= 1
    assert profile_stage["sample_outputs"]["evidence_profile"]["quick_counts"]["connectors"] >= 2
    assert profile_stage["sample_outputs"]["evidence_profile"]["quick_counts"]["diagrams"] >= 5
    assert profile_stage["sample_outputs"]["evidence_profile"]["quick_counts"]["layout_elements"] >= 4
    assert profile_stage["sample_outputs"]["evidence_profile"]["quick_counts"]["slides"] >= 2
    assert profile_stage["sample_outputs"]["evidence_profile"]["quick_counts"]["hidden_slides"] >= 1
    assert profile_stage["sample_outputs"]["evidence_profile"]["quick_counts"]["speaker_notes"] >= 1
    assert profile_stage["sample_outputs"]["evidence_profile"]["quick_counts"]["styled_text_blocks"] >= 1
    assert profile_stage["sample_outputs"]["evidence_profile"]["graph_evidence"]["styled_text_relationship_count"] >= 1
    assert profile_stage["sample_outputs"]["evidence_profile"]["graph_evidence"]["diagram_relationship_count"] >= 1
    assert profile_stage["sample_outputs"]["evidence_profile"]["graph_evidence"]["speaker_notes_relationship_count"] >= 1
    interaction_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_interaction_plan")
    suggested_questions = interaction_stage["sample_outputs"]["suggested_questions"]
    assert any(question["label"] == "Audit connectors" for question in suggested_questions)
    assert any(question["label"] == "Audit diagrams" for question in suggested_questions)
    assert any(question["label"] == "Inspect layout evidence" for question in suggested_questions)
    assert any(question["label"] == "Inspect slide structure" for question in suggested_questions)
    assert any(question["label"] == "Inspect emphasized text" for question in suggested_questions)
    assert any(question["label"] == "Trace links" for question in suggested_questions)
    assert any(question["label"] == "Trace source artifacts" for question in suggested_questions)
    assert any(
        "clicked image" in question["prompt"]
        and "relationship IDs" in question["prompt"]
        for question in suggested_questions
        if question["label"] == "Trace links"
    )
    assert any(
        "Product Revenue" in question["prompt"] and "Regional Demand" in question["prompt"]
        for question in suggested_questions
    )
    assert any(
        "Product Revenue" in question["prompt"]
        and "Regional Demand" in question["prompt"]
        and question.get("source")
        and question.get("target")
        for question in suggested_questions
    )
    assert result["document_profile"]["evidence_profile"]["graph_evidence"]["chart_relationship_count"] >= 1
    assert result["document_profile"]["evidence_profile"]["graph_evidence"]["speaker_notes_relationship_count"] >= 1
    assert result["document_profile"]["evidence_profile"]["graph_evidence"]["link_relationship_count"] >= 3
    assert result["document_profile"]["evidence_profile"]["quick_counts"]["hidden_slides"] >= 1
    assert result["document_profile"]["evidence_profile"]["quick_counts"]["speaker_notes"] >= 1
    assert result["document_profile"]["relationship_type_counts"]["has_hidden_slide"] >= 1
    assert result["document_profile"]["relationship_type_counts"]["has_slide_visibility"] >= 1
    assert result["document_profile"]["relationship_type_counts"]["contains_speaker_notes"] >= 1
    assert result["interaction_handoff"]["recommended_mode"] in {"graph_path_reasoning", "chunk_path_retrieval"}
    assert result["interaction_handoff"]["counts"]["graph_anchors"] >= 1
    assert result["interaction_handoff"]["counts"]["capability_cards"] >= 1
    assert any(card["id"] == "layout" for card in result["interaction_handoff"]["capability_cards"])
    assert any(card["id"] == "diagrams" for card in result["interaction_handoff"]["capability_cards"])
    assert any(card["id"] == "presentation_structure" for card in result["interaction_handoff"]["capability_cards"])
    assert any(card["id"] == "speaker_notes" for card in result["interaction_handoff"]["capability_cards"])
    assert any(card["id"] == "charts" for card in result["interaction_handoff"]["capability_cards"])
    assert any(card["id"] == "links" for card in result["interaction_handoff"]["capability_cards"])
    assert any(anchor["kind"] in {"path", "relationship"} for anchor in result["interaction_handoff"]["graph_anchors"])
    assert any(
        "Product Revenue" in prompt["prompt"]
        and "Regional Demand" in prompt["prompt"]
        and prompt.get("source")
        and prompt.get("target")
        for prompt in result["interaction_handoff"]["starter_prompts"]
    )
    assert any(
        anchor["kind"] == "link"
        and (
            anchor.get("uri") == "https://example.com/risk-register"
            or anchor.get("metadata", {}).get("uri") == "https://example.com/risk-register"
        )
        and anchor.get("metadata", {}).get("image_relationship_id") == "rId9"
        for anchor in result["interaction_handoff"]["graph_anchors"]
    )
    assert "layout" in result["interaction_handoff"]["citation_scopes"]
    assert "chart" in result["interaction_handoff"]["citation_scopes"]
    assert "diagram" in result["interaction_handoff"]["citation_scopes"]
    assert "link" in result["interaction_handoff"]["citation_scopes"]
    assert "slide visibility" in result["interaction_handoff"]["citation_scopes"]
    assert "speaker notes" in result["interaction_handoff"]["citation_scopes"]
    assert result["interaction_handoff"]["counts"]["retrieval_rehearsals"] >= 1
    assert result["interaction_handoff"]["counts"]["rehearsed_chunk_matches"] >= 1
    assert any(
        "Product Revenue" in rehearsal["prompt"]
        and rehearsal["matched_chunk_count"] >= 1
        for rehearsal in result["interaction_handoff"]["retrieval_rehearsals"]
    )
    local_graph_stage = next(stage for stage in result["stages"] if stage["name"] == "local_graph_artifact")
    assert local_graph_stage["counts"]["edges"] >= 20
    assert any(edge["label"] == "contains_connector" for edge in local_graph_stage["sample_outputs"]["edges"])
    local_context_stage = next(stage for stage in result["stages"] if stage["name"] == "local_context_cache")
    assert local_context_stage["counts"]["graph_context_relationships"] >= 1
    assert local_context_stage["counts"]["graph_path_context_paths"] >= 1
    assert local_context_stage["sample_outputs"]["graph_context"]
    assert local_context_stage["sample_outputs"]["graph_paths"]
    reasoning_stage = next(stage for stage in result["stages"] if stage["name"] == "local_graph_reasoning_index")
    assert reasoning_stage["counts"]["bridge_entities"] >= 1
    assert reasoning_stage["counts"]["path_candidates"] >= 1
    assert reasoning_stage["sample_outputs"]["bridge_entities"]
    assert reasoning_stage["sample_outputs"]["path_candidates"]
    assert all(
        path.get("explanation")
        and path.get("answer_hint")
        and path.get("evidence_families")
        for path in reasoning_stage["sample_outputs"]["path_candidates"]
    )
    assert any(
        "Product Revenue" in path["nodes"] and "Regional Demand" in path["nodes"]
        for path in result["local_graph_reasoning"]["path_candidates"]
    )
    connector_path = next(
        path
        for path in result["local_graph_reasoning"]["path_candidates"]
        if "Product Revenue" in path["nodes"] and "Regional Demand" in path["nodes"] and "Risk Map" in path["nodes"]
    )
    assert {"connector", "layout", "semantic"}.issubset(set(connector_path["evidence_families"]))
    assert "Regional Demand connects Product Revenue and Risk Map" in connector_path["explanation"]
    assert "Product Revenue --[drives]-- Regional Demand" in connector_path["explanation"]
    assert "Regional Demand --[references]-- Risk Map" in connector_path["explanation"]
    assert any("slide=Revenue Architecture" in item for item in connector_path["provenance_summary"])
    assert result["summary"]["local_graph"]["edge_count"] >= 20
    assert result["summary"]["local_context"]["cached_chunk_count"] >= 1
    assert result["summary"]["local_context"]["graph_path_context_count"] >= 1
    assert result["summary"]["local_graph_reasoning"]["path_candidate_count"] >= 1
    assert result["local_context"]["chunk_count"] == result["summary"]["chunk_count"]
    assert result["local_context"]["chunks"][0]["content"]
    assert result["local_context"]["chunks"][0]["metadata"].get("filename") == "strategy.pptx"
    assert any(
        "Scope: Slide 1: Revenue Architecture (slide)" in chunk["content"]
        for chunk in result["local_context"]["chunks"]
    )
    assert any(
        "Graph path context:" in chunk["content"]
        and "why=" in chunk["content"]
        and "connects" in chunk["content"]
        for chunk in result["local_context"]["chunks"]
    )
    assert any(
        "Northwind connects product revenue to regional demand." in chunk["content"]
        for chunk in result["local_context"]["chunks"]
    )
    assert any(
        "Scope: chart source ppt/embeddings/chartWorkbook.xlsx" in chunk["content"]
        for chunk in result["local_context"]["chunks"]
    )
    assert any(
        "Scope: attachment operations.xlsx table Operations (table)" in chunk["content"]
        for chunk in result["local_context"]["chunks"]
    )
    assert any(
        "Evidence focus:" in chunk["content"] and "Forecast Risk" in chunk["content"]
        for chunk in result["local_context"]["chunks"]
    )
    assert result["local_graph"]["node_count"] >= 10
    assert any(
        edge["label"] == "contains_connector"
        and edge["target"] == "Product Revenue -> Regional Demand: drives"
        and edge["metadata"].get("connector_id") == "60"
        and edge["metadata"].get("layout_zone") == "middle_center"
        and edge["metadata"].get("layout_bbox") == "x=3.00in y=2.40in w=1.00in h=0.30in"
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        edge["source"] == "Slide 1: Revenue Architecture"
        and edge["label"] == "contains_table"
        and edge["target"] == "R3C1: Strategic Notes"
        and edge["metadata"].get("table_cell_reference") == "R3C1"
        and edge["metadata"].get("table_cell_grid_span") == 2
        and edge["metadata"].get("layout_zone") == "bottom_left"
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        edge["source"] == "Regional Revenue Forecast"
        and edge["label"] == "contains_chart_series"
        and edge["target"] == "Revenue"
        and edge["metadata"].get("chart_label") == "Regional Revenue Forecast"
        and edge["metadata"].get("chart_part") == "ppt/charts/chart1.xml"
        and edge["metadata"].get("chart_series_fill_color") == "#5B9BD5"
        and edge["metadata"].get("chart_series_visual_summary", "").startswith("fill #5B9BD5")
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        edge["source"] == "Regional Revenue Forecast"
        and edge["label"] == "contains_chart_point"
        and edge["target"] == "Revenue Q2"
        and edge["metadata"].get("chart_point_label") == "Revenue Q2"
        and edge["metadata"].get("chart_point_fill_color") == "#C00000"
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        edge["source"] == "Regional Revenue Forecast"
        and edge["label"] == "has_axis"
        and edge["target"] == "Category axis: Quarter"
        and edge["metadata"].get("chart_axis_type") == "category"
        and edge["metadata"].get("chart_axis_title") == "Quarter"
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        edge["source"] == "Regional Revenue Forecast"
        and edge["label"] == "has_axis"
        and edge["target"] == "Value axis: Revenue ($M)"
        and edge["metadata"].get("chart_axis_number_format") == "$#,##0"
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        edge["source"] == "Regional Revenue Forecast"
        and edge["label"] == "has_legend"
        and edge["metadata"].get("chart_legend_position") == "r"
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        edge["source"] == "Regional Revenue Forecast"
        and edge["label"] == "has_data_labels"
        and edge["target"] == "Revenue data labels"
        and "value" in edge["metadata"].get("chart_label_fields", [])
        and edge["metadata"].get("chart_label_position") == "outEnd"
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        edge["source"] == "Regional Revenue Forecast"
        and edge["label"] == "has_trendline"
        and edge["target"] == "Revenue linear trendline"
        and edge["metadata"].get("chart_trendline_type") == "linear"
        and edge["metadata"].get("chart_trendline_line_color") == "#7030A0"
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        edge["source"] == "Revenue Q2"
        and edge["label"] == "has_value"
        and edge["target"] == "18"
        and edge["metadata"].get("category") == "Q2"
        for edge in result["local_graph"]["edges"]
    )
    assert result["local_context"]["graph_context_relationship_count"] >= 1
    assert any(
        "Graph context:" in chunk["content"]
        and "Product Revenue --[drives]--> Regional Demand" in chunk["content"]
        for chunk in result["local_context"]["chunks"]
    )
    assert any(
        "Graph path context:" in chunk["content"]
        and chunk.get("graph_paths")
        for chunk in result["local_context"]["chunks"]
    )
    assert any(
        graph_edge["source"] == "Product Revenue"
        and graph_edge["relation"] == "drives"
        and graph_edge["target"] == "Regional Demand"
        for chunk in result["local_context"]["chunks"]
        for graph_edge in chunk.get("graph_context", [])
    )
    assert any(
        graph_edge.get("chunk_match_score", 0) > 0
        and graph_edge.get("chunk_match_reasons")
        for chunk in result["local_context"]["chunks"]
        for graph_edge in chunk.get("graph_context", [])
    )
    assert any(
        chunk["metadata"].get("graph_context_max_match_score", 0) > 0
        and chunk["metadata"].get("graph_context_match_reasons")
        for chunk in result["local_context"]["chunks"]
    )
    assert any(
        entity["name"] == "GraphRAG Revenue Strategy" and entity["type"] == "document"
        for entity in result["samples"]["entities"]
    )
    assert any(entity["name"] == "Revenue Architecture" for entity in result["samples"]["entities"])
    assert any(entity["name"] == "Launch Plan" and entity["type"] == "diagram" for entity in result["samples"]["entities"])
    assert any(entity["name"] == "Risk Narrative" and entity["type"] == "shape" for entity in result["samples"]["entities"])
    assert any(
        element["type"] == "document_metadata"
        and element["metadata"].get("metadata_field") == "title"
        and element["metadata"].get("metadata_part") == "docProps/core.xml"
        and element["metadata"].get("metadata_value") == "GraphRAG Revenue Strategy"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "slide_metadata"
        and element["metadata"].get("slide_title") == "Hidden Assumptions"
        and element["metadata"].get("slide_visibility") == "hidden"
        and element["metadata"].get("slide_part") == "ppt/slides/slide2.xml"
        and element["metadata"].get("slide_relationship_id") == "rIdSlide2"
        for element in result["samples"]["elements"]
    )
    assert any(
        "Forecast Risk" in element["metadata"].get("emphasis_terms", [])
        and element["metadata"].get("style_summary") == "bold"
        for element in result["samples"]["elements"]
        if element["type"] == "slide_text"
    )
    assert any(
        element["type"] == "slide_text"
        and element["text"] == "Template Governance Note"
        and element["metadata"].get("inherited_from") == "slide_layout"
        and element["metadata"].get("layout_part") == "ppt/slideLayouts/slideLayout1.xml"
        and element["metadata"].get("layout_relationship_id") == "rIdLayout1"
        and element["metadata"].get("placeholder") == "body"
        and element["metadata"].get("style_summary") == "bold"
        and element["metadata"].get("slide_title") == "Revenue Architecture"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "slide_text"
        and element["text"] == "Global Footer: graph evidence required"
        and element["metadata"].get("inherited_from") == "slide_master"
        and element["metadata"].get("master_part") == "ppt/slideMasters/slideMaster1.xml"
        and element["metadata"].get("master_relationship_id") == "rIdMaster1"
        and element["metadata"].get("layout_part") == "ppt/slideLayouts/slideLayout1.xml"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "image"
        and element["metadata"].get("inherited_from") == "slide_layout"
        and element["metadata"].get("image_part") == "ppt/media/layout-logo.png"
        and element["metadata"].get("title") == "Llama Nexus Logo"
        and element["metadata"].get("layout_part") == "ppt/slideLayouts/slideLayout1.xml"
        and element["metadata"].get("layout_zone") == "top_right"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "chart_series"
        and element["location"].startswith("slide 1 chart")
        and element["metadata"].get("slide_title") == "Revenue Architecture"
        and element["metadata"].get("chart_part") == "ppt/charts/chart1.xml"
        and element["metadata"].get("chart_title") == "Regional Revenue Forecast"
        and element["metadata"].get("chart_source_part") == "ppt/embeddings/chartWorkbook.xlsx"
        and element["metadata"].get("chart_source_relationship_id") == "rIdChartWorkbook"
        and element["metadata"].get("chart_source_kind") == "embedded_package"
        and element["metadata"].get("layout_bbox") == "x=1.00in y=3.10in w=4.00in h=1.40in"
        and element["metadata"].get("layout_zone") == "bottom_left"
        and element["metadata"].get("chart_series_fill_color") == "#5B9BD5"
        and element["metadata"].get("chart_series_line_color") == "#1F4E79"
        and element["metadata"].get("chart_series_line_width_points") == 2.0
        and element["metadata"].get("chart_marker_symbol") == "circle"
        and element["metadata"].get("chart_marker_fill_color") == "#70AD47"
        and "fill #5B9BD5" in element["metadata"].get("chart_series_visual_summary", "")
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "chart_point"
        and element["metadata"].get("series_name") == "Revenue"
        and element["metadata"].get("category") == "Q2"
        and element["metadata"].get("chart_point_fill_color") == "#C00000"
        and element["metadata"].get("chart_point_visual_summary") == "fill #C00000"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "chart_axis"
        and element["metadata"].get("chart_axis_type") == "category"
        and element["metadata"].get("chart_axis_title") == "Quarter"
        and element["metadata"].get("chart_axis_position") == "b"
        and element["metadata"].get("chart_part") == "ppt/charts/chart1.xml"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "chart_axis"
        and element["metadata"].get("chart_axis_type") == "value"
        and element["metadata"].get("chart_axis_title") == "Revenue ($M)"
        and element["metadata"].get("chart_axis_number_format") == "$#,##0"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "chart_legend"
        and element["metadata"].get("chart_legend_position") == "r"
        and "Revenue" in element["metadata"].get("chart_legend_series", [])
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "chart_data_label"
        and element["metadata"].get("series_name") == "Revenue"
        and {"series name", "category name", "value"}.issubset(set(element["metadata"].get("chart_label_fields", [])))
        and element["metadata"].get("chart_label_position") == "outEnd"
        and element["metadata"].get("chart_label_number_format") == "$#,##0.0"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "chart_trendline"
        and element["metadata"].get("series_name") == "Revenue"
        and element["metadata"].get("chart_trendline_type") == "linear"
        and element["metadata"].get("chart_trendline_name") == "Revenue trend"
        and element["metadata"].get("chart_trendline_display_equation") is True
        and element["metadata"].get("chart_trendline_display_r_squared") is True
        and element["metadata"].get("chart_trendline_line_color") == "#7030A0"
        and element["metadata"].get("chart_trendline_line_dash") == "dash"
        and "line #7030A0" in element["metadata"].get("chart_trendline_visual_summary", "")
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "table_cell"
        and element["text"] == "Strategic Notes"
        and element["location"] == "slide 1 table 1 cell 3.1"
        and element["metadata"].get("table_cell_reference") == "R3C1"
        and element["metadata"].get("table_cell_grid_span") == 2
        and element["metadata"].get("row") == 3
        and element["metadata"].get("column") == 1
        and element["metadata"].get("style_summary") == "bold; italic; color #7F6000"
        and element["metadata"].get("table_cell_fill_color") == "#FFF2CC"
        and element["metadata"].get("table_cell_text_anchor") == "ctr"
        and element["metadata"].get("table_cell_margin_left_inches") == 0.1
        and element["metadata"].get("table_cell_border_left_color") == "#D6B656"
        and element["metadata"].get("table_cell_border_left_width_inches") == 0.014
        and "fill #FFF2CC" in element["metadata"].get("table_cell_visual_summary", "")
        and element["metadata"].get("layout_bbox") == "x=1.00in y=4.60in w=4.00in h=0.80in"
        and element["metadata"].get("layout_zone") == "bottom_left"
        and "Strategic Notes" in element["metadata"].get("emphasis_terms", [])
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "table_row"
        and element["text"].startswith("Metric: Forecast Risk; Source Owner: Ada Lovelace")
        and element["location"] == "slide 1 chart 1 source 1 sheet 1 row 2"
        and element["metadata"].get("chart_source_part") == "ppt/embeddings/chartWorkbook.xlsx"
        and element["metadata"].get("chart_source_extraction_method") == "xlsx_xml"
        and element["metadata"].get("sheet_name") == "ChartData"
        and element["metadata"].get("formulas", [{}])[0].get("formula") == "SUM(0.4,0.52)"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "image"
        and element["metadata"].get("image_part") == "ppt/media/risk-map.png"
        and element["metadata"].get("slide_title") == "Revenue Architecture"
        and element["metadata"].get("image_width") == 1
        and element["metadata"].get("image_height") == 1
        and element["metadata"].get("image_mime") == "image/png"
        and element["metadata"].get("layout_bbox") == "x=7.00in y=1.00in w=2.00in h=1.50in"
        and element["metadata"].get("layout_zone") == "top_right"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "link"
        and element["metadata"].get("label") == "Source Brief"
        and element["metadata"].get("link_kind") == "click"
        and element["metadata"].get("uri") == "https://example.com/source-brief"
        and element["metadata"].get("relationship_id") == "rIdLink1"
        and element["metadata"].get("target_mode") == "External"
        and element["metadata"].get("slide_title") == "Revenue Architecture"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "link"
        and element["metadata"].get("label") == "Risk Map"
        and element["metadata"].get("link_kind") == "click"
        and element["metadata"].get("uri") == "https://example.com/risk-register"
        and element["metadata"].get("relationship_id") == "rIdImageLink"
        and element["metadata"].get("image_relationship_id") == "rId9"
        and element["metadata"].get("image_part") == "ppt/media/risk-map.png"
        and element["metadata"].get("image_width") == 1
        and element["metadata"].get("image_height") == 1
        and element["metadata"].get("image_mime") == "image/png"
        and element["metadata"].get("layout_bbox") == "x=7.00in y=1.00in w=2.00in h=1.50in"
        and element["metadata"].get("layout_zone") == "top_right"
        and element["metadata"].get("tooltip") == "Open risk register"
        and element["metadata"].get("slide_title") == "Revenue Architecture"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "link"
        and element["metadata"].get("label") == "Llama Nexus Logo"
        and element["metadata"].get("link_kind") == "click"
        and element["metadata"].get("uri") == "https://example.com/brand-kit"
        and element["metadata"].get("relationship_id") == "rIdLayoutLogoLink"
        and element["metadata"].get("image_relationship_id") == "rIdLayoutLogo"
        and element["metadata"].get("image_part") == "ppt/media/layout-logo.png"
        and element["metadata"].get("inherited_from") == "slide_layout"
        and element["metadata"].get("layout_part") == "ppt/slideLayouts/slideLayout1.xml"
        and element["metadata"].get("layout_zone") == "top_right"
        and element["metadata"].get("tooltip") == "Open brand kit"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "comment"
        and "Validate regional risk narrative" in element["text"]
        and element["metadata"].get("author") == "Ada Lovelace"
        and element["metadata"].get("comment_part") == "ppt/comments/comment1.xml"
        and element["metadata"].get("slide_title") == "Revenue Architecture"
        and element["metadata"].get("layout_position") == "x=2.00in y=1.00in"
        and element["metadata"].get("layout_left_pct") == 20.0
        and element["metadata"].get("layout_top_pct") == 17.8
        and element["metadata"].get("layout_zone") == "top_left"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "speaker_notes"
        and "graph-grounded launch path" in element["text"]
        and element["metadata"].get("notes_part") == "ppt/notesSlides/notesSlide1.xml"
        and element["metadata"].get("notes_relationship_id") == "rIdNotes1"
        and element["metadata"].get("slide_title") == "Revenue Architecture"
        and element["metadata"].get("slide_part") == "ppt/slides/slide1.xml"
        and element["metadata"].get("slide_relationship_id") == "rIdSlide1"
        and element["metadata"].get("slide_visibility") == "visible"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "media"
        and element["metadata"].get("media_part") == "ppt/media/narration.mp3"
        and element["metadata"].get("media_kind") == "audio"
        and element["metadata"].get("media_mime") == "audio/mpeg"
        and element["metadata"].get("slide_title") == "Revenue Architecture"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "media"
        and element["metadata"].get("uri") == "https://example.com/launch-demo.mp4"
        and element["metadata"].get("media_kind") == "video"
        and element["metadata"].get("target_mode") == "External"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "embedded_object"
        and element["metadata"].get("media_part") == "ppt/embeddings/oleObject1.bin"
        and element["metadata"].get("media_kind") == "embedded_object"
        and element["metadata"].get("relationship_id") == "rIdOle1"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "embedded_object"
        and element["metadata"].get("media_part") == "ppt/embeddings/operations.xlsx"
        and element["metadata"].get("media_kind") == "embedded_package"
        and element["metadata"].get("relationship_id") == "rIdPackage1"
        and element["metadata"].get("embedded_file_filename") == "operations.xlsx"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "table_row"
        and element["text"] == "Artifact: Operational Workbook; Owner: Grace Hopper; Status: Ready"
        and element["metadata"].get("embedded_file_filename") == "operations.xlsx"
        and element["metadata"].get("embedded_file_extraction_method") == "xlsx_xml"
        and element["metadata"].get("sheet_name") == "Operations"
        and element["metadata"].get("media_part") == "ppt/embeddings/operations.xlsx"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "form_field"
        and element["metadata"].get("form_kind") == "pptx_control"
        and element["metadata"].get("label") == "Approved for launch"
        and element["metadata"].get("value") == "True"
        and element["metadata"].get("field_type") == "Forms.CheckBox.1"
        and element["metadata"].get("control_part") == "ppt/activeX/activeX1.xml"
        and element["metadata"].get("control_id") == "_x0000_s1025"
        and element["metadata"].get("form_field_alias") == "Approved for launch"
        and element["metadata"].get("form_field_tag") == "LaunchApproval"
        and element["metadata"].get("form_field_id") == "_x0000_s1025"
        and element["metadata"].get("form_field_checked") is True
        and element["metadata"].get("relationship_id") == "rIdControl1"
        and element["metadata"].get("slide_title") == "Revenue Architecture"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "form_field"
        and element["metadata"].get("form_kind") == "pptx_control"
        and element["metadata"].get("label") == "Launch region"
        and element["metadata"].get("value") == "Europe"
        and element["metadata"].get("field_type") == "Forms.ComboBox.1"
        and element["metadata"].get("control_part") == "ppt/activeX/activeX2.xml"
        and element["metadata"].get("control_id") == "_x0000_s1026"
        and element["metadata"].get("form_field_alias") == "Launch region"
        and element["metadata"].get("form_field_tag") == "LaunchRegion"
        and element["metadata"].get("form_field_id") == "_x0000_s1026"
        and element["metadata"].get("form_field_options") == ["North America", "Europe", "APAC"]
        and element["metadata"].get("form_field_selected_option") == "Europe"
        and element["metadata"].get("relationship_id") == "rIdControl2"
        and element["metadata"].get("slide_title") == "Revenue Architecture"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "diagram_node"
        and element["text"] == "Risk Review"
        and element["metadata"].get("diagram_part") == "ppt/diagrams/data1.xml"
        and element["metadata"].get("parent_label") == "Launch Plan"
        and element["metadata"].get("diagram_level") == 1
        and element["metadata"].get("slide_title") == "Revenue Architecture"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "diagram_relationship"
        and element["metadata"].get("source_label") == "Launch Plan"
        and element["metadata"].get("relation") == "contains"
        and element["metadata"].get("target_label") == "Risk Review"
        and element["metadata"].get("relationship_id") == "rIdDiagram1"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "shape_metadata"
        and element["text"] == "Risk Narrative: Callout shape describing regional launch blockers"
        and element["metadata"].get("shape_id") == "44"
        and element["metadata"].get("shape_name") == "Risk Narrative Callout"
        and element["metadata"].get("shape_title") == "Risk Narrative"
        and element["metadata"].get("shape_description") == "Callout shape describing regional launch blockers"
        and element["metadata"].get("slide_title") == "Revenue Architecture"
        and element["metadata"].get("layout_zone") == "middle_right"
        and element["metadata"].get("layout_bbox") == "x=7.00in y=2.75in w=2.00in h=0.75in"
        and element["metadata"].get("layout_center_x_pct") == 80.0
        and element["metadata"].get("slide_width_inches") == 10.0
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "shape_connector"
        and element["text"] == "Product Revenue -> Regional Demand: drives"
        and element["metadata"].get("connector_id") == "60"
        and element["metadata"].get("connector_title") == "drives"
        and element["metadata"].get("connector_description") == "Product revenue drives regional demand"
        and element["metadata"].get("source_shape_id") == "20"
        and element["metadata"].get("target_shape_id") == "21"
        and element["metadata"].get("source_label") == "Product Revenue"
        and element["metadata"].get("target_label") == "Regional Demand"
        and element["metadata"].get("relation") == "drives"
        and element["metadata"].get("layout_bbox") == "x=3.00in y=2.40in w=1.00in h=0.30in"
        and element["metadata"].get("layout_zone") == "middle_center"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "shape_connector"
        and element["text"] == "Regional Demand -> Risk Map: references"
        and element["metadata"].get("connector_id") == "61"
        and element["metadata"].get("connector_title") == "references"
        and element["metadata"].get("source_shape_id") == "21"
        and element["metadata"].get("target_shape_id") == "4"
        and element["metadata"].get("source_label") == "Regional Demand"
        and element["metadata"].get("target_label") == "Risk Map"
        and element["metadata"].get("relation") == "references"
        for element in result["samples"]["elements"]
    )

    relationships = [
        *result["samples"]["relationships"],
        *[
            {
                "source": edge.get("source"),
                "relation": edge.get("label"),
                "target": edge.get("target"),
                "metadata": edge.get("metadata", {}),
                "evidence": edge.get("evidence"),
            }
            for edge in result["local_graph"]["edges"]
        ],
    ]
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "created_by"
        and relationship["target"] == "Ada Lovelace"
        and relationship["metadata"].get("metadata_field") == "creator"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "has_keyword"
        and relationship["target"] == "GraphRAG"
        and relationship["metadata"].get("metadata_part") == "docProps/core.xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "created_with"
        and relationship["target"] == "Microsoft PowerPoint"
        and relationship["metadata"].get("metadata_field") == "application"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "has_company"
        and relationship["target"] == "Llama Nexus Labs"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "links_to"
        and relationship["target"] == "https://example.com/source-brief"
        and relationship["metadata"].get("slide_title") == "Revenue Architecture"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "contains_link"
        and relationship["target"] == "Source Brief"
        and relationship["metadata"].get("uri") == "https://example.com/source-brief"
        and relationship["metadata"].get("container_type") == "slide"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Source Brief"
        and relationship["relation"] == "links_to"
        and relationship["target"] == "https://example.com/source-brief"
        and relationship["metadata"].get("slide_title") == "Revenue Architecture"
        and relationship["metadata"].get("link_label") == "Source Brief"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "links_to"
        and relationship["target"] == "https://example.com/risk-register"
        and relationship["metadata"].get("relationship_id") == "rIdImageLink"
        and relationship["metadata"].get("image_relationship_id") == "rId9"
        and relationship["metadata"].get("image_part") == "ppt/media/risk-map.png"
        and relationship["metadata"].get("image_mime") == "image/png"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "contains_link"
        and relationship["target"] == "Risk Map"
        and relationship["metadata"].get("uri") == "https://example.com/risk-register"
        and relationship["metadata"].get("image_relationship_id") == "rId9"
        and relationship["metadata"].get("layout_zone") == "top_right"
        and relationship["metadata"].get("container_type") == "slide"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Risk Map"
        and relationship["relation"] == "links_to"
        and relationship["target"] == "https://example.com/risk-register"
        and relationship["metadata"].get("link_label") == "Risk Map"
        and relationship["metadata"].get("image_relationship_id") == "rId9"
        and relationship["metadata"].get("image_part") == "ppt/media/risk-map.png"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "contains_link"
        and relationship["target"] == "Llama Nexus Logo"
        and relationship["metadata"].get("uri") == "https://example.com/brand-kit"
        and relationship["metadata"].get("image_relationship_id") == "rIdLayoutLogo"
        and relationship["metadata"].get("inherited_from") == "slide_layout"
        and relationship["metadata"].get("layout_part") == "ppt/slideLayouts/slideLayout1.xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "has_comment"
        and "Validate regional risk narrative" in relationship["target"]
        and relationship["metadata"].get("comment_part") == "ppt/comments/comment1.xml"
        and relationship["metadata"].get("layout_position") == "x=2.00in y=1.00in"
        and relationship["metadata"].get("layout_zone") == "top_left"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Ada Lovelace"
        and relationship["relation"] == "commented_on"
        and relationship["target"] == "strategy.pptx"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "has_speaker_notes"
        and "graph-grounded launch path" in relationship["target"]
        and relationship["metadata"].get("notes_part") == "ppt/notesSlides/notesSlide1.xml"
        and relationship["metadata"].get("notes_relationship_id") == "rIdNotes1"
        and relationship["metadata"].get("slide_part") == "ppt/slides/slide1.xml"
        and relationship["metadata"].get("slide_visibility") == "visible"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "contains_speaker_notes"
        and "graph-grounded launch path" in relationship["target"]
        and relationship["metadata"].get("notes_part") == "ppt/notesSlides/notesSlide1.xml"
        and relationship["metadata"].get("notes_relationship_id") == "rIdNotes1"
        and relationship["metadata"].get("container_type") == "slide"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "emphasizes"
        and relationship["target"] == "Strategic Notes"
        and relationship["metadata"].get("table_cell_reference") == "R3C1"
        and relationship["metadata"].get("table_cell_grid_span") == 2
        and relationship["metadata"].get("table_cell_fill_color") == "#FFF2CC"
        and relationship["metadata"].get("table_cell_visual_summary", "").startswith("fill #FFF2CC")
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "contains_media"
        and relationship["target"] == "ppt/media/narration.mp3"
        and relationship["metadata"].get("media_kind") == "audio"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "links_to"
        and relationship["target"] == "https://example.com/launch-demo.mp4"
        and relationship["metadata"].get("media_kind") == "video"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "contains_embedded_object"
        and relationship["target"] == "oleObject1.bin"
        and relationship["metadata"].get("relationship_id") == "rIdOle1"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "contains_embedded_object"
        and relationship["target"] == "operations.xlsx"
        and relationship["metadata"].get("relationship_id") == "rIdPackage1"
        and relationship["metadata"].get("embedded_file_filename") == "operations.xlsx"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Operational Workbook"
        and relationship["relation"] == "owner"
        and relationship["target"] == "Grace Hopper"
        and relationship["metadata"].get("embedded_file_filename") == "operations.xlsx"
        and relationship["metadata"].get("embedded_file_extraction_method") == "xlsx_xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "has_field"
        and relationship["target"] == "Approved for launch"
        and relationship["metadata"].get("form_kind") == "pptx_control"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Approved for launch"
        and relationship["relation"] == "has_value"
        and relationship["target"] == "checked"
        and relationship["metadata"].get("control_part") == "ppt/activeX/activeX1.xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "contains_diagram_node"
        and relationship["target"] == "Launch Plan"
        and relationship["metadata"].get("diagram_part") == "ppt/diagrams/data1.xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Launch Plan"
        and relationship["relation"] == "contains"
        and relationship["target"] == "Risk Review"
        and relationship["metadata"].get("diagram_connection_type") == "parOf"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "GraphRAG"
        and relationship["relation"] == "region"
        and relationship["target"] == "North America"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Revenue"
        and relationship["relation"] == "q1_value"
        and relationship["target"] == "12"
        and relationship["metadata"].get("slide_title") == "Revenue Architecture"
        and relationship["metadata"].get("chart_part") == "ppt/charts/chart1.xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "uses_chart_source"
        and relationship["target"] == "ppt/embeddings/chartWorkbook.xlsx"
        and relationship["metadata"].get("chart_source_kind") == "embedded_package"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Forecast Risk"
        and relationship["relation"] == "source_owner"
        and relationship["target"] == "Ada Lovelace"
        and relationship["metadata"].get("chart_source_part") == "ppt/embeddings/chartWorkbook.xlsx"
        and relationship["metadata"].get("chart_source_extraction_method") == "xlsx_xml"
        and relationship["metadata"].get("sheet_name") == "ChartData"
        for relationship in relationships
    )
    assert any(
        relationship["relation"] == "contains_image"
        and relationship["metadata"].get("image_part") == "ppt/media/risk-map.png"
        and relationship["metadata"].get("slide_title") == "Revenue Architecture"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "has_slide"
        and relationship["target"] == "Slide 1: Revenue Architecture"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "has_slide"
        and relationship["target"] == "Slide 2: Hidden Assumptions"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "has_hidden_slide"
        and relationship["target"] == "Slide 2: Hidden Assumptions"
        and relationship["metadata"].get("slide_visibility") == "hidden"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 2: Hidden Assumptions"
        and relationship["relation"] == "has_slide_visibility"
        and relationship["target"] == "hidden"
        and relationship["metadata"].get("slide_part") == "ppt/slides/slide2.xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 2: Hidden Assumptions"
        and relationship["relation"] == "uses_slide_part"
        and relationship["target"] == "ppt/slides/slide2.xml"
        and relationship["metadata"].get("slide_relationship_id") == "rIdSlide2"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "contains_chart"
        and relationship["target"] == "ppt/charts/chart1.xml"
        and relationship["metadata"].get("container_type") == "slide"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Regional Revenue Forecast"
        and relationship["relation"] == "has_trendline"
        and relationship["target"] == "Revenue linear trendline"
        and relationship["metadata"].get("chart_trendline_type") == "linear"
        and relationship["metadata"].get("chart_trendline_visual_summary", "").startswith("line #7030A0")
        for relationship in relationships
    )
    assert any(
        edge["source"] == "Slide 1: Revenue Architecture"
        and edge["label"] == "contains_image"
        and edge["target"] == "Risk Map"
        and edge["metadata"].get("image_part") == "ppt/media/risk-map.png"
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        edge["source"] == "Slide 1: Revenue Architecture"
        and edge["label"] == "contains_link"
        and edge["target"] == "Risk Map"
        and edge["metadata"].get("uri") == "https://example.com/risk-register"
        and edge["metadata"].get("image_relationship_id") == "rId9"
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        edge["source"] == "Slide 1: Revenue Architecture"
        and edge["label"] == "contains_link"
        and edge["target"] == "Llama Nexus Logo"
        and edge["metadata"].get("uri") == "https://example.com/brand-kit"
        and edge["metadata"].get("image_relationship_id") == "rIdLayoutLogo"
        and edge["metadata"].get("inherited_from") == "slide_layout"
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "contains_text"
        and relationship["target"] == "Template Governance Note"
        and relationship["metadata"].get("inherited_from") == "slide_layout"
        and relationship["metadata"].get("layout_part") == "ppt/slideLayouts/slideLayout1.xml"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "contains_text"
        and relationship["target"] == "Global Footer: graph evidence required"
        and relationship["metadata"].get("inherited_from") == "slide_master"
        and relationship["metadata"].get("master_part") == "ppt/slideMasters/slideMaster1.xml"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "contains_image"
        and relationship["target"] == "Llama Nexus Logo"
        and relationship["metadata"].get("inherited_from") == "slide_layout"
        and relationship["metadata"].get("image_part") == "ppt/media/layout-logo.png"
        for relationship in result["samples"]["relationships"]
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "contains_diagram_node"
        and relationship["target"] == "Launch Plan"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "contains_form_field"
        and relationship["target"] == "Approved for launch"
        and relationship["metadata"].get("control_part") == "ppt/activeX/activeX1.xml"
        and relationship["metadata"].get("form_field_checked") is True
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "contains_form_field"
        and relationship["target"] == "Launch region"
        and relationship["metadata"].get("control_part") == "ppt/activeX/activeX2.xml"
        and relationship["metadata"].get("form_field_options") == ["North America", "Europe", "APAC"]
        and relationship["metadata"].get("form_field_selected_option") == "Europe"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Launch region"
        and relationship["relation"] == "has_value"
        and relationship["target"] == "Europe"
        and relationship["metadata"].get("form_kind") == "pptx_control"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "contains_shape"
        and relationship["target"] == "Risk Narrative"
        and relationship["metadata"].get("shape_description") == "Callout shape describing regional launch blockers"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "contains_shape"
        and relationship["target"] == "Risk Narrative"
        and relationship["metadata"].get("container_type") == "slide"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Risk Narrative"
        and relationship["relation"] == "has_description"
        and relationship["target"] == "Callout shape describing regional launch blockers"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "strategy.pptx"
        and relationship["relation"] == "contains_connector"
        and relationship["target"] == "Product Revenue -> Regional Demand: drives"
        and relationship["metadata"].get("connector_id") == "60"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Product Revenue"
        and relationship["relation"] == "drives"
        and relationship["target"] == "Regional Demand"
        and relationship["metadata"].get("connector_description") == "Product revenue drives regional demand"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Regional Demand"
        and relationship["relation"] == "references"
        and relationship["target"] == "Risk Map"
        and relationship["metadata"].get("connector_description") == "Regional demand references the risk map"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "contains_connector"
        and relationship["target"] == "Product Revenue -> Regional Demand: drives"
        and relationship["metadata"].get("container_type") == "slide"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "has_section"
        and relationship["target"] == "Revenue Architecture"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Slide 1: Revenue Architecture"
        and relationship["relation"] == "emphasizes"
        and relationship["target"] == "Forecast Risk"
        and relationship["metadata"].get("styled_text") is True
        and relationship["metadata"].get("style_summary") == "bold"
        and relationship["metadata"].get("emphasis_styles") == ["bold"]
        for relationship in relationships
    )
    assert any(
        edge["source"] == "Slide 1: Revenue Architecture"
        and edge["label"] == "contains_text"
        and edge["target"] == "Template Governance Note"
        and edge["metadata"].get("inherited_from") == "slide_layout"
        and edge["metadata"].get("layout_part") == "ppt/slideLayouts/slideLayout1.xml"
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        edge["source"] == "Slide 1: Revenue Architecture"
        and edge["label"] == "contains_text"
        and edge["target"] == "Global Footer: graph evidence required"
        and edge["metadata"].get("inherited_from") == "slide_master"
        and edge["metadata"].get("master_part") == "ppt/slideMasters/slideMaster1.xml"
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        relationship["source"] == "Revenue Architecture"
        and relationship["relation"] == "contains_text"
        and relationship["target"] == "Northwind connects product revenue to regional demand."
        and relationship["metadata"].get("section_label") == "Revenue Architecture"
        for relationship in relationships
    )
    embedding_stage = next(stage for stage in result["stages"] if stage["name"] == "embedding_plan")
    embedding_samples = [
        *embedding_stage["sample_outputs"]["embedding_inputs"],
        *embedding_stage["sample_outputs"]["representative_embedding_inputs"],
    ]
    embedding_text = "\n".join(item["text"] for item in embedding_samples)
    embedding_provenance = [
        provenance
        for item in embedding_samples
        for provenance in item["metadata"].get("provenance", [])
    ]
    embedding_blob = "\n".join([embedding_text, *embedding_provenance])
    assert embedding_stage["sample_outputs"]["embedding_sample_categories"]["link"] >= 1
    assert "slide_title=Revenue Architecture" in embedding_text
    assert "slide_title=Hidden Assumptions" in embedding_text
    assert "slide_visibility=hidden" in embedding_text
    assert "slide_part=ppt/slides/slide2.xml" in embedding_text
    assert "layout_bbox=x=7.00in y=2.75in w=2.00in h=0.75in" in embedding_text
    assert "layout_zone=middle_right" in embedding_text
    assert "Confidential baseline assumptions support regional demand." in embedding_text
    assert "metadata_part=docProps/core.xml" in embedding_text
    assert "metadata_field=title" in embedding_text
    assert "metadata_value=GraphRAG Revenue Strategy" in embedding_text
    assert "style_summary=bold" in embedding_text
    assert 'emphasis_terms=["Forecast Risk"]' in embedding_text
    assert "inherited_from=slide_layout" in embedding_text
    assert "layout_part=ppt/slideLayouts/slideLayout1.xml" in embedding_text
    assert "Template Governance Note" in embedding_text
    assert "inherited_from=slide_master" in embedding_text
    assert "master_part=ppt/slideMasters/slideMaster1.xml" in embedding_text
    assert "Global Footer: graph evidence required" in embedding_text
    assert "chart_part=ppt/charts/chart1.xml" in embedding_text
    assert "chart_axis_title=Quarter" in embedding_text
    assert "chart_axis_title=Revenue ($M)" in embedding_text
    assert "chart_legend_position=r" in embedding_text
    assert 'chart_label_fields=["series name", "category name", "value"]' in embedding_text
    assert "fill #5B9BD5" in embedding_text
    assert "fill #C00000" in embedding_text
    assert "chart_label_position=outEnd" in embedding_text
    assert "Revenue linear trendline" in embedding_text
    assert "line #7030A0" in embedding_text
    assert "table_cell_reference=R3C1" in embedding_text
    assert "table_cell_grid_span=2" in embedding_text
    assert "table_cell_fill_color=#FFF2CC" in embedding_text
    assert "table_cell_visual_summary=fill #FFF2CC" in embedding_text
    assert "Strategic Notes" in embedding_text
    assert "chart_source_part=ppt/embeddings/chartWorkbook.xlsx" in embedding_text
    assert "chart_source_extraction_method=xlsx_xml" in embedding_text
    assert "sheet_name=ChartData" in embedding_text
    assert "Metric: Forecast Risk; Source Owner: Ada Lovelace; Confidence: 0.92" in embedding_text
    assert "chart_source_kind=embedded_package" in embedding_text
    assert "embedded_file_filename=operations.xlsx" in embedding_text
    assert "media_part=ppt/embeddings/operations.xlsx" in embedding_text
    assert "Artifact: Operational Workbook; Owner: Grace Hopper; Status: Ready" in embedding_text
    assert "uri=https://example.com/source-brief" in embedding_text
    assert "uri=https://example.com/risk-register" in embedding_blob
    assert "uri=https://example.com/brand-kit" in embedding_blob
    assert "image_relationship_id=rId9" in embedding_blob
    assert "image_relationship_id=rIdLayoutLogo" in embedding_blob
    assert "diagram_part=ppt/diagrams/data1.xml" in embedding_text
    assert "diagram_node_id=root" in embedding_text
    assert "control_part=ppt/activeX/activeX1.xml" in embedding_text
    assert "field_type=Forms.CheckBox.1" in embedding_text
    assert "form_field_checked=True" in embedding_text
    assert "control_part=ppt/activeX/activeX2.xml" in embedding_text
    assert "field_type=Forms.ComboBox.1" in embedding_text
    assert "form_field_options=[\"North America\", \"Europe\", \"APAC\"]" in embedding_text
    assert "form_field_selected_option=Europe" in embedding_text
    assert any("comment_part=ppt/comments/comment1.xml" in provenance for provenance in embedding_provenance)
    assert any("author=Ada Lovelace" in provenance for provenance in embedding_provenance)
    assert any("layout_position=x=2.00in y=1.00in" in provenance for provenance in embedding_provenance)
    assert any("layout_zone=top_left" in provenance for provenance in embedding_provenance)
    assert any("notes_part=ppt/notesSlides/notesSlide1.xml" in provenance for provenance in embedding_provenance)
    assert any("notes_relationship_id=rIdNotes1" in provenance for provenance in embedding_provenance)
    assert any(
        "notes_part=ppt/notesSlides/notesSlide1.xml" in provenance
        and "slide_part=ppt/slides/slide1.xml" in provenance
        for provenance in embedding_provenance
    )
    assert any("uri=https://example.com/launch-demo.mp4" in provenance for provenance in embedding_provenance)
    assert any(
        "uri=https://example.com/risk-register" in provenance
        and "image_relationship_id=rId9" in provenance
        and "image_part=ppt/media/risk-map.png" in provenance
        for provenance in embedding_provenance
    )
    assert any(
        "uri=https://example.com/brand-kit" in provenance
        and "image_relationship_id=rIdLayoutLogo" in provenance
        and "layout_part=ppt/slideLayouts/slideLayout1.xml" in provenance
        for provenance in embedding_provenance
    )
    assert any("chart_source_part=ppt/embeddings/chartWorkbook.xlsx" in provenance for provenance in embedding_provenance)
    assert any("chart_axis_title=Quarter" in provenance for provenance in embedding_provenance)
    assert any("chart_legend_position=r" in provenance for provenance in embedding_provenance)
    assert any("table_cell_reference=R3C1" in provenance for provenance in embedding_provenance)
    assert any("table_cell_grid_span=2" in provenance for provenance in embedding_provenance)
    assert any("table_cell_fill_color=#FFF2CC" in provenance for provenance in embedding_provenance)
    assert any("table_cell_visual_summary=fill #FFF2CC" in provenance for provenance in embedding_provenance)
    assert any("layout_bbox=x=1.00in y=4.60in w=4.00in h=0.80in" in provenance for provenance in embedding_provenance)
    assert any("image_part=ppt/media/risk-map.png" in provenance for provenance in embedding_provenance)
    assert any("layout_zone=top_right" in provenance for provenance in embedding_provenance)
    assert any("image_part=ppt/media/layout-logo.png" in provenance for provenance in embedding_provenance)
    assert any("layout_part=ppt/slideLayouts/slideLayout1.xml" in provenance for provenance in embedding_provenance)
    assert any("master_part=ppt/slideMasters/slideMaster1.xml" in provenance for provenance in embedding_provenance)
    assert any("image_width=1" in provenance for provenance in embedding_provenance)
    assert any("image_mime=image/png" in provenance for provenance in embedding_provenance)
    assert any("media_part=ppt/media/narration.mp3" in provenance for provenance in embedding_provenance)
    assert any("media_kind=audio" in provenance for provenance in embedding_provenance)
    assert any("media_part=ppt/embeddings/oleObject1.bin" in provenance for provenance in embedding_provenance)
    assert any("embedded_file_filename=operations.xlsx" in provenance for provenance in embedding_provenance)
    assert any("embedded_file_extraction_method=xlsx_xml" in provenance for provenance in embedding_provenance)
    assert any("form_kind=pptx_control" in provenance for provenance in embedding_provenance)
    assert any("form_field_checked=True" in provenance for provenance in embedding_provenance)
    assert any("form_field_selected_option=Europe" in provenance for provenance in embedding_provenance)
    assert any("slide_visibility=hidden" in provenance for provenance in embedding_provenance)
    assert any("slide_part=ppt/slides/slide2.xml" in provenance for provenance in embedding_provenance)
    assert any("source_label=Launch Plan" in provenance for provenance in embedding_provenance)
    assert "shape_title=Risk Narrative" in embedding_text
    assert "shape_description=Callout shape describing regional launch blockers" in embedding_text
    assert "connector_title=drives" in embedding_text
    assert "connector_description=Product revenue drives regional demand" in embedding_text
    assert "source_shape_id=20" in embedding_text
    assert "target_shape_id=21" in embedding_text
    assert "Graph context:" in embedding_text
    assert "Scope: Slide 1: Revenue Architecture (slide)" in embedding_text
    assert "Scope: chart source ppt/embeddings/chartWorkbook.xlsx" in embedding_text
    assert "Scope: attachment operations.xlsx table Operations (table)" in embedding_text
    assert "Evidence focus:" in embedding_text
    assert "Slide 1: Revenue Architecture --[emphasizes]--> Forecast Risk" in embedding_text
    assert "Forecast Risk --[source_owner]--> Ada Lovelace" in embedding_text
    assert "Operational Workbook --[owner]--> Grace Hopper" in embedding_text
    assert "Product Revenue --[drives]--> Regional Demand" in embedding_text
    assert (
        "Regional Demand --[references]--> Risk Map" in embedding_text
        or "Regional Demand -> Risk Map: references" in embedding_text
    )
    assert "Graph path context:" in embedding_text
    assert "score=" in embedding_text and "match=" in embedding_text
    assert any(
        item["metadata"].get("graph_context_relationship_count", 0) >= 1
        and item["metadata"].get("graph_context_max_match_score", 0) > 0
        and item["metadata"].get("graph_context_match_reasons")
        for item in embedding_stage["sample_outputs"]["embedding_inputs"]
    )
    assert any(
        item["metadata"].get("graph_path_context_count", 0) >= 1
        and item["metadata"].get("graph_path_max_match_score", 0) > 0
        and item["metadata"].get("graph_path_match_reasons")
        and item["metadata"].get("graph_paths")
        for item in embedding_stage["sample_outputs"]["embedding_inputs"]
    )
    assert any(
        graph_edge["source"] == "Product Revenue"
        and graph_edge["relation"] == "drives"
        and graph_edge["target"] == "Regional Demand"
        and graph_edge.get("chunk_match_score", 0) > 0
        and graph_edge.get("chunk_match_reasons")
        for item in embedding_stage["sample_outputs"]["embedding_inputs"]
        for graph_edge in item["metadata"].get("graph_context", [])
    )


@pytest.mark.asyncio
async def test_pipeline_preview_uses_representative_samples_for_complex_powerpoint():
    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload(
        "strategy.pptx",
        _minimal_pptx_bytes(),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="slides", sample_limit=5)

    chunking_stage = next(stage for stage in result["stages"] if stage["name"] == "chunking")
    chunk_cards = chunking_stage["sample_outputs"]["chunk_evidence_cards"]
    chunk_categories = {card["category"] for card in chunk_cards}
    chunk_families = {
        family
        for card in chunk_cards
        for family in card.get("categories", [card["category"]])
    }
    assert {"connector", "chart_source", "table", "link", "styled_text"}.issubset(chunk_families)
    assert {"diagram", "image"}.issubset(chunk_families)
    assert any(
        card["category"] == "chart_source"
        and "ppt/embeddings/chartWorkbook.xlsx" in " ".join([card.get("title", ""), card.get("evidence_preview", "")])
        for card in chunk_cards
    )
    assert any(
        card["category"] == "table"
        and (
            "Operational Workbook" in card.get("evidence_preview", "")
            or "Product | Region" in card.get("evidence_preview", "")
        )
        for card in chunk_cards
    )

    embedding_stage = next(stage for stage in result["stages"] if stage["name"] == "embedding_plan")
    representative_inputs = embedding_stage["sample_outputs"]["representative_embedding_inputs"]
    representative_text = "\n".join(item["text"] for item in representative_inputs)
    embedding_cards = embedding_stage["sample_outputs"]["embedding_evidence_cards"]
    embedding_categories = {card["category"] for card in embedding_cards}
    embedding_families = {
        family
        for card in embedding_cards
        for family in card.get("categories", [card["category"]])
    }

    assert {"connector", "chart_source", "table", "link", "styled_text"}.issubset(embedding_families)
    assert {"diagram", "image"}.issubset(embedding_families)
    assert "Scope: chart source ppt/embeddings/chartWorkbook.xlsx" in representative_text
    assert (
        "Scope: attachment operations.xlsx table Operations (table)" in representative_text
        or "Scope: Slide 1: Revenue Architecture table 1" in representative_text
    )
    assert (
        "Product Revenue -> Regional Demand: drives" in representative_text
        or "Regional Demand -> Risk Map: references" in representative_text
    )
    assert embedding_stage["sample_outputs"]["embedding_sample_categories"]["chart_source"] >= 1
    assert embedding_stage["sample_outputs"]["embedding_evidence_family_counts"]["table"] >= 1
    assert embedding_stage["sample_outputs"]["embedding_evidence_family_counts"]["link"] >= 1
    assert embedding_stage["sample_outputs"]["embedding_evidence_family_counts"]["diagram"] >= 1
    assert embedding_stage["counts"]["representative_graph_context_relationships"] >= 1
    assert embedding_stage["counts"]["graph_enriched_representative_inputs"] >= 1


@pytest.mark.asyncio
async def test_pipeline_preview_routes_macro_enabled_ooxml_variants_to_structured_extractors():
    ingest = load_graphrag_route_module("ingest")
    cases = [
        (
            "strategy.pptm",
            _minimal_pptx_bytes(),
            "application/vnd.ms-powerpoint.presentation.macroEnabled.12",
            "pptx_xml",
            {"presentation_structure", "charts", "tables", "shapes", "connectors", "links", "forms"},
        ),
        (
            "implementation.docm",
            _minimal_docx_bytes(),
            "application/vnd.ms-word.document.macroEnabled.12",
            "docx_xml",
            {"tables", "charts", "diagrams", "forms", "comments", "revisions", "links", "styled_text"},
        ),
        (
            "scorecard.xlsm",
            _minimal_xlsx_bytes(),
            "application/vnd.ms-excel.sheet.macroEnabled.12",
            "xlsx_xml",
            {"workbook_structure", "tables", "charts", "formulas", "named_ranges", "conditional_formats", "links", "comments"},
        ),
    ]

    for filename, content, content_type, expected_method, expected_capabilities in cases:
        result = await ingest.graphrag_ingest_pipeline_preview(
            _mock_upload(filename, content, content_type),
            domain="office",
            sample_limit=80,
        )

        assert result["summary"]["extraction_method"] == expected_method
        assert result["document_profile"]["recommended_payload"] == "structured_text"
        assert result["document_profile"]["readiness_score"] >= 80
        assert expected_capabilities.issubset(set(result["document_profile"]["detected_capabilities"]))
        assert result["summary"]["element_count"] > 10
        assert result["summary"]["relationship_candidate_count"] > 5
        assert result["structured_ingest"]["uses_structured_text"] is True


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_legacy_office_ole_stream_text(monkeypatch):
    ingest = load_graphrag_route_module("ingest")

    class FakeStream:
        def __init__(self, payload: bytes):
            self.payload = payload

        def read(self, *_args):
            return self.payload

        def close(self):
            return None

    class FakeOleFile:
        def __init__(self, _fileobj):
            self.streams = {
                ("WordDocument",): (
                    b"Northwind Legacy Launch Plan\x00"
                    b"Market Expansion depends on Partner Enablement\x00"
                    b"Q3 Revenue Bridge"
                ),
                ("\x05SummaryInformation",): "Alice Example Legacy Strategy".encode("utf-16le"),
            }

        def listdir(self, streams=True, storages=False):
            assert streams is True
            assert storages is False
            return [list(path) for path in self.streams]

        def openstream(self, path):
            return FakeStream(self.streams[tuple(path)])

        def close(self):
            return None

    fake_olefile = ModuleType("olefile")
    fake_olefile.OleFileIO = FakeOleFile
    monkeypatch.setitem(sys.modules, "olefile", fake_olefile)

    result = await ingest.graphrag_ingest_pipeline_preview(
        _mock_upload("legacy-plan.doc", b"\xd0\xcf\x11\xe0legacy", "application/msword"),
        domain="office",
        sample_limit=80,
    )

    profile = result["document_profile"]
    elements = result["stages"][1]["sample_outputs"]["elements"]

    assert result["summary"]["extraction_method"] == "legacy_office_ole"
    assert "legacy_office" in profile["detected_capabilities"]
    assert profile["recommended_payload"] == "structured_text"
    assert profile["evidence_profile"]["quick_counts"]["legacy_office_streams"] == 2
    assert profile["evidence_profile"]["quick_counts"]["legacy_office_text_blocks"] >= 1
    assert any("best-effort OLE stream text" in warning for warning in profile["quality_warnings"])
    assert any(
        element["type"] == "legacy_office_stream"
        and element["metadata"].get("legacy_office_stream_path") == "WordDocument"
        for element in elements
    )
    assert any(
        element["type"] == "legacy_office_text"
        and "Northwind Legacy Launch Plan" in element["text"]
        and element["metadata"].get("legacy_office_extraction_method") == "ole_stream_printable_strings"
        for element in elements
    )
    assert any(card["id"] == "legacy_office" for card in result["interaction_handoff"]["capability_cards"])
    assert "legacy Office stream" in result["interaction_handoff"]["citation_scopes"]
    assert result["structured_ingest"]["uses_structured_text"] is True


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_xlsx_formulas_comments_links_images_and_chart_relationships():
    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload(
        "scorecard.xlsx",
        _minimal_xlsx_bytes(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="sheets", sample_limit=120)

    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["document_metadata"] >= 5
    assert by_type["sheet_metadata"] >= 2
    assert by_type["heading"] >= 1
    assert by_type["named_range"] >= 2
    assert by_type["table_definition"] >= 1
    assert by_type["pivot_table"] >= 1
    assert by_type["table_header"] >= 1
    assert by_type["table_row"] >= 1
    assert by_type["form_field"] >= 1
    assert by_type["conditional_format"] >= 2
    assert by_type["formula"] >= 3
    assert by_type["comment"] >= 2
    assert by_type["link"] >= 1
    assert by_type["image"] >= 1
    assert by_type["chart_series"] >= 1
    assert by_type["chart_point"] >= 2
    profile_stage = next(stage for stage in result["stages"] if stage["name"] == "document_profile")
    profile = profile_stage["sample_outputs"]
    assert profile["complexity"] == "complex"
    assert profile["readiness_score"] >= 90
    assert set(["metadata", "workbook_structure", "sections", "tables", "pivot_tables", "formulas", "named_ranges", "conditional_formats", "charts", "images", "links", "forms", "comments"]).issubset(
        set(profile["detected_capabilities"])
    )
    assert profile["evidence_profile"]["quick_counts"]["sheets"] >= 2
    assert profile["evidence_profile"]["quick_counts"]["hidden_sheets"] >= 1
    assert profile["evidence_profile"]["quick_counts"]["pivot_tables"] >= 1
    assert profile["recommended_payload"] == "structured_text"
    assert result["document_profile"]["graph_candidate_count"] >= 10
    assert result["document_profile"]["relationship_type_counts"]["contains_table"] >= 1
    assert result["document_profile"]["relationship_type_counts"]["has_hidden_sheet"] >= 1
    assert result["document_profile"]["relationship_type_counts"]["has_sheet_visibility"] >= 1
    assert result["document_profile"]["relationship_type_counts"]["contains_pivot_table"] >= 1
    assert result["document_profile"]["relationship_type_counts"]["uses_source"] >= 1
    assert result["document_profile"]["relationship_type_counts"]["aggregates"] >= 1
    assert result["document_profile"]["relationship_type_counts"]["contains_chart"] >= 1
    assert result["document_profile"]["relationship_type_counts"]["contains_link"] >= 1
    assert result["document_profile"]["relationship_type_counts"]["contains_image"] >= 1
    graph_candidates_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_candidates")
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["workbook_structure"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["conditional_format"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["spreadsheet_layout"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["formula"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["pivot_table"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["comment"] >= 1
    relationship_highlights = result["samples"]["relationship_highlights"]
    assert any(
        relationship["sample_category"] == "conditional_format"
        and "Scorecard!C2:C20" in {relationship["source"], relationship["target"]}
        for relationship in relationship_highlights
    )
    assert any(
        relationship["sample_category"] == "spreadsheet_layout"
        and "Deployment Gate" in {relationship["source"], relationship["target"]}
        for relationship in relationship_highlights
    )
    assert any(
        relationship["sample_category"] == "workbook_structure"
        and relationship["relation"] == "has_hidden_sheet"
        and relationship["target"] == "RawData"
        for relationship in relationship_highlights
    )
    assert any(
        relationship["sample_category"] == "pivot_table"
        and "QualityPivot" in {relationship["source"], relationship["target"]}
        for relationship in relationship_highlights
    )
    assert any(
        card["category"] == "formula"
        and "Scorecard!D2" in " ".join([card.get("title", ""), card.get("evidence_preview", "")])
        for card in graph_candidates_stage["sample_outputs"]["relationship_evidence_cards"]
    )
    assert any(
        card["category"] == "conditional_format"
        and "Scorecard!C2:C20" in " ".join([card.get("title", ""), card.get("evidence_preview", "")])
        for card in graph_candidates_stage["sample_outputs"]["relationship_evidence_cards"]
    )
    assert any(
        card["category"] == "workbook_structure"
        and "RawData" in " ".join([card.get("title", ""), card.get("evidence_preview", "")])
        for card in graph_candidates_stage["sample_outputs"]["relationship_evidence_cards"]
    )
    assert any(
        card["category"] == "pivot_table"
        and "QualityPivot" in " ".join([card.get("title", ""), card.get("evidence_preview", "")])
        for card in graph_candidates_stage["sample_outputs"]["relationship_evidence_cards"]
    )
    assert any(
        card["category"] == "comment"
        and "Scorecard" in " ".join([card.get("title", ""), card.get("evidence_preview", "")])
        for card in graph_candidates_stage["sample_outputs"]["relationship_evidence_cards"]
    )
    assert any(
        element["type"] == "document_metadata"
        and element["metadata"].get("metadata_field") == "title"
        and element["metadata"].get("metadata_value") == "GraphRAG Quality Scorecard"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "sheet_metadata"
        and element["metadata"].get("sheet_name") == "RawData"
        and element["metadata"].get("sheet_visibility") == "hidden"
        and element["metadata"].get("sheet_part") == "xl/worksheets/sheet2.xml"
        and element["metadata"].get("sheet_relationship_id") == "rId2"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["metadata"].get("formulas", [{}])[0].get("formula") == "SUM(10,8)"
        for element in result["samples"]["elements"]
        if element["type"] == "table_row"
    )
    assert any(
        element["type"] == "formula"
        and element["metadata"].get("formula_cell") == "Scorecard!D2"
        and element["metadata"].get("formula_expression") == "C2+2"
        and element["metadata"].get("formula_result") == "20"
        and element["metadata"].get("formula_references") == ["Scorecard!C2"]
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "formula"
        and element["metadata"].get("formula_cell") == "Scorecard!E2"
        and element["metadata"].get("formula_expression") == "COUNTA(QualityOwners)"
        and element["metadata"].get("formula_result") == "1"
        and element["metadata"].get("formula_references") == ["QualityOwners"]
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "heading"
        and element["text"] == "Deployment Gate"
        and element["metadata"].get("heading_kind") == "spreadsheet_merged_cell"
        and element["metadata"].get("merge_range") == "D1:F1"
        and element["metadata"].get("merge_anchor_cell") == "D1"
        and element["metadata"].get("merge_cell_count") == 3
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "form_field"
        and element["metadata"].get("form_kind") == "spreadsheet_data_validation"
        and element["metadata"].get("validation_range") == "B2:B20"
        and element["metadata"].get("validation_type") == "list"
        and element["metadata"].get("label") == "Owner"
        and element["metadata"].get("value") == "Ada Lovelace, Grace Hopper, Katherine Johnson"
        and element["metadata"].get("validation_prompt") == "Choose the accountable owner"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "conditional_format"
        and element["metadata"].get("conditional_range") == "C2:C20"
        and element["metadata"].get("conditional_range_label") == "Scorecard!C2:C20"
        and element["metadata"].get("conditional_type") == "cellIs"
        and element["metadata"].get("conditional_operator") == "greaterThanOrEqual"
        and element["metadata"].get("conditional_formula1") == "18"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "conditional_format"
        and element["metadata"].get("conditional_range") == "D2:D20"
        and element["metadata"].get("conditional_type") == "expression"
        and element["metadata"].get("conditional_formula1") == "D2>=C2"
        and element["metadata"].get("conditional_formula_references") == ["Scorecard!D2", "Scorecard!C2"]
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "table_definition"
        and element["metadata"].get("table_display_name") == "OwnerScorecard"
        and element["metadata"].get("table_range") == "A1:C2"
        and element["metadata"].get("table_part") == "xl/tables/table1.xml"
        and element["metadata"].get("table_columns") == ["Metric", "Owner", "Score"]
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "pivot_table"
        and element["metadata"].get("pivot_table_name") == "QualityPivot"
        and element["metadata"].get("pivot_table_range") == "H1:J6"
        and element["metadata"].get("pivot_table_source_name") == "OwnerScorecard"
        and element["metadata"].get("pivot_table_source_ref") == "A1:C2"
        and element["metadata"].get("pivot_table_source_sheet") == "Scorecard"
        and element["metadata"].get("pivot_table_cache_fields") == ["Metric", "Owner", "Score"]
        and element["metadata"].get("pivot_table_row_fields") == ["Metric"]
        and element["metadata"].get("pivot_table_column_fields") == ["Owner"]
        and element["metadata"].get("pivot_table_data_fields") == [{"name": "Average Score", "source_field": "Score", "subtotal": "average"}]
        and element["metadata"].get("pivot_table_cache_part") == "xl/pivotCache/pivotCacheDefinition1.xml"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "named_range"
        and element["metadata"].get("named_range_name") == "QualityOwners"
        and element["metadata"].get("named_range_ref") == "Scorecard!$B$2:$B$20"
        and element["metadata"].get("named_range_scope") == "Scorecard"
        and element["metadata"].get("sheet_name") == "Scorecard"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "comment"
        and element["metadata"].get("sheet_name") == "Scorecard"
        and element["metadata"].get("cell") == "B2"
        and element["metadata"].get("author") == "Grace Hopper"
        and element["metadata"].get("comment_anchor_sheet") == "Scorecard"
        and element["metadata"].get("comment_anchor_cell") == "B2"
        and element["metadata"].get("comment_anchor_header") == "Owner"
        and element["metadata"].get("comment_anchor_value") == "Ada Lovelace"
        and element["metadata"].get("comment_anchor_text") == "Owner; Ada Lovelace"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "comment"
        and element["metadata"].get("sheet_name") == "Scorecard"
        and element["metadata"].get("cell") == "C2"
        and element["metadata"].get("author") == "Katherine Johnson"
        and element["metadata"].get("comment_kind") == "threaded_comment"
        and element["metadata"].get("threaded_comment_id") == "thread-comment-1"
        and element["metadata"].get("person_id") == "person-thread-1"
        and element["metadata"].get("comment_anchor_sheet") == "Scorecard"
        and element["metadata"].get("comment_anchor_cell") == "C2"
        and element["metadata"].get("comment_anchor_header") == "Score"
        and element["metadata"].get("comment_anchor_value") == "18 (formula: SUM(10,8))"
        and element["metadata"].get("comment_anchor_text") == "Score; 18 (formula: SUM(10,8))"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "link"
        and element["metadata"].get("sheet_name") == "Scorecard"
        and element["metadata"].get("cell") == "A2"
        and element["metadata"].get("uri") == "https://example.com/scorecard"
        and element["metadata"].get("link_kind") == "spreadsheet_hyperlink"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "image"
        and element["metadata"].get("sheet_name") == "Scorecard"
        and element["metadata"].get("image_part") == "xl/media/scorecard.png"
        and element["metadata"].get("image_mime") == "image/png"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "chart_series"
        and element["location"].startswith("Scorecard chart")
        and element["metadata"].get("sheet_name") == "Scorecard"
        and element["metadata"].get("chart_part") == "xl/charts/chart1.xml"
        for element in result["samples"]["elements"]
    )

    relationships = result["samples"]["relationships"]
    graph_edges = result["local_graph"]["edges"]
    assert any(
        relationship["source"] == "scorecard.xlsx"
        and relationship["relation"] == "created_by"
        and relationship["target"] == "Ada Lovelace"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "scorecard.xlsx"
        and relationship["relation"] == "created_with"
        and relationship["target"] == "Microsoft Excel"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "GraphRAG"
        and relationship["relation"] == "owner"
        and relationship["target"] == "Ada Lovelace"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "scorecard.xlsx"
        and relationship["relation"] == "has_formula"
        and relationship["target"] == "Scorecard!D2"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!D2"
        and relationship["relation"] == "depends_on"
        and relationship["target"] == "Scorecard!C2"
        and relationship["metadata"].get("formula_expression") == "C2+2"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!D2"
        and relationship["relation"] == "calculates"
        and relationship["target"] == "20"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!E2"
        and relationship["relation"] == "depends_on"
        and relationship["target"] == "QualityOwners"
        and relationship["metadata"].get("formula_expression") == "COUNTA(QualityOwners)"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard"
        and relationship["relation"] == "has_conditional_format"
        and relationship["target"] == "Scorecard!C2:C20 cellIs greaterThanOrEqual 18"
        and relationship["metadata"].get("conditional_range") == "C2:C20"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!C2:C20"
        and relationship["relation"] == "evaluated_by"
        and relationship["target"] == "Scorecard!C2:C20 cellIs greaterThanOrEqual 18"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!C2:C20 cellIs greaterThanOrEqual 18"
        and relationship["relation"] == "uses_condition"
        and relationship["target"] == "cellIs greaterThanOrEqual 18"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!D2:D20 expression D2>=C2"
        and relationship["relation"] == "depends_on"
        and relationship["target"] == "Scorecard!C2"
        and relationship["metadata"].get("conditional_formula1") == "D2>=C2"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard"
        and relationship["relation"] == "has_merged_label"
        and relationship["target"] == "Deployment Gate"
        and relationship["metadata"].get("merge_range") == "D1:F1"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Deployment Gate"
        and relationship["relation"] == "covers_range"
        and relationship["target"] == "Scorecard!D1:F1"
        and relationship["metadata"].get("merge_range") == "D1:F1"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!D1:F1"
        and relationship["relation"] == "labeled_by"
        and relationship["target"] == "Deployment Gate"
        and relationship["metadata"].get("merge_range") == "D1:F1"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "scorecard.xlsx"
        and relationship["relation"] == "has_field"
        and relationship["target"] == "Owner"
        and relationship["metadata"].get("validation_range") == "B2:B20"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Owner"
        and relationship["relation"] == "has_value"
        and relationship["target"] == "Ada Lovelace, Grace Hopper, Katherine Johnson"
        and relationship["metadata"].get("form_kind") == "spreadsheet_data_validation"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard"
        and relationship["relation"] == "contains_table"
        and relationship["target"] == "OwnerScorecard"
        and relationship["metadata"].get("table_range") == "A1:C2"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "OwnerScorecard"
        and relationship["relation"] == "covers_range"
        and relationship["target"] == "A1:C2"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "OwnerScorecard"
        and relationship["relation"] == "has_column"
        and relationship["target"] == "Owner"
        and relationship["metadata"].get("table_part") == "xl/tables/table1.xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard"
        and relationship["relation"] == "contains_pivot_table"
        and relationship["target"] == "QualityPivot"
        and relationship["metadata"].get("pivot_table_range") == "H1:J6"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "QualityPivot"
        and relationship["relation"] == "uses_source"
        and relationship["target"] == "OwnerScorecard"
        and relationship["metadata"].get("pivot_table_source_ref") == "A1:C2"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "QualityPivot"
        and relationship["relation"] == "groups_by"
        and relationship["target"] == "Metric"
        and relationship["metadata"].get("pivot_axis") == "row"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "QualityPivot"
        and relationship["relation"] == "columns_by"
        and relationship["target"] == "Owner"
        and relationship["metadata"].get("pivot_axis") == "column"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "QualityPivot"
        and relationship["relation"] == "aggregates"
        and relationship["target"] == "Average Score"
        and relationship["metadata"].get("pivot_subtotal") == "average"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "scorecard.xlsx"
        and relationship["relation"] == "has_named_range"
        and relationship["target"] == "QualityOwners"
        and relationship["metadata"].get("named_range_ref") == "Scorecard!$B$2:$B$20"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "QualityOwners"
        and relationship["relation"] == "covers_range"
        and relationship["target"] == "Scorecard!$B$2:$B$20"
        for relationship in relationships
    )
    assert any(
        edge["source"] == "Quality"
        and edge["label"] == "graphrag_value"
        and edge["target"] == "94"
        and edge["metadata"].get("sheet_name") == "Scorecard"
        and edge["metadata"].get("chart_part") == "xl/charts/chart1.xml"
        for edge in graph_edges
    )
    assert any(
        relationship["source"] == "scorecard.xlsx"
        and relationship["relation"] == "links_to"
        and relationship["target"] == "https://example.com/scorecard"
        and relationship["metadata"].get("cell") == "A2"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "scorecard.xlsx"
        and relationship["relation"] == "has_comment"
        and "Owner requires review" in relationship["target"]
        and relationship["metadata"].get("comment_anchor_cell") == "B2"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard"
        and relationship["relation"] == "comments_on"
        and relationship["target"] == "Owner; Ada Lovelace"
        and relationship["metadata"].get("comment_anchor_cell") == "B2"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!B2"
        and relationship["relation"] == "has_comment"
        and "Owner requires review" in relationship["target"]
        and relationship["metadata"].get("author") == "Grace Hopper"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!B2"
        and relationship["relation"] == "has_header"
        and relationship["target"] == "Owner"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!B2"
        and relationship["relation"] == "has_value"
        and relationship["target"] == "Ada Lovelace"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Grace Hopper"
        and relationship["relation"] == "commented_on"
        and relationship["target"] == "Owner; Ada Lovelace"
        and relationship["metadata"].get("comment_anchor_cell") == "B2"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!C2"
        and relationship["relation"] == "has_comment"
        and "Score formula needs audit trail" in relationship["target"]
        and relationship["metadata"].get("comment_kind") == "threaded_comment"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!C2"
        and relationship["relation"] == "has_header"
        and relationship["target"] == "Score"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Scorecard!C2"
        and relationship["relation"] == "has_value"
        and relationship["target"] == "18 (formula: SUM(10,8))"
        for relationship in relationships
    )
    assert any(
        edge["source"] == "Katherine Johnson"
        and edge["label"] == "commented_on"
        and edge["target"] == "Score; 18 (formula: SUM(10,8))"
        and edge["metadata"].get("person_id") == "person-thread-1"
        for edge in graph_edges
    )
    assert any(
        edge["source"] == "scorecard.xlsx"
        and edge["label"] == "contains_image"
        and edge["target"] == "Scorecard Snapshot"
        and edge["metadata"].get("image_part") == "xl/media/scorecard.png"
        for edge in graph_edges
    )
    assert any(
        relationship["source"] == "scorecard.xlsx"
        and relationship["relation"] == "has_sheet"
        and relationship["target"] == "Scorecard"
        for relationship in relationships
    )
    assert any(
        edge["source"] == "scorecard.xlsx"
        and edge["label"] == "has_hidden_sheet"
        and edge["target"] == "RawData"
        and edge["metadata"].get("sheet_visibility") == "hidden"
        for edge in graph_edges
    )
    assert any(
        edge["source"] == "RawData"
        and edge["label"] == "has_sheet_visibility"
        and edge["target"] == "hidden"
        and edge["metadata"].get("sheet_part") == "xl/worksheets/sheet2.xml"
        for edge in graph_edges
    )
    assert any(
        edge["source"] == "RawData"
        and edge["label"] == "uses_sheet_part"
        and edge["target"] == "xl/worksheets/sheet2.xml"
        and edge["metadata"].get("sheet_relationship_id") == "rId2"
        for edge in graph_edges
    )
    assert any(
        edge["source"] == "Scorecard"
        and edge["label"] == "contains_chart"
        and edge["target"] == "xl/charts/chart1.xml"
        for edge in graph_edges
    )
    assert any(
        relationship["source"] == "Scorecard"
        and relationship["relation"] == "contains_link"
        and relationship["target"] == "https://example.com/scorecard"
        for relationship in relationships
    )
    assert any(
        edge["source"] == "Scorecard"
        and edge["label"] == "contains_image"
        and edge["target"] == "Scorecard Snapshot"
        and edge["metadata"].get("image_part") == "xl/media/scorecard.png"
        for edge in graph_edges
    )
    embedding_stage = next(stage for stage in result["stages"] if stage["name"] == "embedding_plan")
    embedding_text = "\n".join(item["text"] for item in embedding_stage["sample_outputs"]["embedding_inputs"])
    embedding_provenance = [
        provenance
        for item in embedding_stage["sample_outputs"]["embedding_inputs"]
        for provenance in item["metadata"].get("provenance", [])
    ]
    assert "sheet_name=Scorecard" in embedding_text
    assert "heading_kind=spreadsheet_merged_cell" in embedding_text
    assert "merge_range=D1:F1" in embedding_text
    assert "form_kind=spreadsheet_data_validation" in embedding_text
    assert "validation_range=B2:B20" in embedding_text
    assert "validation_formula1=\"Ada Lovelace,Grace Hopper,Katherine Johnson\"" in embedding_text
    assert "conditional_range=C2:C20" in embedding_text
    assert "conditional_type=cellIs" in embedding_text
    assert "conditional_formula1=18" in embedding_text
    assert "conditional_formula_references=[\"Scorecard!D2\", \"Scorecard!C2\"]" in embedding_text
    assert "formula_cell=Scorecard!D2" in embedding_text
    assert "formula_expression=C2+2" in embedding_text
    assert "formula_references=[\"Scorecard!C2\"]" in embedding_text
    assert "formula_cell=Scorecard!E2" in embedding_text
    assert "formula_references=[\"QualityOwners\"]" in embedding_text
    assert "table_display_name=OwnerScorecard" in embedding_text
    assert "table_range=A1:C2" in embedding_text
    assert "table_columns=[\"Metric\", \"Owner\", \"Score\"]" in embedding_text
    assert "pivot_table_name=QualityPivot" in embedding_text
    assert "pivot_table_source_name=OwnerScorecard" in embedding_text
    assert "pivot_table_row_fields=[\"Metric\"]" in embedding_text
    assert "pivot_table_data_fields=[{\"name\": \"Average Score\", \"source_field\": \"Score\", \"subtotal\": \"average\"}]" in embedding_text
    assert "sheet_name=RawData" in embedding_text
    assert "sheet_visibility=hidden" in embedding_text
    assert "Hidden source notes" in embedding_text
    assert "named_range_name=QualityOwners" in embedding_text
    assert "named_range_ref=Scorecard!$B$2:$B$20" in embedding_text
    assert "comment_anchor_text=Owner; Ada Lovelace" in embedding_text
    assert "comment_anchor_cell=B2" in embedding_text
    assert "comment_kind=threaded_comment" in embedding_text
    assert "comment_anchor_text=Score; 18 (formula: SUM(10,8))" in embedding_text
    assert "person_id=person-thread-1" in embedding_text
    assert "metadata_field=title" in embedding_text
    assert "metadata_value=GraphRAG Quality Scorecard" in embedding_text
    assert "cell=B2" in embedding_text
    assert "uri=https://example.com/scorecard" in embedding_text
    assert "image_part=xl/media/scorecard.png" in embedding_text
    assert "chart_part=xl/charts/chart1.xml" in embedding_text
    assert "SUM(10,8)" in embedding_text
    assert any("cell=A2" in provenance for provenance in embedding_provenance)
    assert any("uri=https://example.com/scorecard" in provenance for provenance in embedding_provenance)
    assert any("merge_anchor_cell=D1" in provenance for provenance in embedding_provenance)
    assert any("validation_range=B2:B20" in provenance for provenance in embedding_provenance)
    assert any("conditional_range_label=Scorecard!C2:C20" in provenance for provenance in embedding_provenance)
    assert any("conditional_formula1=D2>=C2" in provenance for provenance in embedding_provenance)
    assert any("formula_reference_count=1" in provenance for provenance in embedding_provenance)
    assert any("formula_result=20" in provenance for provenance in embedding_provenance)
    assert any("table_part=xl/tables/table1.xml" in provenance for provenance in embedding_provenance)
    assert any("sheet_part=xl/worksheets/sheet2.xml" in provenance for provenance in embedding_provenance)
    assert any("sheet_visibility=hidden" in provenance for provenance in embedding_provenance)
    assert any("pivot_table_cache_part=xl/pivotCache/pivotCacheDefinition1.xml" in provenance for provenance in embedding_provenance)
    assert any("pivot_table_source_ref=A1:C2" in provenance for provenance in embedding_provenance)
    assert any("named_range_scope=Scorecard" in provenance for provenance in embedding_provenance)
    assert any("comment_anchor_header=Owner" in provenance for provenance in embedding_provenance)
    assert any("comment_anchor_value=Ada Lovelace" in provenance for provenance in embedding_provenance)
    assert any("comment_anchor_header=Score" in provenance for provenance in embedding_provenance)
    assert any("comment_anchor_value=18 (formula: SUM(10,8))" in provenance for provenance in embedding_provenance)
    assert any("image_part=xl/media/scorecard.png" in provenance for provenance in embedding_provenance)
    assert any("chart_part=xl/charts/chart1.xml" in provenance for provenance in embedding_provenance)


@pytest.mark.asyncio
async def test_pipeline_preview_extracts_docx_revisions_form_fields_styled_text_and_table_relationships():
    ingest = load_graphrag_route_module("ingest")
    upload = _mock_upload(
        "implementation.docx",
        _minimal_docx_bytes(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    result = await ingest.graphrag_ingest_pipeline_preview(upload, domain="docs", sample_limit=80)

    structure_stage = next(stage for stage in result["stages"] if stage["name"] == "structure_extraction")
    by_type = structure_stage["counts"]["by_type"]
    assert by_type["document_metadata"] >= 7
    assert by_type["form_field"] >= 1
    assert by_type["heading"] >= 1
    assert by_type["image"] >= 1
    assert by_type["link"] >= 1
    assert by_type["comment"] >= 1
    assert by_type["revision"] >= 2
    assert by_type["footnote"] >= 1
    assert by_type["endnote"] >= 1
    assert by_type["header"] >= 1
    assert by_type["footer"] >= 1
    assert by_type["table_header"] >= 1
    assert by_type["table_row"] >= 1
    assert by_type["table_cell"] >= 8
    assert by_type["text_box"] >= 2
    assert by_type["chart_series"] >= 1
    assert by_type["chart_point"] >= 2
    assert by_type["chart_axis"] >= 2
    assert by_type["chart_legend"] >= 1
    assert by_type["diagram_node"] >= 3
    assert by_type["diagram_relationship"] >= 2
    assert "shapes" in result["document_profile"]["detected_capabilities"]
    assert "charts" in result["document_profile"]["detected_capabilities"]
    assert "diagrams" in result["document_profile"]["detected_capabilities"]
    assert result["document_profile"]["evidence_profile"]["quick_counts"]["shapes"] >= 2
    assert result["document_profile"]["evidence_profile"]["quick_counts"]["charts"] >= 1
    assert result["document_profile"]["evidence_profile"]["quick_counts"]["diagrams"] >= 5
    assert result["document_profile"]["evidence_profile"]["graph_evidence"]["diagram_relationship_count"] >= 1
    graph_candidates_stage = next(stage for stage in result["stages"] if stage["name"] == "graph_candidates")
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["comment"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["note"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["chart_fact"] >= 1
    assert graph_candidates_stage["sample_outputs"]["relationship_sample_categories"]["diagram"] >= 1
    assert any(
        card["category"] == "comment"
        and "Architecture note" in " ".join([card.get("title", ""), card.get("evidence_preview", "")])
        for card in graph_candidates_stage["sample_outputs"]["relationship_evidence_cards"]
    )
    assert any(
        card["category"] == "note"
        and "Deployment evidence" in " ".join([card.get("title", ""), card.get("evidence_preview", "")])
        for card in graph_candidates_stage["sample_outputs"]["relationship_evidence_cards"]
    )
    assert any(
        card["category"] == "diagram"
        and "word/diagrams/data1.xml" in card.get("provenance", "")
        for card in graph_candidates_stage["sample_outputs"]["relationship_evidence_cards"]
    )
    assert any(
        element["type"] == "document_metadata"
        and element["metadata"].get("metadata_field") == "title"
        and element["metadata"].get("metadata_part") == "docProps/core.xml"
        and element["metadata"].get("metadata_value") == "GraphRAG Implementation Plan"
        for element in result["samples"]["elements"]
    )
    assert any(
        "Implementation Plan" in element["metadata"].get("emphasis_terms", [])
        for element in result["samples"]["elements"]
        if element["type"] == "heading"
    )
    assert any(
        element["type"] == "image"
        and element["metadata"].get("image_part") == "word/media/architecture.png"
        and element["metadata"].get("image_width") == 1
        and element["metadata"].get("image_height") == 1
        and element["metadata"].get("image_mime") == "image/png"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "chart_series"
        and element["metadata"].get("chart_part") == "word/charts/chart1.xml"
        and element["metadata"].get("chart_title") == "Launch Metrics"
        and element["metadata"].get("series_name") == "Readiness"
        and element["metadata"].get("relationship_id") == "rIdChart1"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "chart_point"
        and element["metadata"].get("chart_part") == "word/charts/chart1.xml"
        and element["metadata"].get("category") == "Q2"
        and element["metadata"].get("value") == "88"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "diagram_node"
        and element["text"] == "Release Gate"
        and element["metadata"].get("diagram_part") == "word/diagrams/data1.xml"
        and element["metadata"].get("diagram_node_id") == "release"
        and element["metadata"].get("relationship_id") == "rIdDiagram1"
        and element["metadata"].get("source_part") == "word/document.xml"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "diagram_relationship"
        and element["text"] == "Release Gate --[contains]--> Dependency Review"
        and element["metadata"].get("diagram_part") == "word/diagrams/data1.xml"
        and element["metadata"].get("source_diagram_node_id") == "release"
        and element["metadata"].get("target_diagram_node_id") == "dependency"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "link"
        and element["metadata"].get("uri") == "https://example.com/graphrag-design"
        and element["metadata"].get("relationship_id") == "rIdLink1"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "comment"
        and element["metadata"].get("author") == "Ada Lovelace"
        and element["metadata"].get("comment_id") == "0"
        and element["metadata"].get("comment_anchor_text") == "Architecture note"
        and element["metadata"].get("comment_anchor_section") == "Implementation Plan"
        and element["metadata"].get("comment_anchor_paragraph")
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "revision"
        and "Inserted launch checklist" in element["text"]
        and element["metadata"].get("revision_type") == "inserted"
        and element["metadata"].get("revision_author") == "Grace Hopper"
        and element["metadata"].get("revision_id") == "11"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "revision"
        and "Deprecated manual review" in element["text"]
        and element["metadata"].get("revision_type") == "deleted"
        and element["metadata"].get("revision_author") == "Alan Turing"
        and element["metadata"].get("revision_id") == "12"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "footnote"
        and element["metadata"].get("note_id") == "2"
        and element["metadata"].get("note_anchor_text") == "Deployment evidence"
        and element["metadata"].get("note_anchor_section") == "Implementation Plan"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "endnote"
        and element["metadata"].get("note_id") == "3"
        and element["metadata"].get("note_anchor_text") == "Deployment evidence"
        and element["metadata"].get("note_anchor_section") == "Implementation Plan"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "header"
        and element["metadata"].get("part_name") == "word/header1.xml"
        for element in result["samples"]["elements"]
    )
    approved_field = next(
        element
        for element in result["samples"]["elements"]
        if element["type"] == "form_field" and element["metadata"].get("label") == "Approved"
    )
    assert approved_field["metadata"]["form_kind"] == "docx_content_control"
    assert approved_field["metadata"]["field_type"] == "checkbox"
    assert approved_field["metadata"]["value"] == "checked"
    assert approved_field["metadata"]["checked"] is True
    assert approved_field["metadata"]["form_field_checked"] is True
    assert approved_field["metadata"]["form_field_tag"] == "approved"
    assert approved_field["metadata"]["control_id"] == "101"
    launch_owner_field = next(
        element
        for element in result["samples"]["elements"]
        if element["type"] == "form_field" and element["metadata"].get("label") == "Launch Owner"
    )
    assert launch_owner_field["metadata"]["field_type"] == "dropdown"
    assert launch_owner_field["metadata"]["value"] == "Operations"
    assert launch_owner_field["metadata"]["form_field_selected_option"] == "Operations"
    assert launch_owner_field["metadata"]["form_field_options"] == ["Operations", "Security", "Support"]
    assert launch_owner_field["metadata"]["form_field_option_values"] == ["ops", "sec", "support"]
    assert launch_owner_field["metadata"]["control_id"] == "102"
    review_date_field = next(
        element
        for element in result["samples"]["elements"]
        if element["type"] == "form_field" and element["metadata"].get("label") == "Review Date"
    )
    assert review_date_field["metadata"]["field_type"] == "date"
    assert review_date_field["metadata"]["value"] == "April 15, 2026"
    assert review_date_field["metadata"]["form_field_date_format"] == "MMMM d, yyyy"
    assert review_date_field["metadata"]["form_field_date_full_value"] == "2026-04-15T00:00:00Z"
    assert review_date_field["metadata"]["form_field_calendar"] == "gregorian"
    assert any(
        element["type"] == "text_box"
        and element["text"] == "Risk callout: launch blockers require graph trace"
        and element["metadata"].get("text_box_kind") == "drawing_text_box"
        and element["metadata"].get("text_box_part") == "word/document.xml"
        and element["metadata"].get("shape_title") == "Risk Callout"
        and element["metadata"].get("shape_description") == "Text box captures launch blockers"
        and element["metadata"].get("shape_id") == "8"
        and element["metadata"].get("style_summary") == "bold"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "text_box"
        and element["text"] == "Legacy decision note: preserve source links"
        and element["metadata"].get("text_box_kind") == "vml_text_box"
        and element["metadata"].get("shape_title") == "Legacy Decision Box"
        and element["metadata"].get("shape_description") == "Legacy text box preserves source links"
        and element["metadata"].get("shape_id") == "LegacyDecisionBox"
        for element in result["samples"]["elements"]
    )
    assert any(
        element["type"] == "table_cell"
        and element["text"] == "Governance checkpoint"
        and element["metadata"].get("section") == "Implementation Plan"
        and element["metadata"].get("table_cell_reference") == "R3C1"
        and element["metadata"].get("table_cell_grid_span") == 2
        and element["metadata"].get("style_summary") == "bold; italic"
        and "Governance checkpoint" in element["metadata"].get("emphasis_terms", [])
        for element in result["samples"]["elements"]
    )
    governance_cell = next(
        element
        for element in result["samples"]["elements"]
        if element["type"] == "table_cell" and element["text"] == "Governance checkpoint"
    )
    governance_cell_metadata = governance_cell["metadata"]
    assert governance_cell_metadata["table_cell_style_name"] == "GovernanceCell"
    assert governance_cell_metadata["table_cell_fill_color"] == "#FFF2CC"
    assert governance_cell_metadata["table_cell_shading_pattern"] == "pct20"
    assert governance_cell_metadata["table_cell_text_anchor"] == "center"
    assert governance_cell_metadata["table_cell_vertical_align"] == "center"
    assert governance_cell_metadata["table_cell_text_direction"] == "tbRl"
    assert governance_cell_metadata["table_cell_width_value"] == "4320"
    assert governance_cell_metadata["table_cell_width_type"] == "dxa"
    assert governance_cell_metadata["table_cell_width_inches"] == pytest.approx(3.0)
    assert governance_cell_metadata["table_cell_margin_left_inches"] == pytest.approx(0.1)
    assert governance_cell_metadata["table_cell_margin_right_inches"] == pytest.approx(0.2)
    assert governance_cell_metadata["table_cell_margin_top_inches"] == pytest.approx(0.05)
    assert governance_cell_metadata["table_cell_margin_bottom_inches"] == pytest.approx(0.05)
    assert governance_cell_metadata["table_cell_border_top_color"] == "#4472C4"
    assert governance_cell_metadata["table_cell_border_bottom_color"] == "#70AD47"
    assert governance_cell_metadata["table_cell_border_top_width_inches"] == pytest.approx(0.0139)
    assert governance_cell_metadata["table_cell_border_bottom_width_inches"] == pytest.approx(0.0208)
    assert "fill #FFF2CC" in governance_cell_metadata["table_cell_visual_summary"]
    assert "style GovernanceCell" in governance_cell_metadata["table_cell_visual_summary"]
    assert "width 3.000in dxa" in governance_cell_metadata["table_cell_visual_summary"]
    assert "borders top single #4472C4 0.0139in" in governance_cell_metadata["table_cell_visual_summary"]
    assert any(
        element["type"] == "table_cell"
        and element["text"] == "Risk Register"
        and element["metadata"].get("table_cell_reference") == "R4C1"
        and element["metadata"].get("table_cell_v_merge") == "restart"
        and "vertical" in element["metadata"].get("table_cell_merge_flags", [])
        for element in result["samples"]["elements"]
    )

    relationships = result["samples"]["relationships"]
    assert any(
        relationship["source"] == "implementation.docx"
        and relationship["relation"] == "created_by"
        and relationship["target"] == "Ada Lovelace"
        and relationship["metadata"].get("metadata_field") == "creator"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "implementation.docx"
        and relationship["relation"] == "has_keyword"
        and relationship["target"] == "Deployment"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "implementation.docx"
        and relationship["relation"] == "created_with"
        and relationship["target"] == "Microsoft Word"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "implementation.docx"
        and relationship["relation"] == "has_field"
        and relationship["target"] == "Approved"
        and relationship["metadata"].get("field_type") == "checkbox"
        and relationship["metadata"].get("form_field_checked") is True
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Launch Owner"
        and relationship["relation"] == "has_value"
        and relationship["target"] == "Operations"
        and relationship["metadata"].get("field_type") == "dropdown"
        and relationship["metadata"].get("form_field_options") == ["Operations", "Security", "Support"]
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Review Date"
        and relationship["relation"] == "has_value"
        and relationship["target"] == "April 15, 2026"
        and relationship["metadata"].get("field_type") == "date"
        and relationship["metadata"].get("form_field_date_format") == "MMMM d, yyyy"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Extractor"
        and relationship["relation"] == "owner"
        and relationship["target"] == "Llama Nexus"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "implementation.docx"
        and relationship["relation"] == "contains_image"
        and relationship["target"] == "Architecture Diagram"
        and relationship["metadata"].get("image_part") == "word/media/architecture.png"
        and relationship["metadata"].get("image_width") == 1
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "implementation.docx"
        and relationship["relation"] == "contains_chart"
        and relationship["target"] == "Launch Metrics"
        and relationship["metadata"].get("chart_part") == "word/charts/chart1.xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Launch Metrics"
        and relationship["relation"] == "contains_chart_series"
        and relationship["target"] == "Readiness"
        and relationship["metadata"].get("chart_part") == "word/charts/chart1.xml"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Readiness Q2"
        and relationship["relation"] == "has_value"
        and relationship["target"] == "88"
        and relationship["metadata"].get("chart_point_label") == "Readiness Q2"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "implementation.docx"
        and relationship["relation"] == "contains_diagram_node"
        and relationship["target"] == "Release Gate"
        and relationship["metadata"].get("diagram_part") == "word/diagrams/data1.xml"
        and relationship["metadata"].get("diagram_node_id") == "release"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Release Gate"
        and relationship["relation"] == "contains"
        and relationship["target"] == "Dependency Review"
        and relationship["metadata"].get("diagram_connection_type") == "parOf"
        and relationship["metadata"].get("source_diagram_node_id") == "release"
        and relationship["metadata"].get("target_diagram_node_id") == "dependency"
        for relationship in relationships
    )
    assert any(
        edge["source"] == "Release Gate"
        and edge["label"] == "contains"
        and edge["target"] == "Owner Handoff"
        and edge["metadata"].get("diagram_part") == "word/diagrams/data1.xml"
        for edge in result["local_graph"]["edges"]
    )
    assert any(
        relationship["source"] == "implementation.docx"
        and relationship["relation"] == "links_to"
        and relationship["target"] == "https://example.com/graphrag-design"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "implementation.docx"
        and relationship["relation"] == "has_comment"
        and "GraphRAG extraction boundary" in relationship["target"]
        and relationship["metadata"].get("author") == "Ada Lovelace"
        and relationship["metadata"].get("comment_anchor_text") == "Architecture note"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Implementation Plan"
        and relationship["relation"] == "has_comment"
        and "GraphRAG extraction boundary" in relationship["target"]
        and relationship["metadata"].get("comment_anchor_text") == "Architecture note"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Implementation Plan"
        and relationship["relation"] == "comments_on"
        and relationship["target"] == "Architecture note"
        and relationship["metadata"].get("comment_anchor_section") == "Implementation Plan"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Ada Lovelace"
        and relationship["relation"] == "commented_on"
        and relationship["target"] == "implementation.docx"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Ada Lovelace"
        and relationship["relation"] == "commented_on"
        and relationship["target"] == "Architecture note"
        and relationship["metadata"].get("comment_anchor_text") == "Architecture note"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "implementation.docx"
        and relationship["relation"] == "has_revision"
        and "Inserted launch checklist" in relationship["target"]
        and relationship["metadata"].get("revision_author") == "Grace Hopper"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Grace Hopper"
        and relationship["relation"] == "inserted_in"
        and relationship["target"] == "implementation.docx"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Alan Turing"
        and relationship["relation"] == "deleted_in"
        and relationship["target"] == "implementation.docx"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "implementation.docx"
        and relationship["relation"] == "has_footnote"
        and "deployment caveats" in relationship["target"]
        and relationship["metadata"].get("note_anchor_text") == "Deployment evidence"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Deployment evidence"
        and relationship["relation"] == "has_footnote"
        and "deployment caveats" in relationship["target"]
        and relationship["metadata"].get("note_anchor_section") == "Implementation Plan"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Deployment evidence"
        and relationship["relation"] == "has_endnote"
        and "governance owners" in relationship["target"]
        and relationship["metadata"].get("note_anchor_section") == "Implementation Plan"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Implementation Plan"
        and relationship["relation"] == "has_footnote"
        and "deployment caveats" in relationship["target"]
        and relationship["metadata"].get("note_anchor_text") == "Deployment evidence"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "implementation.docx"
        and relationship["relation"] == "has_section"
        and relationship["target"] == "Implementation Plan"
        and relationship["metadata"].get("section_level") == 1
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Implementation Plan"
        and relationship["relation"] == "contains_link"
        and relationship["target"] == "https://example.com/graphrag-design"
        and relationship["metadata"].get("section_label") == "Implementation Plan"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Implementation Plan"
        and relationship["relation"] == "contains_table"
        and "Extractor" in relationship["target"]
        and relationship["metadata"].get("section_label") == "Implementation Plan"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Implementation Plan"
        and relationship["relation"] == "contains_table"
        and relationship["target"] == "R3C1: Governance checkpoint"
        and relationship["metadata"].get("table_cell_reference") == "R3C1"
        and relationship["metadata"].get("table_cell_grid_span") == 2
        and relationship["metadata"].get("table_cell_fill_color") == "#FFF2CC"
        and relationship["metadata"].get("table_cell_width_inches") == pytest.approx(3.0)
        and relationship["metadata"].get("table_cell_border_top_color") == "#4472C4"
        and "style GovernanceCell" in relationship["metadata"].get("table_cell_visual_summary", "")
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Implementation Plan"
        and relationship["relation"] == "contains_revision"
        and "Deprecated manual review" in relationship["target"]
        and relationship["metadata"].get("section_label") == "Implementation Plan"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Implementation Plan"
        and relationship["relation"] == "contains_text"
        and relationship["target"] == "Risk callout: launch blockers require graph trace"
        and relationship["metadata"].get("text_box_kind") == "drawing_text_box"
        and relationship["metadata"].get("shape_title") == "Risk Callout"
        and relationship["metadata"].get("section_label") == "Implementation Plan"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Implementation Plan"
        and relationship["relation"] == "contains_text"
        and relationship["target"] == "Legacy decision note: preserve source links"
        and relationship["metadata"].get("text_box_kind") == "vml_text_box"
        and relationship["metadata"].get("shape_title") == "Legacy Decision Box"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Implementation Plan"
        and relationship["relation"] == "emphasizes"
        and relationship["target"] == "Risk callout: launch blockers require graph trace"
        and relationship["metadata"].get("styled_text") is True
        and relationship["metadata"].get("text_box_kind") == "drawing_text_box"
        for relationship in relationships
    )
    assert any(
        relationship["source"] == "Implementation Plan"
        and relationship["relation"] == "emphasizes"
        and relationship["target"] == "Governance checkpoint"
        and relationship["metadata"].get("table_cell_reference") == "R3C1"
        and relationship["metadata"].get("table_cell_grid_span") == 2
        for relationship in relationships
    )
    embedding_stage = next(stage for stage in result["stages"] if stage["name"] == "embedding_plan")
    embedding_text = "\n".join(item["text"] for item in embedding_stage["sample_outputs"]["embedding_inputs"])
    embedding_provenance = [
        provenance
        for item in embedding_stage["sample_outputs"]["embedding_inputs"]
        for provenance in item["metadata"].get("provenance", [])
    ]
    assert "metadata_field=title" in embedding_text
    assert "metadata_value=GraphRAG Implementation Plan" in embedding_text
    assert "chart_part=word/charts/chart1.xml" in embedding_text
    assert "chart_title=Launch Metrics" in embedding_text
    assert "chart_point_label=Readiness Q2" in embedding_text
    assert "diagram_part=word/diagrams/data1.xml" in embedding_text
    assert "diagram_node_id=release" in embedding_text
    assert "Release Gate --[contains]--> Dependency Review" in embedding_text
    assert any("part_name=word/header1.xml" in provenance for provenance in embedding_provenance)
    assert any("chart_part=word/charts/chart1.xml" in provenance for provenance in embedding_provenance)
    assert any("diagram_part=word/diagrams/data1.xml" in provenance for provenance in embedding_provenance)
    assert any("diagram_node_id=release" in provenance for provenance in embedding_provenance)
    assert any("comment_id=0" in provenance for provenance in embedding_provenance)
    assert any("comment_anchor_text=Architecture note" in provenance for provenance in embedding_provenance)
    assert any("comment_anchor_section=Implementation Plan" in provenance for provenance in embedding_provenance)
    assert any("author=Ada Lovelace" in provenance for provenance in embedding_provenance)
    assert any("note_id=2" in provenance for provenance in embedding_provenance)
    assert any("note_anchor_text=Deployment evidence" in provenance for provenance in embedding_provenance)
    assert any("note_anchor_section=Implementation Plan" in provenance for provenance in embedding_provenance)
    assert any("revision_author=Grace Hopper" in provenance for provenance in embedding_provenance)
    assert any("revision_type=deleted" in provenance for provenance in embedding_provenance)
    assert any("uri=https://example.com/graphrag-design" in provenance for provenance in embedding_provenance)
    assert "form_kind=docx_content_control" in embedding_text
    assert "field_type=dropdown" in embedding_text
    assert "form_field_options=[\"Operations\", \"Security\", \"Support\"]" in embedding_text
    assert "form_field_date_format=MMMM d, yyyy" in embedding_text
    assert "text_box_kind=drawing_text_box" in embedding_text
    assert "shape_title=Risk Callout" in embedding_text
    assert "table_cell_reference=R3C1" in embedding_text
    assert "table_cell_grid_span=2" in embedding_text
    assert "table_cell_fill_color=#FFF2CC" in embedding_text
    assert "table_cell_visual_summary=fill #FFF2CC" in embedding_text
    assert "width 3.000in dxa" in embedding_text
    assert "borders top single #4472C4 0.0139in" in embedding_text
    assert "Governance checkpoint" in embedding_text
    assert any("text_box_kind=vml_text_box" in provenance for provenance in embedding_provenance)
    assert any("shape_title=Legacy Decision Box" in provenance for provenance in embedding_provenance)
    assert any("table_cell_reference=R3C1" in provenance for provenance in embedding_provenance)
    assert any("table_cell_width_value=4320" in provenance for provenance in embedding_provenance)
    assert any("table_cell_v_merge=restart" in provenance for provenance in embedding_provenance)


@pytest.mark.asyncio
async def test_pipeline_upload_forwards_to_graphrag_and_tracks_upstream_stage():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.get = AsyncMock(return_value=_mock_response({"status": "healthy"}))
    mock_get_client.return_value.post = AsyncMock(return_value=_mock_response({"processed": 1, "total_chunks": 2}))
    upload = _mock_upload("notes.txt", b"GraphRAG connects entities to evidence.", "text/plain")

    with patch.object(ingest, "get_http_client", mock_get_client):
        result = await ingest.graphrag_ingest_pipeline_upload(
            upload,
            domain="docs",
            use_semantic_chunking=True,
            build_knowledge_graph=True,
            sample_limit=3,
        )

    assert result["status"] == "complete"
    assert result["completed_at"]
    assert result["operational_handoff"]["status"] == "remote_graph_submitted"
    assert result["operational_handoff"]["local_trace_ready"] is True
    assert result["operational_handoff"]["chat_handoff_ready"] is True
    assert result["operational_handoff"]["remote_ingest_ready"] is True
    assert result["operational_handoff"]["remote_ingest_submitted"] is True
    assert result["operational_handoff"]["primary_action"] == "ask_chat"
    assert result["operational_handoff"]["target_url"] == "http://graphrag.test/ingest-documents"
    assert {action["id"] for action in result["operational_handoff"]["actions"]} >= {"inspect_trace", "ask_chat", "remote_ingest"}
    assert result["upstream_preflight"]["status"] == "ready"
    assert result["upstream_preflight"]["ready"] is True
    assert result["upstream_preflight"]["service_status"] == "healthy"
    assert result["upstream_preflight"]["health_url"] == "http://graphrag.test/health"
    assert result["upstream_preflight"]["payload_kind"] == "original_file"
    assert result["upstream_preflight"]["sample_outputs"]["service_base_url"] == "http://graphrag.test"
    assert result["upstream_preflight"]["sample_outputs"]["target_url"] == "http://graphrag.test/ingest-documents"
    assert "http://graphrag.test/process-document" in result["upstream_preflight"]["sample_outputs"]["compatible_ingest_urls"]
    assert result["upstream_preflight"]["sample_outputs"]["compatible_health_urls"] == [
        "http://graphrag.test/health",
        "http://graphrag.test/api/health",
        "http://graphrag.test/api/v1/health",
    ]
    assert result["upstream_preflight"]["sample_outputs"]["attempted_health_urls"] == ["http://graphrag.test/health"]
    assert result["upstream_preflight"]["sample_outputs"]["payload_bytes"] == len(b"GraphRAG connects entities to evidence.")
    assert len(result["upstream_preflight"]["sample_outputs"]["payload_sha256"]) == 64
    assert result["upstream_preflight"]["sample_outputs"]["health_status_code"] == 200
    assert result["upstream_preflight"]["sample_outputs"]["services_status_path"] == "/api/v1/graphrag/services/status"
    assert result["upstream_preflight"]["sample_outputs"]["diagnostics_path"] == "/api/v1/graphrag/deployment/diagnostics"
    assert "health probe passed" in result["upstream_preflight"]["sample_outputs"]["next_step"]
    assert result["upstream_ingest"]["status"] == "complete"
    assert result["upstream_ingest"]["submitted"] is True
    assert result["upstream_ingest"]["target_url"] == "http://graphrag.test/ingest-documents"
    assert result["stages"][-2]["name"] == "upstream_preflight"
    assert result["stages"][-1]["name"] == "upstream_ingest"
    post_url = mock_get_client.return_value.post.await_args.args[0]
    kwargs = mock_get_client.return_value.post.await_args.kwargs
    mock_get_client.return_value.get.assert_awaited_once_with("http://graphrag.test/health", timeout=5.0)
    assert post_url == "http://graphrag.test/ingest-documents"
    assert kwargs["data"]["domain"] == "docs"
    assert kwargs["files"]["files"][0] == "notes.txt"
    assert result["upstream_ingest"]["sample_outputs"]["payload_kind"] == "original_file"
    assert result["upstream_ingest"]["sample_outputs"]["target_url"] == "http://graphrag.test/ingest-documents"
    assert result["upstream_ingest"]["sample_outputs"]["domain"] == "docs"
    assert result["upstream_ingest"]["sample_outputs"]["payload_sha256"] == result["upstream_preflight"]["sample_outputs"]["payload_sha256"]
    assert result["structured_ingest"]["artifact_available"] is True
    structured_response = await ingest.graphrag_ingest_pipeline_artifact(result["run_id"], "structured")
    assert "GraphRAG connects entities to evidence." in structured_response.body.decode("utf-8")

    remembered = await ingest.graphrag_ingest_pipeline_status(result["run_id"])
    assert remembered["upstream_ingest"]["sample_outputs"]["response"]["processed"] == 1
    listed = await ingest.graphrag_list_ingestion_pipeline_runs(limit=1)
    assert listed["runs"][0]["upstream_preflight"]["payload_sha256"] == result["upstream_preflight"]["sample_outputs"]["payload_sha256"]
    assert listed["runs"][0]["upstream_preflight"]["diagnostics_path"] == "/api/v1/graphrag/deployment/diagnostics"
    assert listed["runs"][0]["upstream_ingest"]["target_url"] == "http://graphrag.test/ingest-documents"
    assert listed["runs"][0]["operational_handoff"]["status"] == "remote_graph_submitted"
    assert listed["runs"][0]["operational_handoff"]["chat_handoff_ready"] is True


@pytest.mark.asyncio
async def test_pipeline_upload_can_return_fast_async_handoff_then_complete_background():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.get = AsyncMock(return_value=_mock_response({"status": "healthy"}))
    mock_get_client.return_value.post = AsyncMock(return_value=_mock_response({"processed": 1, "total_chunks": 2}))
    upload = _mock_upload("notes.txt", b"GraphRAG async upload keeps the UI responsive.", "text/plain")
    background_tasks = BackgroundTasks()

    with patch.object(ingest, "get_http_client", mock_get_client):
        result = await ingest.graphrag_ingest_pipeline_upload(
            upload,
            background_tasks=background_tasks,
            domain="docs",
            use_semantic_chunking=True,
            build_knowledge_graph=True,
            sample_limit=3,
            submit_async=True,
        )

        run_id = result["run_id"]
        assert result["status"] == "submitted"
        assert result["async_submission"]["status"] == "submitted"
        assert result["upstream_ingest"]["status"] == "running"
        assert result["stages"][-1]["name"] == "upstream_ingest"
        assert result["stages"][-1]["status"] == "running"
        assert result["source_document_artifact"]["filename"] == "notes.txt"
        assert result["source_document_artifact"]["size_bytes"] == len(
            b"GraphRAG async upload keeps the UI responsive.",
        )
        artifact_manifest = ingest._pipeline_artifact_manifest(result)
        assert any(
            artifact["kind"] == "source_document" and artifact["available"] is True
            for artifact in artifact_manifest["artifacts"]
        )
        assert result["structured_ingest"]["artifact_available"] is True
        mock_get_client.return_value.post.assert_not_awaited()

        remembered = await ingest.graphrag_ingest_pipeline_status(run_id)
        assert remembered["status"] == "submitted"
        assert remembered["upstream_ingest"]["status"] == "running"

        await background_tasks()

    completed = await ingest.graphrag_ingest_pipeline_status(run_id)
    assert completed["run_id"] == run_id
    assert completed["status"] == "complete"
    assert completed["completed_at"]
    assert completed["async_submission"]["status"] == "complete"
    assert completed["upstream_ingest"]["status"] == "complete"
    assert completed["upstream_preflight"]["ready"] is True
    assert completed["operational_handoff"]["remote_ingest_submitted"] is True
    assert completed["operational_handoff"]["chat_handoff_ready"] is True
    mock_get_client.return_value.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_upload_preflight_falls_back_to_compatible_health_endpoint():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.get = AsyncMock(side_effect=[
        _mock_response({"detail": "not found"}, status_code=404),
        _mock_response({"status": "healthy", "service": "graphrag"}),
    ])
    mock_get_client.return_value.post = AsyncMock(return_value=_mock_response({"processed": 1}))
    upload = _mock_upload("notes.txt", b"GraphRAG health fallback.", "text/plain")

    with patch.object(ingest, "get_http_client", mock_get_client):
        result = await ingest.graphrag_ingest_pipeline_upload(
            upload,
            domain="docs",
            use_semantic_chunking=True,
            build_knowledge_graph=True,
            sample_limit=3,
        )

    assert result["status"] == "complete"
    assert result["completed_at"]
    assert result["upstream_preflight"]["ready"] is True
    assert result["upstream_preflight"]["health_url"] == "http://graphrag.test/api/health"
    assert result["upstream_preflight"]["sample_outputs"]["health_url"] == "http://graphrag.test/api/health"
    assert result["upstream_preflight"]["sample_outputs"]["attempted_health_urls"] == [
        "http://graphrag.test/health",
        "http://graphrag.test/api/health",
    ]
    assert result["upstream_preflight"]["sample_outputs"]["health_attempts"][0]["status"] == "not_found"
    assert mock_get_client.return_value.get.await_args_list[0].args[0] == "http://graphrag.test/health"
    assert mock_get_client.return_value.get.await_args_list[1].args[0] == "http://graphrag.test/api/health"
    mock_get_client.return_value.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_upload_falls_back_to_compatible_ingest_endpoint():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.get = AsyncMock(return_value=_mock_response({"status": "healthy"}))
    mock_get_client.return_value.post = AsyncMock(side_effect=[
        _mock_response({"detail": "not found"}, status_code=404),
        _mock_response({"processed": 1, "endpoint": "process-document"}),
    ])
    upload = _mock_upload("notes.txt", b"GraphRAG fallback evidence.", "text/plain")

    with patch.object(ingest, "get_http_client", mock_get_client):
        result = await ingest.graphrag_ingest_pipeline_upload(
            upload,
            domain="docs",
            use_semantic_chunking=True,
            build_knowledge_graph=True,
            sample_limit=3,
        )

    assert result["status"] == "complete"
    assert result["completed_at"]
    assert result["upstream_ingest"]["status"] == "complete"
    assert result["upstream_ingest"]["submitted"] is True
    assert result["upstream_ingest"]["target_url"] == "http://graphrag.test/process-document"
    assert result["upstream_ingest"]["selected_endpoint"] == "http://graphrag.test/process-document"
    assert result["upstream_ingest"]["next_step"].startswith("Remote ingest succeeded after falling back")
    assert result["upstream_ingest"]["sample_outputs"]["attempted_urls"] == [
        "http://graphrag.test/ingest-documents",
        "http://graphrag.test/process-document",
    ]
    assert result["upstream_ingest"]["sample_outputs"]["endpoint_attempts"][0]["status_code"] == 404
    assert result["upstream_ingest"]["sample_outputs"]["endpoint_attempts"][0]["fallback_allowed"] is True
    assert result["upstream_ingest"]["sample_outputs"]["endpoint_attempts"][1]["status_code"] == 200
    assert result["operational_handoff"]["status"] == "remote_graph_submitted"
    assert result["operational_handoff"]["target_url"] == "http://graphrag.test/process-document"
    assert result["deployment_outcome"]["status"] == "remote_submitted"
    assert result["deployment_outcome"]["severity"] == "success"
    assert result["deployment_outcome"]["ready"] is True
    assert result["deployment_outcome"]["fallback_used"] is True
    assert result["deployment_outcome"]["preflight"]["ready"] is True
    assert result["deployment_outcome"]["preflight"]["health_url"] == "http://graphrag.test/health"
    assert result["deployment_outcome"]["preflight"]["health_attempt_count"] == 1
    assert result["deployment_outcome"]["ingest"]["selected_endpoint"] == "http://graphrag.test/process-document"
    assert result["deployment_outcome"]["ingest"]["endpoint_attempt_count"] == 2
    assert result["deployment_outcome"]["ingest"]["failed_attempt_count"] == 1
    assert result["deployment_outcome"]["operator"]["primary_action"] == "ask_chat"
    assert "upstream_ingest.sample_outputs.endpoint_attempts" in result["deployment_outcome"]["trace_fields"]
    assert result["pipeline_performance"]["stage_count"] == len(result["stages"])
    assert result["pipeline_performance"]["remote_stage_duration_ms"] >= 0
    assert result["pipeline_performance"]["throughput"]["bytes_per_second"] > 0
    assert result["summary"]["pipeline_performance"]["stage_count"] == result["pipeline_performance"]["stage_count"]
    assert [call.args[0] for call in mock_get_client.return_value.post.await_args_list] == [
        "http://graphrag.test/ingest-documents",
        "http://graphrag.test/process-document",
    ]
    listed = await ingest.graphrag_list_ingestion_pipeline_runs(limit=1)
    assert listed["runs"][0]["upstream_ingest"]["selected_endpoint"] == "http://graphrag.test/process-document"
    assert listed["runs"][0]["upstream_ingest"]["endpoint_attempt_count"] == 2
    assert listed["runs"][0]["upstream_ingest"]["attempted_urls"] == [
        "http://graphrag.test/ingest-documents",
        "http://graphrag.test/process-document",
    ]
    assert listed["runs"][0]["deployment_outcome"]["status"] == "remote_submitted"
    assert listed["runs"][0]["deployment_outcome"]["ingest"]["endpoint_attempt_count"] == 2
    assert listed["runs"][0]["deployment_outcome"]["fallback_used"] is True
    assert listed["runs"][0]["pipeline_performance"]["stage_count"] == len(result["stages"])


@pytest.mark.asyncio
async def test_pipeline_upload_records_all_compatible_ingest_failures():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.get = AsyncMock(return_value=_mock_response({"status": "healthy"}))
    mock_get_client.return_value.post = AsyncMock(side_effect=[
        _mock_response({"detail": "missing endpoint"}, status_code=404),
        _mock_response({"detail": "wrong method"}, status_code=405),
        _mock_response({"detail": "payload rejected"}, status_code=422),
    ])
    upload = _mock_upload("notes.txt", b"GraphRAG fallback failure evidence.", "text/plain")

    with patch.object(ingest, "get_http_client", mock_get_client):
        result = await ingest.graphrag_ingest_pipeline_upload(
            upload,
            domain="docs",
            use_semantic_chunking=True,
            build_knowledge_graph=True,
            sample_limit=3,
        )

    assert result["status"] == "upstream_failed"
    assert result["upstream_ingest"]["status"] == "failed"
    assert result["upstream_ingest"]["submitted"] is False
    assert result["upstream_ingest"]["target_url"] == "http://graphrag.test/add-to-vector-store"
    assert result["upstream_ingest"]["status_code"] == 422
    assert result["upstream_ingest"]["next_step"].startswith("All compatible GraphRAG ingest endpoints failed")
    assert result["upstream_ingest"]["sample_outputs"]["attempted_urls"] == [
        "http://graphrag.test/ingest-documents",
        "http://graphrag.test/process-document",
        "http://graphrag.test/add-to-vector-store",
    ]
    assert [attempt["status_code"] for attempt in result["upstream_ingest"]["sample_outputs"]["endpoint_attempts"]] == [404, 405, 422]
    assert result["upstream_ingest"]["sample_outputs"]["endpoint_attempts"][-1]["fallback_allowed"] is False
    assert result["operational_handoff"]["status"] == "remote_graph_ready"
    assert result["operational_handoff"]["primary_action"] == "retry_upload"
    assert result["deployment_outcome"]["status"] == "remote_ingest_failed"
    assert result["deployment_outcome"]["severity"] == "error"
    assert result["deployment_outcome"]["ready"] is False
    assert result["deployment_outcome"]["remote_blocked"] is True
    assert result["deployment_outcome"]["remote_graph_available"] is True
    assert result["deployment_outcome"]["fallback_used"] is True
    assert result["deployment_outcome"]["ingest"]["endpoint_attempt_count"] == 3
    assert result["deployment_outcome"]["ingest"]["failed_attempt_count"] == 3
    assert result["deployment_outcome"]["operator"]["primary_action"] == "retry_upload"
    assert [call.args[0] for call in mock_get_client.return_value.post.await_args_list] == [
        "http://graphrag.test/ingest-documents",
        "http://graphrag.test/process-document",
        "http://graphrag.test/add-to-vector-store",
    ]
    listed = await ingest.graphrag_list_ingestion_pipeline_runs(limit=1)
    assert listed["runs"][0]["upstream_ingest"]["selected_endpoint"] == "http://graphrag.test/add-to-vector-store"
    assert listed["runs"][0]["upstream_ingest"]["endpoint_attempt_count"] == 3
    assert listed["runs"][0]["upstream_ingest"]["attempted_urls"] == [
        "http://graphrag.test/ingest-documents",
        "http://graphrag.test/process-document",
        "http://graphrag.test/add-to-vector-store",
    ]
    assert listed["runs"][0]["deployment_outcome"]["status"] == "remote_ingest_failed"
    assert listed["runs"][0]["deployment_outcome"]["ingest"]["failed_attempt_count"] == 3


@pytest.mark.asyncio
async def test_pipeline_upload_keeps_local_trace_ready_when_remote_ingest_times_out(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_REMOTE_INGEST_TIMEOUT", "1.5")
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.get = AsyncMock(return_value=_mock_response({"status": "healthy"}))
    mock_get_client.return_value.post = AsyncMock(side_effect=httpx.ReadTimeout("remote ingest is still working"))
    upload = _mock_upload("notes.txt", b"GraphRAG timeout evidence still gets a local trace.", "text/plain")

    with patch.object(ingest, "get_http_client", mock_get_client):
        result = await ingest.graphrag_ingest_pipeline_upload(
            upload,
            domain="docs",
            use_semantic_chunking=True,
            build_knowledge_graph=True,
            sample_limit=3,
        )

    assert result["status"] == "preview_only"
    assert result["upstream_ingest"]["status"] == "timed_out"
    assert result["upstream_ingest"]["submitted"] is False
    assert result["upstream_ingest"]["sample_outputs"]["timeout_seconds"] == 1.5
    assert result["upstream_ingest"]["sample_outputs"]["endpoint_attempts"][0]["status"] == "timed_out"
    assert result["upstream_ingest"]["sample_outputs"]["endpoint_attempts"][0]["timeout_seconds"] == 1.5
    assert "local pipeline trace and chat context are ready" in result["upstream_ingest"]["next_step"]
    assert result["operational_handoff"]["status"] == "local_trace_ready"
    assert result["operational_handoff"]["local_trace_ready"] is True
    assert result["operational_handoff"]["chat_handoff_ready"] is True
    assert result["operational_handoff"]["remote_ingest_status"] == "timed_out"
    assert result["operational_handoff"]["primary_action"] == "ask_chat"
    assert result["deployment_outcome"]["status"] == "remote_ingest_timed_out"
    assert result["deployment_outcome"]["severity"] == "warning"
    assert result["deployment_outcome"]["remote_blocked"] is False
    assert result["deployment_outcome"]["operator"]["primary_action"] == "ask_chat"
    kwargs = mock_get_client.return_value.post.await_args.kwargs
    assert kwargs["timeout"] == 1.5


@pytest.mark.asyncio
async def test_pipeline_upload_submits_structured_text_for_complex_documents(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_REMOTE_STRUCTURED_DIGEST_THRESHOLD_CHARS", "1")
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.get = AsyncMock(return_value=_mock_response({"status": "healthy"}))
    mock_get_client.return_value.post = AsyncMock(return_value=_mock_response({"processed": 1, "total_chunks": 4}))
    upload = _mock_upload(
        "strategy.pptx",
        _minimal_pptx_bytes(),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )

    with patch.object(ingest, "get_http_client", mock_get_client):
        result = await ingest.graphrag_ingest_pipeline_upload(
            upload,
            domain="slides",
            use_semantic_chunking=True,
            build_knowledge_graph=True,
            sample_limit=8,
        )

    kwargs = mock_get_client.return_value.post.await_args.kwargs
    submitted_filename, submitted_content, submitted_content_type = kwargs["files"]["files"]
    submitted_text = submitted_content.decode("utf-8")

    assert submitted_filename == "strategy.structured.md"
    assert submitted_content_type == "text/markdown"
    assert "# GraphRAG Remote Ingest Digest: strategy.pptx" in submitted_text
    assert "## Candidate Graph Relationships" in submitted_text
    assert "Revenue --[q1_value]--> 12" in submitted_text
    assert "Forecast Risk" in submitted_text
    assert len(submitted_text) < result["structured_ingest"]["char_count"]
    assert result["structured_ingest_artifact"]["char_count"] == result["structured_ingest"]["char_count"]
    assert "## Extracted Structure" in result["structured_ingest_artifact"]["content"]
    assert result["structured_ingest"]["uses_structured_text"] is True
    assert result["structured_ingest"]["remote_digest_used"] is True
    assert result["upstream_preflight"]["payload_kind"] == "structured_text"
    assert result["upstream_preflight"]["submitted_filename"] == "strategy.structured.md"
    assert result["upstream_preflight"]["service_status"] == "healthy"
    assert result["upstream_preflight"]["sample_outputs"]["remote_digest_used"] is True
    assert result["upstream_preflight"]["sample_outputs"]["remote_source_char_count"] == result["structured_ingest"]["char_count"]
    assert result["upstream_preflight"]["sample_outputs"]["remote_payload_char_count"] == len(submitted_text)
    assert result["upstream_ingest"]["sample_outputs"]["payload_kind"] == "structured_text"
    assert result["upstream_ingest"]["sample_outputs"]["original_filename"] == "strategy.pptx"
    assert result["upstream_ingest"]["sample_outputs"]["remote_digest_used"] is True
    assert result["upstream_ingest"]["sample_outputs"]["timeout_seconds"] == 20.0


@pytest.mark.asyncio
async def test_pipeline_upload_preflight_skips_remote_submit_without_graphrag_url(monkeypatch):
    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    from modules.graphrag.config import reset_config

    reset_config()
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.post = AsyncMock()
    upload = _mock_upload("notes.txt", b"GraphRAG local preview only.", "text/plain")

    with patch.object(ingest, "get_http_client", mock_get_client):
        result = await ingest.graphrag_ingest_pipeline_upload(
            upload,
            domain="docs",
            use_semantic_chunking=True,
            build_knowledge_graph=True,
            sample_limit=3,
        )

    assert result["status"] == "preview_only"
    assert result["operational_handoff"]["status"] == "local_trace_ready"
    assert result["operational_handoff"]["local_trace_ready"] is True
    assert result["operational_handoff"]["chat_handoff_ready"] is True
    assert result["operational_handoff"]["remote_ingest_submitted"] is False
    assert result["operational_handoff"]["primary_action"] == "ask_chat"
    assert result["operational_handoff"]["next_step"].startswith("Set GRAPHRAG_URL")
    assert {action["id"] for action in result["operational_handoff"]["actions"]} >= {"inspect_trace", "ask_chat", "configure_remote_graph"}
    assert result["upstream_preflight"]["status"] == "skipped"
    assert result["upstream_preflight"]["ready"] is False
    assert result["upstream_preflight"]["next_step"].startswith("Set GRAPHRAG_URL")
    assert result["upstream_preflight"]["sample_outputs"]["services_status_path"] == "/api/v1/graphrag/services/status"
    assert result["upstream_preflight"]["sample_outputs"]["diagnostics_path"] == "/api/v1/graphrag/deployment/diagnostics"
    assert result["upstream_ingest"]["status"] == "skipped"
    assert result["upstream_ingest"]["submitted"] is False
    assert result["upstream_ingest"]["next_step"].startswith("Set GRAPHRAG_URL")
    assert result["upstream_ingest"]["sample_outputs"]["payload_bytes"] == len(b"GraphRAG local preview only.")
    assert len(result["upstream_ingest"]["sample_outputs"]["payload_sha256"]) == 64
    assert result["deployment_outcome"]["status"] == "not_configured"
    assert result["deployment_outcome"]["severity"] == "info"
    assert result["deployment_outcome"]["remote_blocked"] is True
    assert result["deployment_outcome"]["preflight"]["ready"] is False
    assert result["deployment_outcome"]["operator"]["primary_action"] == "ask_chat"
    assert result["deployment_outcome"]["operator"]["services_status_path"] == "/api/v1/graphrag/services/status"
    assert result["pipeline_performance"]["stage_count"] == len(result["stages"])
    assert result["pipeline_performance"]["timings"][-1]["name"] == "upstream_ingest"
    mock_get_client.return_value.post.assert_not_called()

    reset_config()


@pytest.mark.asyncio
async def test_pipeline_upload_preflight_skips_remote_submit_when_health_unavailable():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.get = AsyncMock(side_effect=RuntimeError("connection refused"))
    mock_get_client.return_value.post = AsyncMock()
    upload = _mock_upload("notes.txt", b"GraphRAG local preview only.", "text/plain")

    with patch.object(ingest, "get_http_client", mock_get_client):
        result = await ingest.graphrag_ingest_pipeline_upload(
            upload,
            domain="docs",
            use_semantic_chunking=True,
            build_knowledge_graph=True,
            sample_limit=3,
        )

    assert result["status"] == "preview_only"
    assert result["operational_handoff"]["status"] == "local_trace_ready"
    assert result["operational_handoff"]["local_trace_ready"] is True
    assert result["operational_handoff"]["chat_handoff_ready"] is True
    assert result["operational_handoff"]["remote_ingest_ready"] is False
    assert result["operational_handoff"]["remote_ingest_submitted"] is False
    assert result["operational_handoff"]["primary_action"] == "service_smoke_check"
    assert result["operational_handoff"]["service_status"] == "unavailable"
    assert {action["id"] for action in result["operational_handoff"]["actions"]} >= {"inspect_trace", "ask_chat", "service_smoke_check", "retry_upload"}
    assert result["upstream_preflight"]["status"] == "unavailable"
    assert result["upstream_preflight"]["ready"] is False
    assert result["upstream_preflight"]["service_status"] == "unavailable"
    assert "health check failed" in result["upstream_preflight"]["message"]
    assert result["upstream_preflight"]["sample_outputs"]["health_error"] == "connection refused"
    assert result["upstream_preflight"]["sample_outputs"]["next_step"].startswith("Open the Service Smoke Check")
    assert result["upstream_ingest"]["status"] == "skipped"
    assert result["upstream_ingest"]["reason"] == "preflight_not_ready"
    assert result["upstream_ingest"]["sample_outputs"]["services_status_path"] == "/api/v1/graphrag/services/status"
    assert result["upstream_ingest"]["sample_outputs"]["target_url"] == "http://graphrag.test/ingest-documents"
    assert result["upstream_ingest"]["sample_outputs"]["deployment_hint"].startswith("GRAPHRAG_ENABLED=true")
    mock_get_client.return_value.get.assert_awaited_once_with("http://graphrag.test/health", timeout=5.0)
    mock_get_client.return_value.post.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_upload_preflight_skips_remote_submit_when_health_reports_unhealthy():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.get = AsyncMock(return_value=_mock_response({"status": "unhealthy", "detail": "neo4j unavailable"}))
    mock_get_client.return_value.post = AsyncMock()
    upload = _mock_upload("notes.txt", b"GraphRAG local preview only.", "text/plain")

    with patch.object(ingest, "get_http_client", mock_get_client):
        result = await ingest.graphrag_ingest_pipeline_upload(
            upload,
            domain="docs",
            use_semantic_chunking=True,
            build_knowledge_graph=True,
            sample_limit=3,
        )

    assert result["status"] == "preview_only"
    assert result["upstream_preflight"]["status"] == "unhealthy"
    assert result["upstream_preflight"]["ready"] is False
    assert result["upstream_preflight"]["service_status"] == "unhealthy"
    assert "reported unhealthy" in result["upstream_preflight"]["message"]
    assert result["upstream_ingest"]["status"] == "skipped"
    mock_get_client.return_value.post.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_batch_upload_processes_multiple_files_with_traces():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.get = AsyncMock(return_value=_mock_response({"status": "healthy"}))
    mock_get_client.return_value.post = AsyncMock(return_value=_mock_response({"processed": 1, "total_chunks": 2}))
    uploads = [
        _mock_upload("notes.txt", b"GraphRAG connects entities to evidence.", "text/plain"),
        _mock_upload(
            "strategy.pptx",
            _minimal_pptx_bytes(),
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
    ]

    with patch.object(ingest, "get_http_client", mock_get_client):
        result = await ingest.graphrag_ingest_pipeline_batch(
            uploads,
            domain="batch",
            use_semantic_chunking=True,
            build_knowledge_graph=True,
            sample_limit=6,
            max_concurrency=2,
        )

    assert result["status"] == "complete"
    assert result["total_files"] == 2
    assert result["processed"] == 2
    assert result["failed"] == 0
    assert result["concurrency"] == 2
    assert result["summary"]["status_counts"] == {"complete": 2}
    assert result["summary"]["payload_kinds"] == {"original_file": 1, "structured_text": 1}
    assert result["summary"]["remote_submitted"] == 2
    assert result["summary"]["local_preview_only"] == 0
    assert result["summary"]["operational_handoff"]["status"] == "remote_graph_submitted"
    assert result["summary"]["operational_handoff"]["trace_ready"] == 2
    assert result["summary"]["operational_handoff"]["chat_ready"] == 2
    assert result["summary"]["operational_handoff"]["remote_submitted"] == 2
    assert result["summary"]["operational_handoff"]["remote_waiting"] == 0
    assert result["summary"]["operational_handoff"]["primary_action"] == "ask_chat_across_batch"
    assert result["summary"]["operational_handoff"]["handoff_status_counts"] == {"remote_graph_submitted": 2}
    assert result["batch_resume_handoff"] == result["summary"]["batch_resume_handoff"]
    batch_handoff = result["batch_resume_handoff"]
    assert batch_handoff["ready"] is True
    assert batch_handoff["chat_context_endpoint"] == "/api/v1/graphrag/chat-context-v2"
    assert batch_handoff["chat_context_request"]["domain"] == "batch"
    assert batch_handoff["chat_context_request"]["pipeline_run_ids"] == [run["run_id"] for run in result["runs"]]
    assert batch_handoff["chat_context_request"]["include_graph"] is True
    assert batch_handoff["chat_context_request"]["use_local_pipeline_context"] is True
    assert batch_handoff["chat_context_request"]["include_local_pipeline"] is True
    assert batch_handoff["route_state"]["graphragHandoff"]["runIds"] == [run["run_id"] for run in result["runs"]]
    assert batch_handoff["route_state"]["graphragHandoff"]["chatContextRequest"]["pipeline_run_ids"] == [run["run_id"] for run in result["runs"]]
    assert result["summary"]["operational_handoff"]["chat_context_request"]["pipeline_run_ids"] == [run["run_id"] for run in result["runs"]]
    assert result["upload_manifest"] == result["summary"]["upload_manifest"]
    assert result["upload_manifest"]["file_count"] == 2
    assert result["upload_manifest"]["accepted"] == 2
    assert result["upload_manifest"]["failed"] == 0
    assert result["upload_manifest"]["trace_ready"] == 2
    assert result["upload_manifest"]["chat_ready"] == 2
    assert result["upload_manifest"]["remote_submitted"] == 2
    assert result["upload_manifest"]["remote_waiting"] == 0
    assert result["upload_manifest"]["primary_action"] == "ask_chat_across_batch"
    assert result["upload_manifest"]["payload_kinds"] == {"original_file": 1, "structured_text": 1}
    assert result["upload_manifest"]["domains"] == ["batch"]
    assert result["upload_manifest"]["batch_resume_handoff"]["chat_context_request"]["pipeline_run_ids"] == [run["run_id"] for run in result["runs"]]
    assert result["upload_manifest"]["total_bytes"] > 0
    assert result["upload_manifest"]["largest_file"]["filename"] in {"notes.txt", "strategy.pptx"}
    manifest_lanes = {lane["id"]: lane for lane in result["upload_manifest"]["lanes"]}
    assert manifest_lanes["upload"]["complete"] is True
    assert manifest_lanes["trace"]["complete"] is True
    assert manifest_lanes["chat"]["complete"] is True
    assert manifest_lanes["remote_graph"]["complete"] is True
    assert [item["filename"] for item in result["upload_manifest"]["files"]] == ["notes.txt", "strategy.pptx"]
    assert result["upload_manifest"]["files"][0]["payload_kind"] == "original_file"
    assert result["upload_manifest"]["files"][0]["chat_ready"] is True
    assert result["upload_manifest"]["files"][0]["remote_submitted"] is True
    assert result["upload_manifest"]["files"][0]["deployment_status"] == "remote_submitted"
    assert result["upload_manifest"]["files"][0]["deployment_severity"] == "success"
    assert result["upload_manifest"]["files"][0]["endpoint_attempt_count"] == 1
    assert result["upload_manifest"]["files"][0]["endpoint_fallback_used"] is False
    assert result["upload_manifest"]["files"][0]["layout_element_count"] == 0
    assert result["upload_manifest"]["files"][1]["payload_kind"] == "structured_text"
    assert result["upload_manifest"]["files"][1]["target_url"] == "http://graphrag.test/ingest-documents"
    assert result["upload_manifest"]["files"][1]["deployment_status"] == "remote_submitted"
    assert result["upload_manifest"]["files"][1]["layout_element_count"] >= 1
    assert result["summary"]["totals"]["chunks"] >= 2
    assert result["upload_manifest"]["layout_elements"] >= 1
    assert result["upload_manifest"]["layout_relationships"] >= 1
    assert result["summary"]["evidence_totals"]["layout_elements"] >= 1
    assert result["summary"]["evidence_totals"]["distinct_tables"] >= 1
    assert result["summary"]["evidence_totals"]["shapes"] >= 1
    assert result["summary"]["evidence_totals"]["connectors"] >= 1
    assert result["summary"]["evidence_totals"]["layout_relationships"] >= 1
    assert len(result["runs"]) == 2
    assert [item["filename"] for item in result["results"]] == ["notes.txt", "strategy.pptx"]
    assert result["runs"][0]["upstream_ingest"]["sample_outputs"]["payload_kind"] == "original_file"
    assert result["runs"][1]["upstream_ingest"]["sample_outputs"]["payload_kind"] == "structured_text"
    assert result["runs"][0]["upstream_preflight"]["payload_kind"] == "original_file"
    assert result["runs"][1]["upstream_preflight"]["payload_kind"] == "structured_text"
    assert result["runs"][0]["upstream_preflight"]["sample_outputs"]["shared_health_probe"] is True
    assert result["runs"][1]["upstream_preflight"]["sample_outputs"]["shared_health_probe"] is True
    mock_get_client.return_value.get.assert_awaited_once_with("http://graphrag.test/health", timeout=5.0)

    first_call = mock_get_client.return_value.post.await_args_list[0].kwargs
    second_call = mock_get_client.return_value.post.await_args_list[1].kwargs
    submitted_filenames = {
        first_call["files"]["files"][0],
        second_call["files"]["files"][0],
    }
    assert submitted_filenames == {"notes.txt", "strategy.structured.md"}

    remembered = await ingest.graphrag_ingest_pipeline_status(result["runs"][1]["run_id"])
    assert remembered["document"]["filename"] == "strategy.pptx"


def test_pipeline_batch_summary_includes_cross_document_graph_overview():
    ingest = load_graphrag_route_module("ingest")
    results = [
        {
            "index": 0,
            "filename": "alpha.md",
            "success": True,
            "status": "complete",
            "run": {
                "run_id": "run-alpha",
                "status": "complete",
                "document": {"filename": "alpha.md"},
                "summary": {"chunk_count": 1, "entity_candidate_count": 2, "relationship_candidate_count": 1},
                "document_profile": {
                    "evidence_profile": {
                        "quick_counts": {
                            "layout_elements": 2,
                            "tables": 1,
                            "table_elements": 3,
                            "links": 2,
                            "forms": 1,
                            "diagrams": 3,
                            "code_blocks": 1,
                            "comments": 1,
                            "revisions": 1,
                            "styled_text_blocks": 2,
                            "message_body_parts": 1,
                        },
                        "graph_evidence": {
                            "relationship_count": 7,
                            "semantic_relationship_count": 4,
                            "structural_relationship_count": 3,
                            "layout_relationship_count": 1,
                            "link_relationship_count": 2,
                            "code_block_relationship_count": 2,
                            "diagram_relationship_count": 3,
                            "styled_text_relationship_count": 1,
                        },
                    },
                },
                "interaction_handoff": {
                    "ready": True,
                    "recommended_mode": "graph_path_reasoning",
                    "counts": {
                        "graph_anchors": 2,
                        "starter_prompts": 1,
                        "capability_cards": 1,
                        "retrieval_rehearsals": 1,
                        "rehearsed_chunk_matches": 1,
                        "rehearsed_anchor_matches": 2,
                    },
                },
                "local_graph": {
                    "nodes": [
                        {"id": "Shared Metric", "name": "Shared Metric", "type": "metric"},
                        {"id": "Alpha Plan", "name": "Alpha Plan", "type": "document"},
                    ],
                    "edges": [
                        {
                            "source": "Alpha Plan",
                            "label": "supports",
                            "target": "Shared Metric",
                            "weight": 0.91,
                            "evidence": "Alpha plan supports shared metric.",
                        },
                    ],
                },
            },
        },
        {
            "index": 1,
            "filename": "beta.md",
            "success": True,
            "status": "complete",
            "run": {
                "run_id": "run-beta",
                "status": "complete",
                "document": {"filename": "beta.md"},
                "summary": {"chunk_count": 1, "entity_candidate_count": 2, "relationship_candidate_count": 1},
                "document_profile": {
                    "evidence_profile": {
                        "quick_counts": {
                            "layout_elements": 1,
                            "tables": 2,
                            "table_elements": 4,
                            "links": 1,
                            "forms": 2,
                            "diagrams": 2,
                            "code_blocks": 2,
                            "styled_text_blocks": 1,
                            "speaker_notes": 1,
                            "hidden_slides": 1,
                        },
                        "graph_evidence": {
                            "relationship_count": 6,
                            "semantic_relationship_count": 3,
                            "structural_relationship_count": 3,
                            "layout_relationship_count": 0,
                            "link_relationship_count": 1,
                            "code_block_relationship_count": 2,
                            "diagram_relationship_count": 2,
                            "speaker_notes_relationship_count": 1,
                        },
                    },
                },
                "interaction_handoff": {
                    "ready": True,
                    "recommended_mode": "relationship_exploration",
                    "counts": {
                        "graph_anchors": 1,
                        "starter_prompts": 1,
                        "capability_cards": 1,
                        "retrieval_rehearsals": 1,
                        "rehearsed_chunk_matches": 1,
                        "rehearsed_anchor_matches": 1,
                    },
                },
                "local_graph": {
                    "nodes": [
                        {"id": "Shared Metric", "name": "Shared Metric", "type": "metric"},
                        {"id": "Beta Forecast", "name": "Beta Forecast", "type": "document"},
                    ],
                    "edges": [
                        {
                            "source": "Beta Forecast",
                            "label": "explains",
                            "target": "Shared Metric",
                            "weight": 0.83,
                            "evidence": "Beta forecast explains shared metric.",
                        },
                    ],
                },
            },
        },
    ]

    summary = ingest._pipeline_batch_summary(results)
    overview = summary["graph_overview"]

    assert overview["source_count"] == 2
    assert overview["node_count"] == 4
    assert overview["relationship_count"] == 2
    assert overview["relationship_type_counts"] == {"supports": 1, "explains": 1}
    assert overview["shared_entity_count"] == 1
    assert overview["shared_entities"][0]["name"] == "Shared Metric"
    assert overview["shared_entities"][0]["run_count"] == 2
    assert overview["top_relationships"][0]["relationship_type"] == "semantic"
    assert overview["bridge_entity_count"] == 1
    assert overview["bridge_entities"][0]["name"] == "Shared Metric"
    assert overview["bridge_entities"][0]["relationship_count"] == 2
    assert overview["cross_document_path_count"] >= 1
    assert overview["cross_document_paths"][0]["bridge"] == "Shared Metric"
    assert overview["cross_document_paths"][0]["hops"] == 2
    assert set(overview["cross_document_paths"][0]["filenames"]) == {"alpha.md", "beta.md"}
    assert overview["question_count"] >= 4
    assert [question["kind"] for question in overview["suggested_questions"][:3]] == ["summary", "path", "bridge"]
    assert overview["suggested_questions"][1]["label"].startswith("Path:")
    assert overview["suggested_questions"][1]["source"] in {"Alpha Plan", "Beta Forecast"}
    assert overview["suggested_questions"][1]["bridge"] == "Shared Metric"
    assert overview["suggested_questions"][1]["target"] in {"Alpha Plan", "Beta Forecast"}
    assert overview["suggested_questions"][1]["source"] != overview["suggested_questions"][1]["target"]
    assert '"Shared Metric"' in overview["suggested_questions"][2]["prompt"]
    batch_handoff = summary["batch_resume_handoff"]
    assert batch_handoff["ready"] is True
    assert batch_handoff["run_ids"] == ["run-alpha", "run-beta"]
    assert batch_handoff["chat_context_endpoint"] == "/api/v1/graphrag/chat-context-v2"
    assert batch_handoff["chat_context_request"]["pipeline_run_ids"] == ["run-alpha", "run-beta"]
    assert batch_handoff["chat_context_request"]["include_graph"] is True
    assert batch_handoff["chat_context_request"]["use_local_pipeline_context"] is True
    assert batch_handoff["chat_context_request"]["include_local_pipeline"] is True
    assert batch_handoff["chat_context_request"]["pipeline_evidence_totals"]["code_blocks"] == 3
    assert batch_handoff["chat_context_request"]["pipeline_evidence_totals"]["diagram_relationships"] == 5
    assert "code_blocks" in batch_handoff["chat_context_request"]["pipeline_evidence_families"]
    assert batch_handoff["chat_context_request"]["graph_overview_counts"]["cross_document_paths"] >= 1
    assert batch_handoff["chat_context_request"]["source"] in {"Alpha Plan", "Beta Forecast"}
    assert batch_handoff["chat_context_request"]["bridge"] == "Shared Metric"
    assert batch_handoff["chat_context_request"]["target"] in {"Alpha Plan", "Beta Forecast"}
    assert batch_handoff["chat_context_request"]["source"] != batch_handoff["chat_context_request"]["target"]
    assert batch_handoff["route_state"]["graphragHandoff"]["chatContextRequest"]["pipeline_run_ids"] == ["run-alpha", "run-beta"]
    assert batch_handoff["route_state"]["graphragHandoff"]["evidenceTotals"]["links"] == 3
    assert "diagram_relationships" in batch_handoff["route_state"]["graphragHandoff"]["evidenceFamilies"]
    assert batch_handoff["route_state"]["graphragHandoff"]["suggestedQuestions"][1]["bridge"] == "Shared Metric"
    assert summary["operational_handoff"]["chat_context_request"]["pipeline_run_ids"] == ["run-alpha", "run-beta"]
    assert summary["upload_manifest"]["file_count"] == 2
    assert summary["upload_manifest"]["trace_ready"] == 2
    assert summary["upload_manifest"]["chat_ready"] == 2
    assert summary["upload_manifest"]["batch_resume_handoff"]["chat_context_request"]["pipeline_run_ids"] == ["run-alpha", "run-beta"]
    assert summary["interaction_mode_counts"] == {"graph_path_reasoning": 1, "relationship_exploration": 1}
    assert summary["upload_manifest"]["interaction_mode_counts"] == summary["interaction_mode_counts"]
    assert summary["interaction_rehearsal_totals"] == {
        "retrieval_rehearsals": 2,
        "chunk_matches": 2,
        "anchor_matches": 3,
    }
    assert summary["evidence_totals"]["layout_elements"] == 3
    assert summary["evidence_totals"]["distinct_tables"] == 3
    assert summary["evidence_totals"]["table_elements"] == 7
    assert summary["evidence_totals"]["links"] == 3
    assert summary["evidence_totals"]["forms"] == 3
    assert summary["evidence_totals"]["diagrams"] == 5
    assert summary["evidence_totals"]["code_blocks"] == 3
    assert summary["evidence_totals"]["comments"] == 1
    assert summary["evidence_totals"]["revisions"] == 1
    assert summary["evidence_totals"]["styled_text_blocks"] == 3
    assert summary["evidence_totals"]["message_body_parts"] == 1
    assert summary["evidence_totals"]["speaker_notes"] == 1
    assert summary["evidence_totals"]["hidden_slides"] == 1
    assert summary["evidence_totals"]["graph_relationships"] == 13
    assert summary["evidence_totals"]["semantic_relationships"] == 7
    assert summary["evidence_totals"]["structural_relationships"] == 6
    assert summary["evidence_totals"]["layout_relationships"] == 1
    assert summary["evidence_totals"]["link_relationships"] == 3
    assert summary["evidence_totals"]["code_block_relationships"] == 4
    assert summary["evidence_totals"]["diagram_relationships"] == 5
    assert summary["evidence_totals"]["styled_text_relationships"] == 1
    assert summary["evidence_totals"]["speaker_notes_relationships"] == 1
    assert summary["upload_manifest"]["layout_elements"] == 3
    assert summary["upload_manifest"]["links"] == 3
    assert summary["upload_manifest"]["forms"] == 3
    assert summary["upload_manifest"]["diagrams"] == 5
    assert summary["upload_manifest"]["code_blocks"] == 3
    assert summary["upload_manifest"]["graph_relationships"] == 13
    assert summary["upload_manifest"]["layout_relationships"] == 1
    assert summary["upload_manifest"]["link_relationships"] == 3
    assert summary["upload_manifest"]["code_block_relationships"] == 4
    assert summary["upload_manifest"]["diagram_relationships"] == 5
    assert summary["upload_manifest"]["retrieval_rehearsals"] == 2
    assert summary["upload_manifest"]["rehearsed_anchor_matches"] == 3
    assert [item["filename"] for item in summary["upload_manifest"]["files"]] == ["alpha.md", "beta.md"]
    assert summary["upload_manifest"]["files"][0]["interaction_mode"] == "graph_path_reasoning"
    assert summary["upload_manifest"]["files"][0]["graph_anchor_count"] == 2
    assert summary["upload_manifest"]["files"][0]["retrieval_rehearsal_count"] == 1
    assert summary["upload_manifest"]["files"][0]["layout_element_count"] == 2
    assert summary["upload_manifest"]["files"][0]["link_count"] == 2
    assert summary["upload_manifest"]["files"][0]["form_count"] == 1
    assert summary["upload_manifest"]["files"][0]["diagram_count"] == 3
    assert summary["upload_manifest"]["files"][0]["code_block_count"] == 1
    assert summary["upload_manifest"]["files"][0]["evidence_counts"]["revisions"] == 1
    assert summary["upload_manifest"]["files"][0]["evidence_counts"]["styled_text_blocks"] == 2
    assert summary["upload_manifest"]["files"][0]["graph_evidence_counts"]["link_relationship_count"] == 2
    assert summary["upload_manifest"]["files"][1]["layout_element_count"] == 1
    assert summary["upload_manifest"]["files"][1]["link_count"] == 1
    assert summary["upload_manifest"]["files"][1]["form_count"] == 2
    assert summary["upload_manifest"]["files"][1]["diagram_count"] == 2
    assert summary["upload_manifest"]["files"][1]["code_block_count"] == 2
    assert summary["upload_manifest"]["files"][1]["evidence_counts"]["hidden_slides"] == 1
    assert summary["upload_manifest"]["files"][1]["evidence_counts"]["speaker_notes_relationships"] == 1
    assert summary["upload_manifest"]["files"][1]["graph_evidence_counts"]["code_block_relationship_count"] == 2


@pytest.mark.asyncio
async def test_pipeline_batch_upload_reports_execution_metrics_and_keeps_input_order():
    ingest = load_graphrag_route_module("ingest")
    uploads = [
        _mock_upload("slow.txt", b"slow", "text/plain"),
        _mock_upload("fast.txt", b"fast", "text/plain"),
        _mock_upload("third.txt", b"third", "text/plain"),
    ]
    delays = {
        "slow.txt": 0.03,
        "fast.txt": 0.005,
        "third.txt": 0.0,
    }
    active = 0
    observed_peak = 0
    active_lock = asyncio.Lock()

    async def fake_pipeline_run(*, filename, **kwargs):
        nonlocal active, observed_peak
        assert all(upload.read.await_count == 1 for upload in uploads)
        async with active_lock:
            active += 1
            observed_peak = max(observed_peak, active)
        try:
            await asyncio.sleep(delays[str(filename)])
            return {
                "run_id": f"run-{filename}",
                "status": "complete",
                "document": {
                    "filename": filename,
                    "domain": kwargs.get("domain"),
                    "content_type": kwargs.get("content_type"),
                    "size_bytes": len(filename or ""),
                },
                "summary": {
                    "chunk_count": 1,
                    "entity_candidate_count": 1,
                    "relationship_candidate_count": 1,
                    "readiness_score": 88,
                },
                "upstream_preflight": {
                    "status": "ready",
                    "payload_kind": "original_file",
                    "sample_outputs": {"payload_kind": "original_file"},
                },
                "upstream_ingest": {
                    "status": "complete",
                    "submitted": True,
                    "sample_outputs": {"payload_kind": "original_file"},
                },
                "stages": [],
            }
        finally:
            async with active_lock:
                active -= 1

    with (
        patch.object(ingest, "_shared_batch_health_probe", AsyncMock(return_value=None)),
        patch.object(ingest, "_build_and_submit_pipeline_run", fake_pipeline_run),
    ):
        result = await ingest.graphrag_ingest_pipeline_batch(
            uploads,
            domain="batch",
            use_semantic_chunking=True,
            build_knowledge_graph=True,
            sample_limit=3,
            max_concurrency=2,
        )

    assert result["status"] == "complete"
    assert result["requested_concurrency"] == 2
    assert result["max_concurrency"] == 2
    assert result["concurrency"] == 2
    assert result["peak_concurrency"] == 2
    assert observed_peak == 2
    assert result["duration_ms"] >= max(item["duration_ms"] for item in result["results"])
    assert result["throughput_files_per_second"] > 0
    assert result["started_at"]
    assert result["completed_at"]
    assert result["summary"]["execution"]["duration_ms"] == result["duration_ms"]
    assert result["summary"]["execution"]["max_concurrency"] == 2
    assert result["summary"]["execution"]["peak_concurrency"] == 2
    assert result["summary"]["execution"]["upload_materialized"] is True
    assert result["summary"]["execution"]["upload_read_before_processing"] is True
    assert result["summary"]["execution"]["materialized_file_count"] == 3
    assert result["summary"]["execution"]["upload_read_failures"] == 0
    assert result["summary"]["execution"]["materialized_total_bytes"] == len(b"slowfastthird")
    assert result["summary"]["remote_submitted"] == 3
    assert [item["filename"] for item in result["results"]] == ["slow.txt", "fast.txt", "third.txt"]
    assert [run["document"]["filename"] for run in result["runs"]] == ["slow.txt", "fast.txt", "third.txt"]
    assert sorted(item["completion_order"] for item in result["results"]) == [1, 2, 3]
    assert result["results"][0]["completion_order"] > result["results"][1]["completion_order"]
    assert all(item["duration_ms"] >= 0 for item in result["results"])
    assert all(item["read_duration_ms"] >= 0 for item in result["results"])
    assert [item["size_bytes"] for item in result["results"]] == [4, 4, 5]


@pytest.mark.asyncio
async def test_pipeline_batch_upload_keeps_good_files_when_one_upload_read_fails():
    ingest = load_graphrag_route_module("ingest")
    good_upload = _mock_upload("good.txt", b"GraphRAG evidence", "text/plain")
    bad_upload = _mock_upload("bad.txt", b"", "text/plain")
    bad_upload.read = AsyncMock(side_effect=RuntimeError("stream closed"))

    async def fake_pipeline_run(*, filename, **kwargs):
        return {
            "run_id": f"run-{filename}",
            "status": "complete",
            "document": {
                "filename": filename,
                "domain": kwargs.get("domain"),
                "content_type": kwargs.get("content_type"),
                "size_bytes": len(kwargs.get("content") or b""),
            },
            "summary": {
                "chunk_count": 1,
                "entity_candidate_count": 1,
                "relationship_candidate_count": 1,
                "readiness_score": 88,
            },
            "upstream_preflight": {
                "status": "ready",
                "payload_kind": "original_file",
                "sample_outputs": {"payload_kind": "original_file"},
            },
            "upstream_ingest": {
                "status": "complete",
                "submitted": True,
                "sample_outputs": {"payload_kind": "original_file"},
            },
            "stages": [],
        }

    with (
        patch.object(ingest, "_shared_batch_health_probe", AsyncMock(return_value=None)),
        patch.object(ingest, "_build_and_submit_pipeline_run", fake_pipeline_run),
    ):
        result = await ingest.graphrag_ingest_pipeline_batch(
            [bad_upload, good_upload],
            domain="batch",
            use_semantic_chunking=True,
            build_knowledge_graph=True,
            sample_limit=3,
            max_concurrency=2,
        )

    assert result["status"] == "partial"
    assert result["processed"] == 1
    assert result["failed"] == 1
    assert [item["filename"] for item in result["results"]] == ["bad.txt", "good.txt"]
    assert result["results"][0]["status"] == "upload_read_failed"
    assert result["results"][0]["completion_order"] == 1
    assert result["results"][1]["status"] == "complete"
    assert result["results"][1]["completion_order"] == 2
    assert result["summary"]["execution"]["materialized_file_count"] == 1
    assert result["summary"]["execution"]["upload_read_failures"] == 1
    assert result["summary"]["execution"]["upload_read_before_processing"] is True


@pytest.mark.asyncio
async def test_pipeline_batch_upload_shares_unhealthy_preflight_without_remote_posts():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.get = AsyncMock(return_value=_mock_response({"status": "unhealthy"}, status_code=200))
    mock_get_client.return_value.post = AsyncMock()
    uploads = [
        _mock_upload("notes.txt", b"GraphRAG connects entities to evidence.", "text/plain"),
        _mock_upload("strategy.txt", b"GraphRAG strategy evidence.", "text/plain"),
    ]

    with patch.object(ingest, "get_http_client", mock_get_client):
        result = await ingest.graphrag_ingest_pipeline_batch(
            uploads,
            domain="batch",
            use_semantic_chunking=True,
            build_knowledge_graph=True,
            sample_limit=3,
            max_concurrency=2,
        )

    assert result["status"] == "complete"
    assert result["processed"] == 2
    assert result["failed"] == 0
    assert result["summary"]["status_counts"] == {"preview_only": 2}
    assert result["summary"]["payload_kinds"] == {"original_file": 2}
    assert result["summary"]["remote_submitted"] == 0
    assert result["summary"]["local_preview_only"] == 2
    assert result["summary"]["operational_handoff"]["status"] == "local_traces_ready"
    assert result["summary"]["operational_handoff"]["trace_ready"] == 2
    assert result["summary"]["operational_handoff"]["chat_ready"] == 2
    assert result["summary"]["operational_handoff"]["remote_submitted"] == 0
    assert result["summary"]["operational_handoff"]["remote_waiting"] == 2
    assert result["summary"]["operational_handoff"]["primary_action"] == "ask_chat_across_batch"
    assert result["summary"]["operational_handoff"]["handoff_status_counts"] == {"local_trace_ready": 2}
    assert result["upload_manifest"]["file_count"] == 2
    assert result["upload_manifest"]["accepted"] == 2
    assert result["upload_manifest"]["trace_ready"] == 2
    assert result["upload_manifest"]["chat_ready"] == 2
    assert result["upload_manifest"]["remote_submitted"] == 0
    assert result["upload_manifest"]["remote_waiting"] == 2
    assert result["upload_manifest"]["primary_action"] == "ask_chat_across_batch"
    unhealthy_lanes = {lane["id"]: lane for lane in result["upload_manifest"]["lanes"]}
    assert unhealthy_lanes["remote_graph"]["complete"] is False
    assert unhealthy_lanes["remote_graph"]["waiting"] == 2
    assert all(item["service_status"] == "unhealthy" for item in result["upload_manifest"]["files"])
    assert all(item["remote_waiting"] is True for item in result["upload_manifest"]["files"])
    assert result["summary"]["next_steps"]
    assert len(result["runs"]) == 2
    assert {run["status"] for run in result["runs"]} == {"preview_only"}
    assert {run["upstream_preflight"]["status"] for run in result["runs"]} == {"unhealthy"}
    assert {run["upstream_ingest"]["status"] for run in result["runs"]} == {"skipped"}
    assert all(run["upstream_preflight"]["sample_outputs"]["shared_health_probe"] is True for run in result["runs"])
    mock_get_client.return_value.get.assert_awaited_once_with("http://graphrag.test/health", timeout=5.0)
    mock_get_client.return_value.post.assert_not_called()


@pytest.mark.asyncio
async def test_document_list_forwards_to_upstream_documents_list():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.get = AsyncMock(
        return_value=_mock_response({"documents": [{"name": "notes.txt"}], "total_documents": 1})
    )

    with (
        patch.object(ingest, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(ingest, "get_http_client", mock_get_client),
    ):
        result = await ingest.graphrag_list_documents()

    assert result["total_documents"] == 1
    mock_get_client.return_value.get.assert_awaited_once_with("http://graphrag.test/documents/list")


@pytest.mark.asyncio
async def test_document_delete_encodes_document_name_for_upstream():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.delete = AsyncMock(return_value=_mock_response({"deleted": True}))

    with (
        patch.object(ingest, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(ingest, "get_http_client", mock_get_client),
    ):
        result = await ingest.graphrag_delete_document("docs/note 1.md")

    assert result["deleted"] is True
    mock_get_client.return_value.delete.assert_awaited_once_with(
        "http://graphrag.test/documents/docs%2Fnote%201.md"
    )


@pytest.mark.asyncio
async def test_rebuild_kg_forwards_domain_as_upstream_params():
    ingest = load_graphrag_route_module("ingest")
    mock_get_client = Mock()
    mock_get_client.return_value.post = AsyncMock(return_value=_mock_response({"message": "rebuilt"}))
    request = Mock(spec=Request)
    request.json = AsyncMock(return_value={"domain": "docs"})

    with (
        patch.object(ingest, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(ingest, "get_http_client", mock_get_client),
    ):
        result = await ingest.graphrag_rebuild_kg(request)

    assert result["message"] == "rebuilt"
    mock_get_client.return_value.post.assert_awaited_once_with(
        "http://graphrag.test/rebuild-knowledge-graph",
        params={"domain": "docs"},
        timeout=600.0,
    )


@pytest.mark.asyncio
async def test_text_code_detection_works_without_legacy_graphrag_url(monkeypatch):
    from modules.graphrag.config import reset_config
    search = load_graphrag_route_module("search")

    monkeypatch.setenv("GRAPHRAG_URL", "")
    reset_config()

    request = Mock(spec=Request)
    request.headers = {}
    request.json = AsyncMock(return_value={
        "filename": "hello.py",
        "text": "def hello():\n    return 'world'",
    })

    result = await search.graphrag_detect_code(request)

    assert result["has_code"] is True
    assert result["language"] == "python"
    assert result["statistics"]["total_blocks"] == 1
    assert result["code_blocks"][0]["content"].startswith("def hello")


@pytest.mark.asyncio
async def test_direct_code_rag_search_is_normalized(monkeypatch):
    from modules.graphrag.config import reset_config
    search = load_graphrag_route_module("search")

    monkeypatch.setenv("CODE_RAG_URL", "http://code-rag.test")
    reset_config()

    mock_get_client = Mock()
    mock_get_client.return_value.post = AsyncMock(return_value=_mock_response({
        "query": "factory",
        "results": [
            {
                "entity": {
                    "name": "make_widget",
                    "entity_type": "function",
                    "file_path": "src/widgets.py",
                    "line_start": 12,
                    "line_end": 18,
                    "language": "python",
                    "parameters": [{"name": "name"}],
                    "docstring": "Create a widget.",
                },
                "score": 0.91,
                "explanation": "function name and docstring match",
                "metadata": {"match": "semantic"},
            }
        ],
        "total_results": 1,
    }))

    request = Mock(spec=Request)
    request.json = AsyncMock(return_value={"query": "factory", "languages": ["python"], "limit": 5})

    with patch.object(search, "get_http_client", mock_get_client):
        result = await search.graphrag_search_code(request)

    mock_get_client.return_value.post.assert_awaited_once()
    post_url = mock_get_client.return_value.post.await_args.args[0]
    kwargs = mock_get_client.return_value.post.await_args.kwargs
    assert post_url == "http://code-rag.test/search"
    assert kwargs["json"]["top_k"] == 5
    assert kwargs["json"]["language"] == "python"
    assert result["results"][0]["file_path"] == "src/widgets.py"
    assert result["results"][0]["code_type"] == "function"
    assert "Create a widget." in result["results"][0]["content"]


def test_upstream_bridge_allowlist_rejects_unknown_paths():
    upstream = load_graphrag_route_module("upstream")

    assert upstream._normalize_upstream_path("/evaluate/retrieval/status") == "evaluate/retrieval/status"

    with pytest.raises(HTTPException) as exc_info:
        upstream._normalize_upstream_path("/openapi.json")

    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException) as clear_exc:
        upstream._normalize_upstream_path("/clear-all")

    assert clear_exc.value.status_code == 403


@pytest.mark.asyncio
async def test_deployment_diagnostics_reports_configured_graph_api():
    upstream = load_graphrag_route_module("upstream")

    result = await upstream.graphrag_deployment_diagnostics()

    assert result["enabled"] is True
    assert result["status"] == "ready"
    assert result["features"]["local_pipeline"]["ready"] is True
    assert result["features"]["graph_api"]["ready"] is True
    assert result["features"]["graph_api"]["url"] == "http://graphrag.test"
    assert "full_helper_stack" in result["commands"]


@pytest.mark.asyncio
async def test_deployment_diagnostics_reports_local_mode_without_external_urls(monkeypatch):
    from modules.graphrag.config import reset_config

    upstream = load_graphrag_route_module("upstream")
    for name in (
        "GRAPHRAG_URL",
        "GRAPHRAG_SERVICE_URL",
        "NER_API_URL",
        "GRAPHRAG_NER_URL",
        "REL_API_URL",
        "GRAPHRAG_REL_URL",
        "CODE_RAG_URL",
        "GRAPHRAG_CODE_RAG_URL",
        "REDDIT_CRAWLER_URL",
        "GRAPHRAG_REDDIT_CRAWLER_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_config()

    result = await upstream.graphrag_deployment_diagnostics()

    assert result["enabled"] is True
    assert result["status"] == "local"
    assert result["features"]["local_pipeline"]["ready"] is True
    assert result["features"]["graph_api"]["ready"] is False
    assert "run-scoped chat fallback" in result["summary"]
    assert any("local pipeline traces" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_deployment_diagnostics_warns_about_docker_localhost(monkeypatch):
    from modules.graphrag.config import reset_config

    upstream = load_graphrag_route_module("upstream")
    monkeypatch.setenv("RUNNING_IN_DOCKER", "true")
    monkeypatch.setenv("GRAPHRAG_URL", "http://localhost:8010")
    reset_config()

    result = await upstream.graphrag_deployment_diagnostics()

    assert result["docker_runtime"] is True
    assert result["status"] == "ready"
    assert any("GRAPHRAG_URL points at localhost inside Docker" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_services_status_reports_reachability_and_deployment_hints(monkeypatch):
    from modules.graphrag.config import reset_config

    monkeypatch.setenv("NER_API_URL", "http://ner.test")
    for name in (
        "REL_API_URL",
        "GRAPHRAG_REL_URL",
        "CODE_RAG_URL",
        "GRAPHRAG_CODE_RAG_URL",
        "REDDIT_CRAWLER_URL",
        "GRAPHRAG_REDDIT_CRAWLER_URL",
        "RUNNING_IN_DOCKER",
        "USE_DOCKER",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_config()

    health = load_graphrag_route_module("health")
    get = AsyncMock(side_effect=[
        _mock_response({"status": "healthy", "service": "graphrag"}),
        _mock_response({"status": "healthy", "model": "gliner"}),
    ])
    mock_client = Mock()
    mock_client.get = get

    with patch.object(health, "get_http_client", return_value=mock_client):
        result = await health.graphrag_services_status()

    assert result["graphrag"]["configured"] is True
    assert result["graphrag"]["available"] is True
    assert result["graphrag"]["ready"] is True
    assert result["graphrag"]["service"]["health_url"] == "http://graphrag.test/health"
    assert result["graphrag"]["service"]["compatible_health_urls"] == [
        "http://graphrag.test/health",
        "http://graphrag.test/api/health",
        "http://graphrag.test/api/v1/health",
    ]
    assert result["ner"]["configured"] is True
    assert result["ner"]["ready"] is True
    assert result["ner"]["required_env"][0] == "NER_API_URL"
    assert result["rel"]["status"] == "not_configured"
    assert result["rel"]["ready"] is False
    assert result["code_rag"]["deploy_command"].startswith("GRAPHRAG_ENABLED=true")
    assert result["_meta"]["status"] == "ready"
    assert result["_meta"]["ready"] is True
    assert result["_meta"]["operational_ready"] is True
    assert result["_meta"]["operating_mode"]["id"] == "remote_graph"
    assert result["_meta"]["operating_mode"]["upload_mode"] == "local_trace_plus_remote_ingest"
    assert result["_meta"]["interaction_contract"]["mode"] == "remote_graph"
    assert result["_meta"]["interaction_contract"]["upload"]["behavior"] == "local_trace_plus_adaptive_remote_ingest"
    assert result["_meta"]["interaction_contract"]["upload"]["remote_submission"] is True
    assert result["_meta"]["interaction_contract"]["upload"]["endpoint_verification"] == "health_preflight_then_adaptive_upload_attempts"
    assert result["_meta"]["interaction_contract"]["upload"]["health_url"] == "http://graphrag.test/health"
    assert result["_meta"]["interaction_contract"]["upload"]["compatible_health_urls"] == [
        "http://graphrag.test/health",
        "http://graphrag.test/api/health",
        "http://graphrag.test/api/v1/health",
    ]
    assert result["_meta"]["interaction_contract"]["upload"]["preflight"] == {
        "method": "GET",
        "url": "http://graphrag.test/health",
        "status": "required_before_remote_submit",
        "compatible_urls": [
            "http://graphrag.test/health",
            "http://graphrag.test/api/health",
            "http://graphrag.test/api/v1/health",
        ],
        "trace_field": "upstream_preflight.sample_outputs.health_url",
    }
    assert result["_meta"]["interaction_contract"]["upload"]["preferred_endpoint"] == "http://graphrag.test/ingest-documents"
    assert result["_meta"]["interaction_contract"]["upload"]["compatible_endpoints"] == [
        "http://graphrag.test/ingest-documents",
        "http://graphrag.test/process-document",
        "http://graphrag.test/add-to-vector-store",
    ]
    assert result["_meta"]["interaction_contract"]["upload"]["attempt_trace_fields"] == [
        "upstream_ingest.sample_outputs.attempted_urls",
        "upstream_ingest.sample_outputs.endpoint_attempts",
        "upstream_ingest.selected_endpoint",
    ]
    upload_smoke_tests = {
        test["id"]: test
        for test in result["_meta"]["interaction_contract"]["upload"]["smoke_tests"]
    }
    assert upload_smoke_tests["remote_health_preflight"]["ready"] is True
    assert upload_smoke_tests["remote_health_preflight"]["command"] == "curl -fsS http://graphrag.test/health"
    assert upload_smoke_tests["remote_health_preflight"]["urls"] == [
        "http://graphrag.test/health",
        "http://graphrag.test/api/health",
        "http://graphrag.test/api/v1/health",
    ]
    assert upload_smoke_tests["adaptive_remote_ingest"]["urls"] == [
        "http://graphrag.test/ingest-documents",
        "http://graphrag.test/process-document",
        "http://graphrag.test/add-to-vector-store",
    ]
    assert upload_smoke_tests["adaptive_remote_ingest"]["trace_fields"] == result["_meta"]["interaction_contract"]["upload"]["attempt_trace_fields"]
    deployment_smoke_tests = {
        test["id"]: test
        for test in result["_meta"]["interaction_contract"]["deployment"]["smoke_tests"]
    }
    assert deployment_smoke_tests["llama_nexus_services_status"]["path"] == "/api/v1/graphrag/services/status"
    assert deployment_smoke_tests["local_pipeline_preview"]["status"] == "local"
    assert deployment_smoke_tests["document_pipeline_contract"]["status"] == "local"
    assert "samples.relationship_highlights" in deployment_smoke_tests["document_pipeline_contract"]["trace_fields"]
    assert "interaction_handoff.graph_anchors" in deployment_smoke_tests["document_pipeline_contract"]["trace_fields"]
    assert "document_profile.evidence_profile.graph_evidence.link_relationship_count" in deployment_smoke_tests["document_pipeline_contract"]["trace_fields"]
    assert deployment_smoke_tests["remote_health_preflight"]["url"] == "http://graphrag.test/health"
    assert deployment_smoke_tests["adaptive_remote_ingest"]["preferred_url"] == "http://graphrag.test/ingest-documents"
    document_pipeline = result["_meta"]["interaction_contract"]["document_pipeline"]
    assert document_pipeline["status"] == "local"
    assert document_pipeline["endpoint"] == "/api/v1/graphrag/ingest/pipeline/preview"
    assert document_pipeline["sample_limit_cap"] == 80
    assert document_pipeline["relationship_sample_mode"] == "raw_prefix_plus_representative_highlights"
    assert {"doc", "ppt", "xls", "docx", "pptx", "xlsx", "zip"}.issubset(set(document_pipeline["supported_formats"]))
    assert "relationship_highlights" in document_pipeline["evidence_outputs"]
    assert document_pipeline["features"]["balanced_relationship_samples"] is True
    assert document_pipeline["features"]["chart_relationships"] is True
    assert document_pipeline["features"]["link_action_relationships"] is True
    assert document_pipeline["features"]["markdown_graph_evidence"] is True
    evidence_capabilities = {capability["id"]: capability for capability in document_pipeline["evidence_capabilities"]}
    assert document_pipeline["evidence_capability_count"] >= 9
    assert {
        "charts",
        "forms",
        "legacy_office",
        "link_actions",
        "markdown_semantics",
        "shapes_connectors",
        "source_artifacts",
        "styled_text",
    }.issubset(evidence_capabilities)
    assert "legacy_office_text" in evidence_capabilities["legacy_office"]["element_types"]
    assert "contains_legacy_office_text" in evidence_capabilities["legacy_office"]["relationship_types"]
    assert "shape_connector" in evidence_capabilities["shapes_connectors"]["element_types"]
    assert "contains_connector" in evidence_capabilities["shapes_connectors"]["relationship_types"]
    assert "links_to" in evidence_capabilities["link_actions"]["relationship_types"]
    assert "md" in evidence_capabilities["link_actions"]["formats"]
    assert "graph_anchor" in evidence_capabilities["link_actions"]["interaction_outputs"]
    assert "retrieval_rehearsal" in evidence_capabilities["link_actions"]["interaction_outputs"]
    assert "code_block" in evidence_capabilities["markdown_semantics"]["element_types"]
    assert "contains_code_block" in evidence_capabilities["markdown_semantics"]["relationship_types"]
    assert "uses_language" in evidence_capabilities["markdown_semantics"]["relationship_types"]
    assert "structured_ingest_export" in evidence_capabilities["markdown_semantics"]["interaction_outputs"]
    assert "emphasizes" in evidence_capabilities["styled_text"]["relationship_types"]
    assert "uses_bullet_structure" in evidence_capabilities["styled_text"]["relationship_types"]
    assert "artifact_manifest" in evidence_capabilities["source_artifacts"]["interaction_outputs"]
    assert result["_meta"]["interaction_contract"]["chat"]["uses_remote_graph"] is True
    assert "remote_graph_api" in result["_meta"]["interaction_contract"]["chat"]["context_sources"]
    assert result["_meta"]["interaction_contract"]["search"]["available"] is True
    assert result["_meta"]["ready_count"] == 2
    assert result["_meta"]["configured_count"] == 2
    assert result["_meta"]["diagnostics_path"] == "/api/v1/graphrag/deployment/diagnostics"
    assert "full_helper_stack" in result["_meta"]["commands"]
    assert result["_meta"]["workflows"]["local_pipeline_preview"]["status"] == "local"
    assert result["_meta"]["workflows"]["pipeline_chat_handoff"]["status"] == "local"
    assert result["_meta"]["workflows"]["pipeline_chat_handoff"]["ready"] is True
    assert result["_meta"]["workflows"]["remote_ingest"]["ready"] is True
    assert result["_meta"]["workflows"]["graph_search"]["ready"] is True
    assert result["_meta"]["workflows"]["extraction"]["ready"] is True
    assert "pipeline_chat_handoff" in result["_meta"]["available_workflows"]
    assert "remote_ingest" in result["_meta"]["available_workflows"]
    pathways = {item["id"]: item for item in result["_meta"]["interaction_pathways"]}
    assert result["_meta"]["recommended_pathway_id"] == "remote_graph_search"
    assert pathways["upload_trace_chat"]["ready"] is True
    assert pathways["upload_trace_chat"]["uses_local_pipeline"] is True
    assert pathways["upload_trace_chat"]["secondary_endpoint"] == "/api/v1/graphrag/chat-context-v2"
    assert "interaction_handoff" in pathways["upload_trace_chat"]["outputs"]
    assert pathways["remote_graph_search"]["recommended"] is True
    assert pathways["remote_graph_search"]["status"] == "ready"
    assert pathways["remote_graph_search"]["uses_remote_graph"] is True
    assert pathways["remote_graph_search"]["endpoint"] == "http://graphrag.test/ingest-documents"
    assert "upstream_ingest.sample_outputs.endpoint_attempts" in pathways["remote_graph_search"]["checks"]
    checklist = {item["id"]: item for item in result["_meta"]["operator_checklist"]}
    assert checklist["enable_graphrag_routes"]["status"] == "ready"
    assert checklist["preview_upload_trace"]["status"] == "local"
    assert checklist["preview_upload_trace"]["ready"] is True
    assert "retrieval rehearsals" in checklist["preview_upload_trace"]["checks"]
    assert checklist["ask_chat_from_trace"]["status"] == "local"
    assert checklist["remote_graph_ingest"]["status"] == "ready"
    assert checklist["remote_graph_ingest"]["endpoint"] == "http://graphrag.test/ingest-documents"
    assert "verification: health_preflight_then_adaptive_upload_attempts" in checklist["remote_graph_ingest"]["checks"]
    assert "health preflight: http://graphrag.test/health" in checklist["remote_graph_ingest"]["checks"]
    assert "upstream_ingest.sample_outputs.endpoint_attempts" in checklist["remote_graph_ingest"]["checks"]
    assert checklist["graph_search_reasoning"]["status"] == "ready"
    quickstart = {item["id"]: item for item in result["_meta"]["deployment_quickstart"]}
    assert quickstart["local_trace_chat"]["status"] == "ready"
    assert quickstart["local_trace_chat"]["command"] == "GRAPHRAG_ENABLED=true docker compose up -d backend-api"
    assert "POST /api/v1/graphrag/chat-context-v2" in quickstart["local_trace_chat"]["checks"]
    assert quickstart["helper_extraction"]["status"] == "ready"
    assert "graphrag-ner" in quickstart["helper_extraction"]["profiles"]
    assert quickstart["remote_graph"]["status"] == "ready"
    assert quickstart["remote_graph"]["endpoint"] == "http://graphrag.test/ingest-documents"
    assert "GRAPHRAG_URL=http://host.docker.internal:18000" in quickstart["remote_graph"]["env_vars"]
    assert quickstart["document_ocr"]["status"] == "disabled"
    assert get.await_count == 2


@pytest.mark.asyncio
async def test_services_status_uses_compatible_health_endpoint_fallback(monkeypatch):
    from modules.graphrag.config import reset_config

    for name in (
        "NER_API_URL",
        "GRAPHRAG_NER_URL",
        "REL_API_URL",
        "GRAPHRAG_REL_URL",
        "CODE_RAG_URL",
        "GRAPHRAG_CODE_RAG_URL",
        "REDDIT_CRAWLER_URL",
        "GRAPHRAG_REDDIT_CRAWLER_URL",
        "RUNNING_IN_DOCKER",
        "USE_DOCKER",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_config()

    health = load_graphrag_route_module("health")
    get = AsyncMock(side_effect=[
        _mock_response({"detail": "not found"}, status_code=404),
        _mock_response({"status": "healthy", "service": "graphrag"}),
    ])
    mock_client = Mock()
    mock_client.get = get

    with patch.object(health, "get_http_client", return_value=mock_client):
        result = await health.graphrag_services_status()

    assert result["graphrag"]["ready"] is True
    assert result["graphrag"]["service"]["health_url"] == "http://graphrag.test/api/health"
    assert result["graphrag"]["service"]["attempted_health_urls"] == [
        "http://graphrag.test/health",
        "http://graphrag.test/api/health",
    ]
    assert result["_meta"]["interaction_contract"]["upload"]["health_url"] == "http://graphrag.test/api/health"
    assert result["_meta"]["interaction_contract"]["upload"]["preflight"]["url"] == "http://graphrag.test/api/health"
    assert get.await_args_list[0].args[0] == "http://graphrag.test/health"
    assert get.await_args_list[1].args[0] == "http://graphrag.test/api/health"


@pytest.mark.asyncio
async def test_services_status_reports_partial_workflow_readiness(monkeypatch):
    from modules.graphrag.config import reset_config

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    monkeypatch.setenv("NER_API_URL", "http://ner.test")
    for name in (
        "REL_API_URL",
        "GRAPHRAG_REL_URL",
        "CODE_RAG_URL",
        "GRAPHRAG_CODE_RAG_URL",
        "REDDIT_CRAWLER_URL",
        "GRAPHRAG_REDDIT_CRAWLER_URL",
        "RUNNING_IN_DOCKER",
        "USE_DOCKER",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_config()

    health = load_graphrag_route_module("health")
    get = AsyncMock(return_value=_mock_response({"status": "healthy", "model": "gliner"}))
    mock_client = Mock()
    mock_client.get = get

    with patch.object(health, "get_http_client", return_value=mock_client):
        result = await health.graphrag_services_status()

    assert result["_meta"]["status"] == "partial"
    assert result["_meta"]["ready"] is False
    assert result["_meta"]["operational_ready"] is True
    assert result["_meta"]["local_ready"] is True
    assert result["_meta"]["operating_mode"]["id"] == "hybrid_helpers"
    assert result["_meta"]["operating_mode"]["chat_mode"] == "local_pipeline_graph_context"
    assert result["_meta"]["workflows"]["local_pipeline_preview"]["ready"] is True
    assert result["_meta"]["workflows"]["pipeline_chat_handoff"]["status"] == "local"
    assert result["_meta"]["workflows"]["pipeline_chat_handoff"]["ready"] is True
    assert result["_meta"]["workflows"]["remote_ingest"]["status"] == "blocked"
    assert result["_meta"]["workflows"]["graph_search"]["status"] == "blocked"
    assert result["_meta"]["workflows"]["extraction"]["status"] == "partial"
    assert "pipeline_chat_handoff" in result["_meta"]["available_workflows"]
    assert "extraction" in result["_meta"]["available_workflows"]
    assert "remote_ingest" in result["_meta"]["blocked_workflows"]
    checklist = {item["id"]: item for item in result["_meta"]["operator_checklist"]}
    assert checklist["preview_upload_trace"]["status"] == "local"
    assert checklist["ask_chat_from_trace"]["status"] == "local"
    assert checklist["remote_graph_ingest"]["status"] == "local"
    assert checklist["graph_search_reasoning"]["status"] == "local"
    assert checklist["graph_search_reasoning"]["action"] == "Search or reason over recent pipeline traces, or Ask Chat from a trace."
    assert checklist["helper_extraction"]["status"] == "partial"
    assert get.await_count == 1


@pytest.mark.asyncio
async def test_services_status_marks_docker_default_helper_sources(monkeypatch):
    from modules.graphrag.config import reset_config

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    for name in (
        "NER_API_URL",
        "GRAPHRAG_NER_URL",
        "REL_API_URL",
        "GRAPHRAG_REL_URL",
        "CODE_RAG_URL",
        "GRAPHRAG_CODE_RAG_URL",
        "REDDIT_CRAWLER_URL",
        "GRAPHRAG_REDDIT_CRAWLER_URL",
        "USE_DOCKER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RUNNING_IN_DOCKER", "true")
    reset_config()

    health = load_graphrag_route_module("health")
    get = AsyncMock(return_value=_mock_response({"status": "healthy"}))
    mock_client = Mock()
    mock_client.get = get

    with patch.object(health, "get_http_client", return_value=mock_client):
        result = await health.graphrag_services_status()

    assert result["ner"]["url"] == "http://graphrag-ner:8001"
    assert result["rel"]["url"] == "http://graphrag-rel:8002"
    assert result["code_rag"]["url"] == "http://graphrag-code:8000"
    assert result["ner"]["url_source"] == "docker_default"
    assert result["ner"]["url_source_label"] == "Docker service default"
    assert result["ner"]["auto_discovered"] is True
    assert result["rel"]["auto_discovered"] is True
    assert result["code_rag"]["auto_discovered"] is True
    assert result["_meta"]["auto_discovered_count"] == 3
    assert result["_meta"]["explicit_url_count"] == 0
    assert result["_meta"]["url_sources"]["ner"]["auto_discovered"] is True
    assert "Docker default helper URL" in result["_meta"]["source_summary"]
    assert result["_meta"]["interaction_contract"]["deployment"]["auto_discovered_service_count"] == 3
    assert result["_meta"]["operating_mode"]["id"] == "hybrid_helpers"
    assert get.await_count == 3


@pytest.mark.asyncio
async def test_services_status_reports_local_workflows_without_external_urls(monkeypatch):
    from modules.graphrag.config import reset_config

    for name in (
        "GRAPHRAG_URL",
        "GRAPHRAG_SERVICE_URL",
        "NER_API_URL",
        "GRAPHRAG_NER_URL",
        "REL_API_URL",
        "GRAPHRAG_REL_URL",
        "CODE_RAG_URL",
        "GRAPHRAG_CODE_RAG_URL",
        "REDDIT_CRAWLER_URL",
        "GRAPHRAG_REDDIT_CRAWLER_URL",
        "RUNNING_IN_DOCKER",
        "USE_DOCKER",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_config()

    health = load_graphrag_route_module("health")
    mock_client = Mock()
    mock_client.get = AsyncMock()

    with patch.object(health, "get_http_client", return_value=mock_client):
        result = await health.graphrag_services_status()

    assert result["_meta"]["status"] == "local"
    assert result["_meta"]["ready"] is False
    assert result["_meta"]["operational_ready"] is True
    assert result["_meta"]["local_ready"] is True
    assert result["_meta"]["operating_mode"]["id"] == "local_pipeline"
    assert result["_meta"]["operating_mode"]["upload_mode"] == "local_trace_only"
    assert result["_meta"]["interaction_contract"]["mode"] == "local_pipeline"
    assert result["_meta"]["interaction_contract"]["upload"]["behavior"] == "local_trace_only"
    assert result["_meta"]["interaction_contract"]["upload"]["remote_submission"] is False
    assert result["_meta"]["interaction_contract"]["upload"]["endpoint_verification"] == "not_configured"
    assert result["_meta"]["interaction_contract"]["upload"]["health_url"] is None
    assert result["_meta"]["interaction_contract"]["upload"]["preflight"]["status"] == "not_configured"
    assert result["_meta"]["interaction_contract"]["upload"]["compatible_endpoints"] == []
    assert "upstream_ingest.sample_outputs.endpoint_attempts" in result["_meta"]["interaction_contract"]["upload"]["attempt_trace_fields"]
    local_smoke_tests = {
        test["id"]: test
        for test in result["_meta"]["interaction_contract"]["deployment"]["smoke_tests"]
    }
    assert local_smoke_tests["llama_nexus_services_status"]["ready"] is True
    assert local_smoke_tests["local_pipeline_preview"]["status"] == "local"
    assert local_smoke_tests["document_pipeline_contract"]["status"] == "local"
    assert "interaction_handoff.graph_anchors" in local_smoke_tests["document_pipeline_contract"]["trace_fields"]
    assert "document_profile.evidence_profile.graph_evidence.link_relationship_count" in local_smoke_tests["document_pipeline_contract"]["trace_fields"]
    assert local_smoke_tests["remote_health_preflight"]["status"] == "not_configured"
    assert "adaptive_remote_ingest" not in local_smoke_tests
    assert result["_meta"]["interaction_contract"]["document_pipeline"]["format_count"] >= 20
    assert result["_meta"]["interaction_contract"]["document_pipeline"]["features"]["local_context_cache"] is True
    assert result["_meta"]["interaction_contract"]["document_pipeline"]["features"]["link_action_relationships"] is True
    assert result["_meta"]["interaction_contract"]["document_pipeline"]["evidence_capability_count"] >= 9
    assert any(
        capability["id"] == "layout_ocr"
        and "image_contains_text" in capability["relationship_types"]
        for capability in result["_meta"]["interaction_contract"]["document_pipeline"]["evidence_capabilities"]
    )
    assert any(
        capability["id"] == "link_actions"
        and "graph_anchor" in capability["interaction_outputs"]
        for capability in result["_meta"]["interaction_contract"]["document_pipeline"]["evidence_capabilities"]
    )
    assert result["_meta"]["interaction_contract"]["chat"]["available"] is True
    assert result["_meta"]["interaction_contract"]["chat"]["uses_remote_graph"] is False
    assert result["_meta"]["interaction_contract"]["search"]["available"] is False
    assert result["_meta"]["interaction_contract"]["search"]["local_pipeline_available"] is True
    assert result["_meta"]["interaction_contract"]["search"]["fallback"] == "local_pipeline_search"
    assert result["_meta"]["interaction_contract"]["search"]["fallback_endpoint"] == "/api/v1/graphrag/search"
    assert result["_meta"]["workflows"]["local_pipeline_preview"]["status"] == "local"
    assert result["_meta"]["workflows"]["pipeline_chat_handoff"]["status"] == "local"
    assert result["_meta"]["workflows"]["remote_ingest"]["status"] == "blocked"
    assert "local mode" in result["_meta"]["summary"]
    assert "local_pipeline_preview" in result["_meta"]["available_workflows"]
    assert "pipeline_chat_handoff" in result["_meta"]["available_workflows"]
    assert "remote_ingest" in result["_meta"]["blocked_workflows"]
    pathways = {item["id"]: item for item in result["_meta"]["interaction_pathways"]}
    assert result["_meta"]["recommended_pathway_id"] == "upload_trace_chat"
    assert pathways["upload_trace_chat"]["recommended"] is True
    assert pathways["upload_trace_chat"]["status"] == "local"
    assert pathways["upload_trace_chat"]["uses_local_pipeline"] is True
    assert pathways["upload_trace_chat"]["uses_remote_graph"] is False
    assert "local_context" in pathways["upload_trace_chat"]["outputs"]
    assert pathways["remote_graph_search"]["status"] == "local"
    assert pathways["remote_graph_search"]["uses_local_pipeline"] is True
    assert pathways["remote_graph_search"]["uses_remote_graph"] is False
    assert pathways["remote_graph_search"]["action"]
    checklist = {item["id"]: item for item in result["_meta"]["operator_checklist"]}
    assert checklist["enable_graphrag_routes"]["status"] == "ready"
    assert checklist["preview_upload_trace"]["status"] == "local"
    assert checklist["ask_chat_from_trace"]["status"] == "local"
    assert checklist["remote_graph_ingest"]["status"] == "local"
    assert checklist["graph_search_reasoning"]["status"] == "local"
    assert checklist["graph_search_reasoning"]["mode"] == "local_pipeline_search"
    quickstart = {item["id"]: item for item in result["_meta"]["deployment_quickstart"]}
    assert quickstart["local_trace_chat"]["status"] == "ready"
    assert quickstart["helper_extraction"]["status"] == "blocked"
    assert quickstart["remote_graph"]["status"] == "local"
    assert quickstart["remote_graph"]["next_step"]
    assert quickstart["document_ocr"]["status"] == "disabled"
    mock_client.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_services_status_reports_document_ocr_runtime(monkeypatch):
    from modules.graphrag.config import reset_config

    for name in (
        "GRAPHRAG_URL",
        "GRAPHRAG_SERVICE_URL",
        "NER_API_URL",
        "GRAPHRAG_NER_URL",
        "REL_API_URL",
        "GRAPHRAG_REL_URL",
        "CODE_RAG_URL",
        "GRAPHRAG_CODE_RAG_URL",
        "REDDIT_CRAWLER_URL",
        "GRAPHRAG_REDDIT_CRAWLER_URL",
        "RUNNING_IN_DOCKER",
        "USE_DOCKER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GRAPHRAG_ENABLE_IMAGE_OCR", "true")
    reset_config()

    health = load_graphrag_route_module("health")
    mock_client = Mock()
    mock_client.get = AsyncMock()

    def fake_find_spec(name):
        return object() if name in {"PIL", "pytesseract"} else None

    with (
        patch.object(health.importlib.util, "find_spec", side_effect=fake_find_spec),
        patch.object(health.shutil, "which", return_value="/usr/bin/tesseract"),
        patch.object(health, "get_http_client", return_value=mock_client),
    ):
        result = await health.graphrag_services_status()
        health_result = await health.graphrag_health()

    ocr_status = result["_meta"]["document_ocr"]
    assert ocr_status["enabled"] is True
    assert ocr_status["ready"] is True
    assert ocr_status["runtime"]["pillow"] is True
    assert ocr_status["runtime"]["pytesseract"] is True
    assert ocr_status["runtime"]["tesseract_binary"] is True
    assert result["_meta"]["interaction_contract"]["deployment"]["document_ocr"]["status"] == "ready"
    smoke_tests = {
        test["id"]: test
        for test in result["_meta"]["interaction_contract"]["deployment"]["smoke_tests"]
    }
    assert smoke_tests["document_ocr_runtime"]["ready"] is True
    assert smoke_tests["document_ocr_runtime"]["status"] == "ready"
    assert health_result["document_ocr_available"] is True
    assert health_result["document_ocr"]["status"] == "ready"
    assert health_result["local_pipeline_available"] is True
    quickstart = {item["id"]: item for item in result["_meta"]["deployment_quickstart"]}
    assert quickstart["document_ocr"]["status"] == "ready"
    assert quickstart["document_ocr"]["next_step"] is None
    assert "tesseract binary" in quickstart["document_ocr"]["checks"]
    mock_client.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_services_status_reports_disabled_workflows_when_graphrag_disabled(monkeypatch):
    from modules.graphrag.config import reset_config

    monkeypatch.setenv("GRAPHRAG_ENABLED", "false")
    for name in (
        "GRAPHRAG_URL",
        "GRAPHRAG_SERVICE_URL",
        "NER_API_URL",
        "GRAPHRAG_NER_URL",
        "REL_API_URL",
        "GRAPHRAG_REL_URL",
        "CODE_RAG_URL",
        "GRAPHRAG_CODE_RAG_URL",
        "REDDIT_CRAWLER_URL",
        "GRAPHRAG_REDDIT_CRAWLER_URL",
        "RUNNING_IN_DOCKER",
        "USE_DOCKER",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_config()

    health = load_graphrag_route_module("health")
    mock_client = Mock()
    mock_client.get = AsyncMock()

    with patch.object(health, "get_http_client", return_value=mock_client):
        result = await health.graphrag_services_status()

    assert result["_meta"]["status"] == "disabled"
    assert result["_meta"]["ready"] is False
    assert result["_meta"]["operational_ready"] is False
    assert result["_meta"]["local_ready"] is False
    assert result["_meta"]["operating_mode"]["id"] == "disabled"
    assert result["_meta"]["interaction_contract"]["mode"] == "disabled"
    assert result["_meta"]["interaction_contract"]["upload"]["behavior"] == "disabled"
    assert result["_meta"]["interaction_contract"]["upload"]["endpoint_verification"] == "disabled"
    assert result["_meta"]["interaction_contract"]["upload"]["preflight"]["status"] == "not_configured"
    assert result["_meta"]["interaction_contract"]["chat"]["available"] is False
    assert result["_meta"]["interaction_contract"]["deployment"]["enabled"] is False
    disabled_smoke_tests = {
        test["id"]: test
        for test in result["_meta"]["interaction_contract"]["deployment"]["smoke_tests"]
    }
    assert disabled_smoke_tests["llama_nexus_services_status"]["status"] == "disabled"
    assert disabled_smoke_tests["local_pipeline_preview"]["status"] == "blocked"
    assert disabled_smoke_tests["document_pipeline_contract"]["status"] == "disabled"
    assert disabled_smoke_tests["remote_health_preflight"]["status"] == "disabled"
    assert result["_meta"]["interaction_contract"]["document_pipeline"]["status"] == "disabled"
    assert result["_meta"]["interaction_contract"]["document_pipeline"]["ready"] is False
    assert result["_meta"]["workflows"]["local_pipeline_preview"]["status"] == "blocked"
    assert result["_meta"]["workflows"]["pipeline_chat_handoff"]["status"] == "blocked"
    assert result["_meta"]["available_workflows"] == []
    pathways = {item["id"]: item for item in result["_meta"]["interaction_pathways"]}
    assert result["_meta"]["recommended_pathway_id"] is None
    assert pathways["upload_trace_chat"]["status"] == "disabled"
    assert pathways["upload_trace_chat"]["ready"] is False
    assert pathways["remote_graph_search"]["status"] == "disabled"
    assert pathways["helper_extraction"]["status"] == "disabled"
    checklist = {item["id"]: item for item in result["_meta"]["operator_checklist"]}
    assert checklist["enable_graphrag_routes"]["status"] == "disabled"
    assert checklist["enable_graphrag_routes"]["action"] == "Set GRAPHRAG_ENABLED=true and restart the backend."
    assert checklist["preview_upload_trace"]["status"] == "blocked"
    assert checklist["ask_chat_from_trace"]["status"] == "blocked"
    quickstart = {item["id"]: item for item in result["_meta"]["deployment_quickstart"]}
    assert quickstart["local_trace_chat"]["status"] == "disabled"
    assert quickstart["local_trace_chat"]["next_step"] == "Set GRAPHRAG_ENABLED=true and restart the backend."
    assert quickstart["helper_extraction"]["status"] == "disabled"
    assert quickstart["remote_graph"]["status"] == "disabled"
    mock_client.get.assert_not_awaited()
