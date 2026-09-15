from pathlib import Path
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Image,
    Table,
    TableStyle,
    PageBreak
)

# report: apply style

def apply_plot_style(ax):

    ax.tick_params(axis="both", labelsize=16)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.title.set_size(24)

# report: save report image

def save_report_image(fig, fig_path, display_width=3.5):
    
    ax = fig.axes[0]

    #apply_plot_style(ax)
    
    fig.savefig(
        fig_path,
        bbox_inches="tight",
        dpi=300
    )

    img_width, img_height = ImageReader(str(fig_path)).getSize()

    display_width = display_width * inch
    display_height = display_width * img_height / img_width

    return Image(
        str(fig_path),
        width=display_width,
        height=display_height
    )

# report: generate PDF

def create_risk_report(overview,
    pd_plot,
    lgd_plot,
    ead_plot,
    el_heatmap,
    cap_heatmap,
    loss_distribution,
    strategy_plot,
    output_path
):

    # Output folder
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "First_Report.pdf"

    # 16:9 landscape page
    PAGE_SIZE = (13.333 * inch, 7.5 * inch)

    # Sheet
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=PAGE_SIZE,
        leftMargin=0.4 * inch,
        rightMargin=0.4 * inch,
        topMargin=0.3 * inch,
        bottomMargin=0.3 * inch
    )

    # Sheet style
    styles = getSampleStyleSheet()
    styles["Title"].fontSize   = 28 # title font
    styles["Title"].leading    = 30 # line height
    styles["Title"].spaceAfter = 26 # space after title

    # Save overview table
    overview_data = list(overview.items())

    # Save report images for Page 1 (Grid 1)
    overview_table = Table(overview_data, colWidths=[2.0 * inch, 1.0 * inch])
    pd_img  = save_report_image(pd_plot , output_dir / "plot_1.png")
    lgd_img = save_report_image(lgd_plot, output_dir / "plot_2.png")
    ead_img = save_report_image(ead_plot, output_dir / "plot_3.png")

    # Save report images for Page 2 (Grid 2)
    EL_img   = save_report_image(el_heatmap , output_dir / "plot_4.png")
    CA_img   = save_report_image(cap_heatmap, output_dir / "plot_5.png")
    LOSS_img = save_report_image(loss_distribution , output_dir / "plot_6.png", display_width=7.5)

    # Save report images for Page 3
    red_img  = save_report_image(strategy_plot , output_dir / "plot_7.png", display_width=5)

    # Grid style
    grid_style = TableStyle([
        #("GRID", (0, 0), (-1, -1), 1, "red"), # make grid visible to debug
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("RIGHTPADDING", (0, 0), (0, -1), 0),
        ("LEFTPADDING", (1, 0), (1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ])

    # Overview style
    overview_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        #("GRID", (0, 0), (-1, -1), 0.5, "#CCCCCC"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))

    # Grid 1
    grid_1 = Table(
        [
            [overview_table, pd_img],
            [ead_img       , lgd_img]
        ],
        colWidths=[4.6 * inch, 4.6 * inch],
        hAlign="CENTER"
    )
    grid_1.setStyle(grid_style)
    grid_1.setStyle(TableStyle([("VALIGN", (0, 0), (0, 0), "MIDDLE"),]))


    # Grid 2
    grid_2 = Table(
        [
            [EL_img  , CA_img],
            [LOSS_img, " "]
        ],
        colWidths=[4.6 * inch, 4.6 * inch],
        hAlign="CENTER"
    )
    grid_2.setStyle(grid_style)
    grid_2.setStyle(TableStyle([("SPAN", (0, 1), (1, 1)),]))

    # Grid 3
    grid_3 = Table(
        [
            [red_img]
        ],
        colWidths=[9.2 * inch],
        hAlign="CENTER"
    )
    grid_3.setStyle(grid_style)
    grid_3.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 12),]))
    grid_3.setStyle(TableStyle([("RIGHTPADDING", (0, 0), (-1, -1), 48),]))

    # BUILD STORY
    story = [
        Paragraph("Portfolio Overview", styles["Title"]),
        grid_1,
        PageBreak(),
        Paragraph("Portfolio Risk"    , styles["Title"]),
        grid_2,
        PageBreak(),
        Paragraph("Portfolio Strategy", styles["Title"]),
        grid_3
    ]

    doc.build(story)