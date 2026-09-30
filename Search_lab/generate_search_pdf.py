import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak

def create_search_pdf():
    pdf_path = r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Search_lab\search_lab_report.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#4a5568'),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontSize=11.5,
        leading=15.5,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading3'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2d3748'),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1a202c'),
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1a202c'),
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2.5
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor('#2c5282'),
        backColor=colors.HexColor('#ebf8ff'),
        borderColor=colors.HexColor('#bee3f8'),
        borderWidth=1,
        borderPadding=5,
        spaceBefore=4,
        spaceAfter=6,
        keepWithNext=True
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Artificial Intelligence Laboratory Report: Search and A*", title_style))
    story.append(Paragraph("<b>Course:</b> CSF407 / Artificial Intelligence Laboratory &nbsp;|&nbsp; <b>Date:</b> September 30, 2026 &nbsp;|&nbsp; <b>Topic:</b> Heuristic Search & Algorithm Validation", subtitle_style))
    story.append(Spacer(1, 4))

    # Task 0
    story.append(Paragraph("1. Task 0: Understand the Search Problem", h1_style))
    story.append(Paragraph("A search problem is formulated as the 6-tuple &Pscr; = (S, A, T, s0, G, c):<br/>"
                           "• <b>State S:</b> (r, c) &isin; {0..8} &times; {0..16} representing valid free cells.<br/>"
                           "• <b>Actions A:</b> {Up, Down, Left, Right} with step cost c=1.<br/>"
                           "• <b>Transition T:</b> Deterministic orthogonal movement T(s, a) = s + &Delta;a.<br/>"
                           "• <b>Initial state s0:</b> (1, 1) &nbsp;|&nbsp; <b>Goal G:</b> {(7, 15)}.", body_style))
    story.append(Paragraph("• <i>State specification:</i> Coordinates (r, c) fully determine legal actions and future path costs.<br/>"
                           "• <i>Invalid actions:</i> Moving out-of-bounds or into obstacle blocks ('#').<br/>"
                           "• <i>Determinism & Solution:</i> Problem is deterministic (probability 1.0). A solution is a sequence of actions from s0 reaching G.", bullet_style))

    # Task 1 & 2
    story.append(Paragraph("2. Task 1 & 2: Agent Design and Prompt Engineering", h1_style))
    story.append(Paragraph("<b>Design:</b> Frontier priority queue ordered by f(n) = g(n) + h(n), closed-set hash set, came_from dictionary, and Manhattan heuristic h(n) = |x - xG| + |y - yG|.<br/>"
                           "<b>LLM Prompt:</b> <i>'Implement A* search for warehouse grid with obstacles '#', free '.', start S, goal G, cardinal moves cost 1, Manhattan heuristic. Maintain frontier, calculate g/h/f, avoid duplicate expansion, reconstruct path, report path length and states expanded.'</i>", body_style))

    # Task 3
    story.append(Paragraph("3. Task 3: Systematic Testing of A* Implementation", h1_style))
    table_test = [
        ["Test Case", "Scenario / Condition", "Expected Result", "Observed Result", "Status"],
        ["Test 1: Original", "Full 9x17 serpentine maze", "Optimal collision-free path", "40 moves, 64 expanded", "PASS"],
        ["Test 2: Trivial", "Adjacent goal #SG##", "Immediate 1-step reach", "1 move, 2 expanded", "PASS"],
        ["Test 3: No Solution", "Goal blocked by obstacles", "Terminates reporting failure", "found=False, 9 expanded", "PASS"],
        ["Test 4: Alternative", "Direct (len 6) vs Detour (len 10)", "Selects shortest path", "6 moves, 7 expanded", "PASS"]
    ]
    t_test = Table(table_test, colWidths=[85, 125, 115, 115, 50])
    t_test.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7fafc')])
    ]))
    story.append(t_test)
    story.append(Spacer(1, 4))

    # Path Image
    path_img = r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Search_lab\search_warehouse_path.png"
    if os.path.exists(path_img):
        story.append(Image(path_img, width=280, height=140))
        story.append(Spacer(1, 4))

    # Task 4 & 5
    story.append(Paragraph("4. Task 4 & 5: Algorithm Inspection and BFS vs A* Comparison", h1_style))
    story.append(Paragraph("<b>Code Mechanics:</b> Frontier uses heapq storing (f, counter, state). Next state popped via heappop(). Duplicate exploration prevented by closed_set and g_score relaxation checks.", body_style))
    
    table_comp = [
        ["Measure", "Original Serpentine Map (BFS / A*)", "Open Warehouse Map (BFS)", "Open Warehouse Map (A*)"],
        ["Solution Found", "True / True", "True", "True"],
        ["Path Length", "40 / 40", "20", "20 (Optimal)"],
        ["States Expanded", "64 / 64 (All Free Cells)", "59", "23 (>60% Reduction)"]
    ]
    t_comp = Table(table_comp, colWidths=[110, 160, 110, 110])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7fafc')])
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Insight:</b> On the serpentine map, exactly 64 traversable cells exist in a single corridor; thus BFS and A* both expand all 64 cells. In open environments, A* prunes unpromising branches, expanding over 60% fewer states than BFS.", body_style))

    # Comparison Plot
    comp_img = r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Search_lab\search_bfs_vs_astar_comparison.png"
    if os.path.exists(comp_img):
        story.append(Image(comp_img, width=280, height=140))
        story.append(Spacer(1, 4))

    # Task 6
    story.append(Paragraph("5. Task 6: Heuristic Investigations and Admissibility", h1_style))
    story.append(Paragraph("• <b>Manhattan Distance:</b> Admissible (h &le; h*) and consistent on 4-way grids, guaranteeing optimality.<br/>"
                           "• <b>h(n) = 0:</b> Admissible, but reduces A* to blind Dijkstra/BFS expanding symmetrically.<br/>"
                           "• <b>Euclidean Distance:</b> Admissible, but dominated by Manhattan distance on grids because diagonal moves are prohibited.<br/>"
                           "• <b>2 &times; Manhattan:</b> Inadmissible (h > h*). Acts as weighted A*, driving aggressively toward goal but risking suboptimality on multi-path topologies.", body_style))

    # Task 7 & Final Reflection
    story.append(Paragraph("6. Task 7 & Final Reflection: LLM as an Engineering Tool", h1_style))
    story.append(Paragraph("<b>Q1 (Prior Formulation):</b> Mathematical specification defines state spaces, transitions, and goals independently of code, preventing implementation of unintended problems.", bullet_style))
    story.append(Paragraph("<b>Q2 (Informed Nature of A*):</b> A* uses heuristic domain knowledge h(n) to evaluate promising future paths, combining actual cost g(n) with estimated remaining cost h(n).", bullet_style))
    story.append(Paragraph("<b>Q3 (Role of Heuristic):</b> Dictates pruning power and optimality. Admissible heuristics guarantee shortest paths; tighter heuristics expand fewer states.", bullet_style))
    story.append(Paragraph("<b>Q4 (LLM Contribution):</b> Rapidly generated clean, syntactically correct boilerplate code and data structures in seconds, accelerating prototyping.", bullet_style))
    story.append(Paragraph("<b>Q5 (Risks of Untested LLM Code):</b> Untested code may harbor subtle failure modes: tie-breaking crashes, infinite loops on unreachable goals, or heuristic admissibility violations. Human verification is non-negotiable.", bullet_style))

    doc.build(story)
    print(f"Successfully generated PDF report at: {pdf_path}")

if __name__ == '__main__':
    create_search_pdf()
