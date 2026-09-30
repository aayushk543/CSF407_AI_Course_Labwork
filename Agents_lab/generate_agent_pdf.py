import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak

def create_agents_pdf():
    pdf_path = r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Agents_lab\agents_lab_report.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    
    # Custom typography styles
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
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=11,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading3'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2d3748'),
        spaceBefore=8,
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
    story.append(Paragraph("Artificial Intelligence Laboratory Report: Goal-Based Agent", title_style))
    story.append(Paragraph("<b>Course:</b> CSF407 / AI Laboratory &nbsp;|&nbsp; <b>Date:</b> September 30, 2026 &nbsp;|&nbsp; <b>Topic:</b> Autonomous Warehouse Navigation", subtitle_style))
    story.append(Spacer(1, 4))

    # Task 1
    story.append(Paragraph("1. Task 1: Understanding the Problem", h1_style))
    story.append(Paragraph("<b>The Environment:</b> Discrete 2D warehouse grid of dimensions 7 rows &times; 21 columns (147 cells). According to Russell & Norvig's taxonomy, the environment is <b>fully observable</b> (all walls and open cells known), <b>static</b> (shelves and targets remain fixed), <b>deterministic</b> (cardinal moves succeed with probability 1.0), <b>discrete</b> (grid coordinates and step actions), and <b>single-agent</b> (autonomous vehicle operates alone).", body_style))
    story.append(Paragraph("<b>Goal of the Agent:</b> Transport packages from loading bay S = (1, 1) to dispatch destination G = (1, 19) along a collision-free path that traverses only free space ('.'), avoids obstacles ('#'), and minimizes total movement actions.", body_style))
    story.append(Paragraph("<b>Available Actions:</b> Movement in four cardinal directions: Up (-1, 0), Down (+1, 0), Left (0, -1), Right (0, +1). Each action shifts the vehicle by one adjacent cell provided the target is in bounds and unblocked.", body_style))
    story.append(Paragraph("<b>Required Information & Internal State:</b> 1) Map of the environment (grid dimensions and obstacle coordinates); 2) Current vehicle state (r, c); 3) Goal location G; 4) Deterministic transition model; 5) Search planning data (frontier queue and predecessor came_from mapping).", body_style))
    story.append(Paragraph("<b>Goal-Based vs Simple Reflex Agent:</b> A <i>simple reflex agent</i> acts purely on current sensory input via condition-action rules ('if blocked, turn'). It has no memory, cannot evaluate future outcomes, and easily becomes trapped in concave obstacles or infinite loops. A <i>goal-based agent</i> explicitly models its destination (G) and deliberates over sequences of future actions using search/planning. It selects actions specifically because they form part of a simulated trajectory that achieves the goal.", body_style))
    
    story.append(Paragraph("<b>Think About It (Scaling to Doubled Warehouse):</b><br/>"
                           "• <i>BFS Suitability:</i> Breadth-First Search time and space complexities are O(b^d). For a modestly doubled grid (14 &times; 42 = 588 cells), BFS remains fast. However, for industrial-scale facilities (tens of thousands of cells), uninformed BFS wastes memory expanding an undirected radial wave in all directions.<br/>"
                           "• <i>Informed Heuristic Search (A*):</i> An informed A* algorithm guided by Manhattan distance h(n) = |r_n - r_G| + |c_n - c_G| prioritizes expansion towards the goal, reducing search space by orders of magnitude while preserving shortest-path optimality.<br/>"
                           "• <i>Dynamic Obstacles:</i> Real warehouses feature moving vehicles and human pickers, requiring dynamic replanning (e.g., D* Lite, MAPF) rather than pure offline planning.", callout_style))

    # Task 2
    story.append(Paragraph("2. Task 2: Designing the Intelligent Agent", h1_style))
    story.append(Paragraph("<b>Design Components:</b><br/>"
                           "• <b>Environment:</b> Discrete 2D grid containing obstacles, open corridors, start S=(1, 1), and goal G=(1, 19).<br/>"
                           "• <b>Current State:</b> Coordinate tuple (r, c).<br/>"
                           "• <b>Goal Formulation:</b> Boolean condition: state == (1, 19).<br/>"
                           "• <b>Available Actions:</b> {Up, Down, Left, Right} with deterministic transition function Result(s, a).<br/>"
                           "• <b>Decision-Making Component:</b> Breadth-First Search (BFS) graph-search procedure simulating forward paths to goal.", body_style))

    # Architecture Diagram Image
    arch_img_path = r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Agents_lab\agent_architecture_diagram.png"
    if os.path.exists(arch_img_path):
        story.append(Image(arch_img_path, width=280, height=170))
        story.append(Spacer(1, 4))

    # Task 3
    story.append(Paragraph("3. Task 3: Prompt Engineering, Implementation, and Evaluation", h1_style))
    story.append(Paragraph("<b>Prompt Provided to LLM:</b> <i>'Write a well-documented Python program implementing a goal-based agent for the warehouse navigation problem shown above. The program should: represent the warehouse as a two-dimensional grid; determine a collision-free path from S to G; avoid all obstacles; print either the path found or a suitable message if no path exists; explain the search algorithm that has been chosen and why it is appropriate.'</i>", body_style))
    story.append(Paragraph("<b>Evaluation Questions:</b>", h2_style))
    story.append(Paragraph("1. <b>Did the LLM generate a working program on the first attempt?</b><br/>"
                           "&bull; <i>Yes.</i> The program executed without syntax or logic errors on the first try. It parsed the grid, located S and G, executed BFS graph search, and correctly reconstructed the 20-move optimal path.", bullet_style))
    story.append(Paragraph("2. <b>If not, how can you improve your prompt?</b><br/>"
                           "&bull; Prompting can be enhanced by specifying formal data structures (e.g. typing annotations), requiring unit test assertions to mathematically verify no obstacle collisions, requesting search telemetry (nodes expanded, queue peak), and asking for comparative benchmarks against A* search.", bullet_style))
    story.append(Paragraph("3. <b>What search algorithm did the LLM choose?</b><br/>"
                           "&bull; The LLM selected <b>Breadth-First Search (BFS)</b> using a FIFO double-ended queue (collections.deque).", bullet_style))
    story.append(Paragraph("4. <b>Why do you think the LLM selected this algorithm?</b><br/>"
                           "&bull; <i>Unweighted Step Cost Optimality:</i> In grid navigation with uniform step costs (c=1), BFS is mathematically guaranteed to find the shallowest goal node, ensuring the shortest possible path.<br/>"
                           "&bull; <i>Simplicity & Robustness:</i> BFS avoids heuristic tuning or priority queue overhead. For a 147-cell grid, BFS is optimal, complete, and terminates in milliseconds.", bullet_style))

    # Path Visualization & Results
    story.append(Paragraph("4. Path Execution & Visual Verification", h1_style))
    story.append(Paragraph("• <b>Start State:</b> (1, 1) &nbsp;|&nbsp; <b>Goal State:</b> (1, 19)<br/>"
                           "• <b>Nodes Expanded:</b> 68 cells &nbsp;|&nbsp; <b>Optimal Path Length:</b> 21 cells (20 moves)<br/>"
                           "• <b>Action Sequence:</b> Right &rarr; Right &rarr; Right &rarr; Down &rarr; Right &rarr; Right &rarr; Right &rarr; Up &rarr; Right &rarr; Right &rarr; Right &rarr; Right &rarr; Right &rarr; Right &rarr; Right &rarr; Right &rarr; Right &rarr; Right &rarr; Right &rarr; Right", body_style))

    path_img_path = r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Agents_lab\warehouse_path_visualization.png"
    if os.path.exists(path_img_path):
        story.append(Image(path_img_path, width=320, height=130))
        story.append(Spacer(1, 4))

    doc.build(story)
    print(f"Successfully generated PDF report at: {pdf_path}")

if __name__ == '__main__':
    create_agents_pdf()
