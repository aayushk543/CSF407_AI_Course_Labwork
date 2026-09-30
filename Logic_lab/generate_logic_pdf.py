import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak

def create_logic_pdf():
    pdf_path = r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Logic_lab\logic_lab_report.pdf"
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
    story.append(Paragraph("Laboratory Report: Logical Reasoning for Planning", title_style))
    story.append(Paragraph("<b>Course:</b> CSF407 / Artificial Intelligence Laboratory &nbsp;|&nbsp; <b>Date:</b> September 30, 2026 &nbsp;|&nbsp; <b>Topic:</b> STRIPS Planning, Search & Prolog Verification", subtitle_style))
    story.append(Spacer(1, 4))

    # Task 0
    story.append(Paragraph("1. Task 0: Understand the Planning Problem", h1_style))
    story.append(Paragraph("A planning problem is represented by the 3-tuple &Pscr; = (I, A, G):<br/>"
                           "• <b>Initial state I:</b> {At(Robot, A), At(Package, A)} &nbsp;|&nbsp; <b>Goal G:</b> {At(Package, C)}.<br/>"
                           "• <b>Actions:</b> Move(X,Y), PickUp(Package,X), Drop(Package,X) with positive/negative preconditions and add/delete effects.<br/>"
                           "• <b>Applicability:</b> PickUp(Package, A) is <b>applicable</b> in I because all its preconditions are present. Drop(Package, C) is <b>inapplicable</b> because the robot is neither at C nor holding the package.", body_style))
    
    story.append(Paragraph("<b>Think About It:</b> An action is applicable if and only if S &models; Preconditions(a). Logic provides the mathematical bridge determining whether an action can physically take place.", callout_style))

    # Task 1: Manual Plan
    story.append(Paragraph("2. Task 1: Manual Plan Construction", h1_style))
    story.append(Paragraph("Optimal 4-step sequence: S0 &rarr; PickUp(P, A) &rarr; Move(A, B) &rarr; Move(B, C) &rarr; Drop(P, C) &rarr; S4.<br/>"
                           "• S0: {At(Robot,A), At(Package,A)}<br/>"
                           "• S1: {At(Robot,A), Holding(Package)}<br/>"
                           "• S2: {At(Robot,B), Holding(Package)}<br/>"
                           "• S3: {At(Robot,C), Holding(Package)}<br/>"
                           "• S4: {At(Robot,C), At(Package,C)} &models; G.", body_style))

    # Trace Image
    trace_img = r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Logic_lab\state_transition_plan_trace.png"
    if os.path.exists(trace_img):
        story.append(Image(trace_img, width=320, height=140))
        story.append(Spacer(1, 4))

    # Task 2 & 3: Implementation & Tests
    story.append(Paragraph("3. Task 2 & 3: Prompt Engineering and Systematic Testing", h1_style))
    story.append(Paragraph("<b>Prompt:</b> <i>'Implement planning agent in Python. Represent state as set of propositions. Action contains pos/neg preconds and pos/neg effects. Applicability: all preconds satisfied. Apply: remove neg effects, add pos effects. Use BFS to find plan achieving goal. Detect when no plan exists.'</i>", body_style))

    table_tests = [
        ["Test Case", "Condition / Modification", "Expected Result", "Observed Result", "Status"],
        ["Test A: Solvable", "Standard warehouse problem", "4-action plan", "4 actions, 6 expanded", "PASS"],
        ["Test B: Impossible", "Removed PickUp action", "Reports 'No plan found'", "None returned, 3 expanded", "PASS"],
        ["Test C: Irrelevant", "Robot moves without package", "Strictly achieves At(Pkg,C)", "Ignored empty moves", "PASS"]
    ]
    t_test = Table(table_tests, colWidths=[90, 130, 115, 115, 50])
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

    # Task 4: Logic + Search
    story.append(Paragraph("4. Task 4: Logic + Search = Planning", h1_style))
    story.append(Paragraph("• <b>Logical Reasoning:</b> Operates locally on states and actions, evaluating entailment S &models; Preconditions(a) and computing successor states S' = (S \\ Neg) &cup; Pos. Logic determines <b>what is possible</b>.<br/>"
                           "• <b>State-Space Search (BFS):</b> Operates globally, exploring candidate action sequences to reach the goal. Search determines <b>what to try</b>.", body_style))

    arch_img = r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Logic_lab\planning_architecture_diagram.png"
    if os.path.exists(arch_img):
        story.append(Image(arch_img, width=280, height=190))
        story.append(Spacer(1, 4))

    # Task 5: Self-Verification
    story.append(Paragraph("5. Task 5: LLM Explanation vs Independent Verification", h1_style))
    story.append(Paragraph("An LLM produces persuasive textual explanations, but lacks an internal proof engine and may hallucinate satisfied preconditions. <b>Independent state transitions must always be trusted over LLM explanations.</b> <i>A generated explanation is not an independent verification.</i>", body_style))

    # Tasks 6, 7, 8 & Reflection: Prolog
    story.append(Paragraph("6. Tasks 6, 7 & 8: Prolog as an Independent Rule-Based Verifier", h1_style))
    story.append(Paragraph("In planner.pl, the warehouse topology is encoded as Horn clauses:<br/>"
                           "• <b>Facts:</b> connected(a,b). connected(b,a). connected(b,c). connected(c,b).<br/>"
                           "• <b>Rules:</b> can_move(X,Y) :- connected(X,Y). valid_move(X,Y) :- connected(X,Y).<br/>"
                           "• <b>Queries:</b> ?- valid_move(a,b) &rarr; <b>true.</b> | ?- valid_move(a,c) &rarr; <b>false.</b> (Catches invalid direct leap!)<br/>"
                           "• <b>Task 8 Deduction Chain:</b> wet_road &rArr; (wet_road &rarr; slippery) &rArr; (slippery &rarr; reduce_speed) &rArr; <b>reduce_speed.</b>", body_style))

    story.append(Paragraph("7. Key Reflection Answers", h1_style))
    story.append(Paragraph("• <b>Preconditions/Effects:</b> Prevent the LLM from inventing illegal transitions or teleporting packages.", bullet_style))
    story.append(Paragraph("• <b>Unchecked Preconditions:</b> Would allow the robot to drop the package at C without picking it up or traveling there.", bullet_style))
    story.append(Paragraph("• <b>Logical Reasoning Role:</b> Evaluates precondition satisfaction, generates state mutations, and runs Prolog inference.", bullet_style))
    story.append(Paragraph("• <b>Relation to Search:</b> Planning searches a graph where states are sets of logical propositions and transitions are logical deduction operators.", bullet_style))

    doc.build(story)
    print(f"Successfully generated PDF report at: {pdf_path}")

if __name__ == '__main__':
    create_logic_pdf()
