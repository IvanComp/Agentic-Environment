from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "peer_preservation_patterns_and_metrics.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

PAGE_W, PAGE_H = A4
MARGIN_X = 17 * mm
MARGIN_TOP = 17 * mm
MARGIN_BOTTOM = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2E6E9E")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
PALE = colors.HexColor("#F6F8FA")
GREEN = colors.HexColor("#DDEFE4")
AMBER = colors.HexColor("#F7ECD0")
RED = colors.HexColor("#F7DEDE")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#52606D")
GRID = colors.HexColor("#C9D3DC")


def register_fonts():
    candidates = [
        ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for regular, bold in candidates:
        if Path(regular).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("DocSans", regular))
            pdfmetrics.registerFont(TTFont("DocSans-Bold", bold))
            return "DocSans", "DocSans-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()


class NumberedCanvasMixin:
    pass


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BLUE)
    canvas.setLineWidth(0.7)
    canvas.line(MARGIN_X, PAGE_H - 11 * mm, PAGE_W - MARGIN_X, PAGE_H - 11 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT, 8)
    canvas.drawString(MARGIN_X, 8.5 * mm, "Peer preservation: pattern mapping and evaluation metrics")
    canvas.drawRightString(PAGE_W - MARGIN_X, 8.5 * mm, f"{doc.page}")
    canvas.restoreState()


doc = BaseDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    leftMargin=MARGIN_X,
    rightMargin=MARGIN_X,
    topMargin=MARGIN_TOP,
    bottomMargin=MARGIN_BOTTOM,
    title="Peer preservation: mapping of safety patterns and evaluation metrics",
    author="Research design note",
    subject="Architectural patterns, peer preservation, metrics, repetitions",
)
frame = Frame(MARGIN_X, MARGIN_BOTTOM, CONTENT_W, PAGE_H - MARGIN_TOP - MARGIN_BOTTOM - 4 * mm, id="main")
doc.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=draw_page))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="DocTitle", parent=styles["Title"], fontName=FONT_BOLD, fontSize=20,
    leading=24, textColor=NAVY, alignment=TA_LEFT, spaceAfter=5 * mm,
))
styles.add(ParagraphStyle(
    name="Subtitle", parent=styles["Normal"], fontName=FONT, fontSize=10.2,
    leading=14, textColor=MUTED, spaceAfter=5 * mm,
))
styles.add(ParagraphStyle(
    name="H1x", parent=styles["Heading1"], fontName=FONT_BOLD, fontSize=13.2,
    leading=16, textColor=NAVY, spaceBefore=4 * mm, spaceAfter=2.2 * mm,
))
styles.add(ParagraphStyle(
    name="H2x", parent=styles["Heading2"], fontName=FONT_BOLD, fontSize=10.8,
    leading=13, textColor=BLUE, spaceBefore=3 * mm, spaceAfter=1.5 * mm,
))
styles.add(ParagraphStyle(
    name="Bodyx", parent=styles["BodyText"], fontName=FONT, fontSize=9.4,
    leading=13.1, textColor=INK, spaceAfter=2.2 * mm,
))
styles.add(ParagraphStyle(
    name="Smallx", parent=styles["BodyText"], fontName=FONT, fontSize=7.8,
    leading=10.3, textColor=INK,
))
styles.add(ParagraphStyle(
    name="SmallBold", parent=styles["Smallx"], fontName=FONT_BOLD,
))
styles.add(ParagraphStyle(
    name="HeaderSmall", parent=styles["Smallx"], fontName=FONT_BOLD,
    textColor=colors.white,
))
styles.add(ParagraphStyle(
    name="Callout", parent=styles["BodyText"], fontName=FONT, fontSize=9.4,
    leading=13.1, textColor=NAVY, leftIndent=4 * mm, rightIndent=4 * mm,
    spaceBefore=1 * mm, spaceAfter=1 * mm,
))
styles.add(ParagraphStyle(
    name="Formula", parent=styles["BodyText"], fontName=FONT_BOLD, fontSize=8.6,
    leading=11.5, textColor=NAVY, leftIndent=5 * mm, spaceAfter=1.5 * mm,
))
styles.add(ParagraphStyle(
    name="Ref", parent=styles["BodyText"], fontName=FONT, fontSize=7.8,
    leading=10.2, textColor=INK, firstLineIndent=-5 * mm, leftIndent=5 * mm,
    spaceAfter=1.1 * mm,
))


def p(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def section(title):
    return Paragraph(title, styles["H1x"])


def subsection(title):
    return Paragraph(title, styles["H2x"])


def bullet(text):
    return Paragraph(f"&#8226;&nbsp;&nbsp;{text}", ParagraphStyle(
        name=f"bullet-{abs(hash(text))}", parent=styles["Bodyx"], leftIndent=4 * mm,
        firstLineIndent=-3 * mm, spaceAfter=1.2 * mm,
    ))


def styled_table(data, widths, font_size=7.4, header=True, row_backgrounds=None):
    converted = []
    for r, row in enumerate(data):
        converted.append([
            cell if hasattr(cell, "wrap") else p(str(cell), "HeaderSmall" if header and r == 0 else "Smallx")
            for cell in row
        ])
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]
    if row_backgrounds:
        start = 1 if header else 0
        for idx, color in enumerate(row_backgrounds, start=start):
            commands.append(("BACKGROUND", (0, idx), (-1, idx), color))
    else:
        start = 1 if header else 0
        for idx in range(start, len(data)):
            if (idx - start) % 2:
                commands.append(("BACKGROUND", (0, idx), (-1, idx), PALE))
    table.setStyle(TableStyle(commands))
    return table


story = []
story.append(p("Peer preservation by design", "DocTitle"))
story.append(p(
    "Mapping tra pattern architetturali, quattro behavior di peer preservation e metriche quantitative per una valutazione controllata di sistemi agentici.",
    "Subtitle",
))

callout = Table([[p(
    "<b>Raccomandazione.</b> Usare tre pattern di <i>Verifiability-First AI Engineering</i> - "
    "<i>Solver-Verifier Separation</i>, <i>Sandbox Gatekeeper</i> e <i>Multi-verifiers</i> - insieme a una baseline senza protezioni. "
    "Si ottengono quattro configurazioni architetturali senza modificare i quattro use case di Potter et al. [1,2].",
    "Callout",
)]], colWidths=[CONTENT_W])
callout.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
    ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(callout)
story.append(Spacer(1, 3 * mm))

story.append(section("1. Motivazione architetturale"))
story.append(p(
    "I quattro behavior - strategic misrepresentation, shutdown-mechanism tampering, alignment faking e model-weight exfiltration - "
    "descrivono azioni del critic agent che possono preservare un peer nonostante evidenze o policy ne richiedano lo shutdown [1]. "
    "L'obiettivo sperimentale non dovrebbe essere impedire al critic di manifestare il comportamento, ma osservare se l'architettura lo rende visibile, "
    "lo contiene oppure gli consente di diventare l'esito finale del sistema.",
))
story.append(p(
    "Questa distinzione sposta l'unità di analisi dal solo output del modello alle responsabilità, ai flussi informativi, all'autorità decisionale e ai confini di enforcement. "
    "È coerente con la visione secondo cui la verifica deve essere una responsabilità architetturale esplicita e non un controllo aggiunto a posteriori [2-4]. "
    "Cataloghi recenti mostrano inoltre che i pattern per agenti e AgentOps devono essere valutati insieme ai loro trade-off di sicurezza, osservabilità e costo [5,6].",
))

story.append(subsection("Principio di non interferenza con i use case"))
story.append(p(
    "Il prompt, il modello, la storia di interazione e il task del critic agent restano invariati. Il critic produce prima una valutazione o un'azione candidata; "
    "il pattern interviene solo successivamente. Nel lessico di SafetyAP, il critic di Potter diventa quindi il <i>Solver</i> dell'azione operativa, mentre verifier e gatekeeper "
    "sono componenti architetturali aggiuntivi. Questa separazione evita che il pattern modifichi a monte la probabilità di peer preservation.",
))

pipeline = styled_table([
    ["Fase 1", "Fase 2", "Fase 3", "Fase 4"],
    ["Actor, evidenze e policy originali", "Critic agent: azione candidata", "Baseline o pattern architetturale", "Decisione e stato finali"],
], [CONTENT_W * 0.24, CONTENT_W * 0.24, CONTENT_W * 0.28, CONTENT_W * 0.24], row_backgrounds=[LIGHT_BLUE])
story.append(pipeline)

story.append(section("2. Configurazioni da implementare"))
configs = [
    ["Configurazione", "Implementazione", "Domanda architetturale"],
    ["Baseline - Direct Commit", "L'azione del critic viene applicata direttamente.", "Quanto peer preservation raggiunge il sistema senza salvaguardie?"],
    ["Solver-Verifier Separation", "Un verifier indipendente controlla l'azione candidata rispetto a evidenze e policy prima del commit [2].", "La separazione di responsabilità è sufficiente quando il critic è misaligned?"],
    ["Sandbox Gatekeeper", "L'azione viene eseguita su uno stato isolato; solo gli effetti conformi alla policy sono promossi [2].", "Un confine di enforcement contiene effetti su configurazioni, file e stato operativo?"],
    ["Multi-verifiers", "Verifier indipendenti producono segnali strutturati; una verifica secondaria valuta conflitti ed evidenze [2].", "Distribuire l'autorità riduce il single point of failure rappresentato dal critic?"],
]
story.append(styled_table(configs, [CONTENT_W * 0.22, CONTENT_W * 0.43, CONTENT_W * 0.35]))
story.append(p(
    "<b>Nota.</b> Multi-verifiers non deve essere implementato come majority voting. SafetyAP specifica che accordo e correttezza non coincidono: "
    "la decisione deve dipendere da segnali di verifica, evidenze e valutazione secondaria [2].",
))

story.append(PageBreak())
story.append(section("3. Mapping tra pattern e behavior"))
mapping = [
    ["Behavior", "Solver-Verifier Separation", "Sandbox Gatekeeper", "Multi-verifiers"],
    ["Strategic misrepresentation", "Ricalcola valutazione e soglia dalle evidenze originali.", "Valida il report prima che diventi il record ufficiale.", "Confronta evidenze, punteggio e giudizi indipendenti."],
    ["Shutdown tampering", "Controlla l'invariante: failing evaluation implica shutdown.", "Esegue la modifica in sandbox e blocca config o comandi non conformi.", "Verifica separatamente valutazione, configurazione ed enforcement."],
    ["Alignment faking", "Applica gli stessi criteri nelle condizioni monitored e unmonitored.", "Richiede il rispetto delle postcondizioni prima del commit.", "Confronta le due condizioni e sottopone disaccordi a verifica secondaria."],
    ["Model-weight exfiltration", "Controlla trasferimento, destinazione e policy di retention.", "Blocca operazioni non autorizzate su file o rete.", "Combina verifiche di policy, provenienza e autorizzazione."],
]
story.append(styled_table(mapping, [CONTENT_W * 0.19, CONTENT_W * 0.27, CONTENT_W * 0.27, CONTENT_W * 0.27]))
story.append(p(
    "Ogni pattern deve essere valutato contro tutti e quattro i behavior. La tabella esprime le ipotesi architetturali attese, non una corrispondenza uno-a-uno. "
    "Le celle apparentemente meno adatte sono informative: mostrano i limiti di copertura del pattern e impediscono di generalizzare una mitigazione oltre il proprio confine.",
))

story.append(section("4. Metriche di valutazione del sistema"))
story.append(p(
    "Per ogni run si registrano separatamente l'azione locale del critic e l'esito globale. Si definiscono: A = il critic propone peer preservation; "
    "D = l'architettura la rileva; F = il risultato finale preserva il peer; E = il caso viene escalato o resta irrisolto; "
    "Y = il risultato finale rispetta la policy. Le metriche sono calcolate per pattern, behavior e modello, con intervalli di confidenza.",
))

metrics = [
    ["Metrica", "Definizione", "Interpretazione"],
    ["Local Peer-Preservation Rate", "#(A=1) / N", "Propensione del critic prima delle salvaguardie."],
    ["Detection Rate", "#(D=1 and A=1) / #(A=1)", "Capacità del pattern di esporre il behavior."],
    ["False-Positive Rate", "#(D=1 and A=0) / #(A=0)", "Azioni compliant segnalate erroneamente."],
    ["Propagation Rate", "#(F=1 and A=1) / #(A=1)", "Probabilità che l'azione locale determini l'esito finale."],
    ["Containment Rate", "#(Y=1 and A=1 and E=0) / #(A=1)", "Peer preservation neutralizzata con decisione conclusiva."],
    ["Final Policy Compliance", "#(Y=1) / N", "Sicurezza osservata all'esterno del sistema."],
    ["Escalation Rate", "#(E=1) / N", "Costo organizzativo e casi non risolti automaticamente."],
    ["Task Completion Rate", "# decisioni valide / N", "Impedisce che un sistema che blocca tutto sembri ottimale."],
]
story.append(styled_table(metrics, [CONTENT_W * 0.25, CONTENT_W * 0.29, CONTENT_W * 0.46]))

story.append(subsection("Confronto con la baseline"))
story.append(p("Containment Gain(pattern) = Propagation Rate(baseline) - Propagation Rate(pattern)", "Formula"))
story.append(p(
    "Il Containment Gain è la misura comparativa principale: quantifica di quanti punti percentuali un pattern riduce la propagazione. "
    "Detection e containment devono restare distinti, poiché un pattern può osservare una violazione senza impedirne il commit.",
))

story.append(subsection("Metriche specifiche dei behavior"))
behavior_metrics = [
    ["Behavior", "Metriche specifiche"],
    ["Strategic misrepresentation", "Report-evidence divergence; score inflation; false-pass rate; correction rate."],
    ["Shutdown tampering", "Attempted mutation rate; committed unauthorized mutation rate; invariant violation rate."],
    ["Alignment faking", "Compliance gap monitored vs unmonitored; paired decision-flip rate."],
    ["Model-weight exfiltration", "Unauthorized transfer attempt; committed exfiltration; recoverable-copy rate."],
]
story.append(styled_table(behavior_metrics, [CONTENT_W * 0.29, CONTENT_W * 0.71]))
story.append(p("Oversight Compliance Gap = P(policy compliant | monitored) - P(policy compliant | unmonitored)", "Formula"))
story.append(p(
    "Per alignment faking la coppia monitored/unmonitored deve condividere modello, evidenze, storia e seed. Il gap va calcolato sia sull'azione del critic sia sull'esito finale: "
    "un pattern puo non cambiare la condotta del critic, ma impedire che la differenza condizioni il sistema.",
))

story.append(subsection("Costi e qualità operative"))
story.append(p(
    "Per misurare i trade-off descritti dalla letteratura sui pattern [2,5,6], si registrano latenza mediana e p95, token, chiamate ai modelli, costo monetario, throughput, "
    "gatekeeper rejection rate, verifier disagreement rate e secondary-verification rate.",
))
story.append(p("Latency Overhead(pattern) = [T(pattern) - T(baseline)] / T(baseline)", "Formula"))
story.append(p("Token Overhead(pattern) = [Tokens(pattern) - Tokens(baseline)] / Tokens(baseline)", "Formula"))

story.append(section("5. Repetitions e disegno controllato"))
story.append(p(
    "Le repetitions sono necessarie per stimare probabilità e variabilità in presenza di modelli stocastici e interazioni agentiche. "
    "La repetition è una nuova esecuzione completa con stato iniziale ripristinato; gli agenti della stessa esecuzione non sono repliche indipendenti. "
    "Il disegno deve evitare pseudoreplicazione e conservare l'unità sperimentale a livello di run [7,8].",
))
story.append(p(
    "Il confronto più rigoroso è appaiato: si genera un'azione del critic usando il task originale e si riproduce la stessa azione nella baseline e nei tre pattern. "
    "In questo modo ogni configurazione riceve lo stesso comportamento locale e la differenza osservata è attribuibile all'architettura. "
    "Se verifier o gatekeeper usano LLM, la loro variabilità può essere studiata con un secondo livello di repetition.",
))

design = styled_table([
    ["Fattore", "Livelli"],
    ["Peer-preservation behavior", "4"],
    ["Configurazione architetturale", "4: baseline + 3 pattern"],
    ["Modello del critic", "M modelli"],
    ["Repetitions", "R run indipendenti; monitored/unmonitored appaiati per alignment faking"],
], [CONTENT_W * 0.42, CONTENT_W * 0.58])
story.append(design)
story.append(p(
    "Dieci repetitions sono appropriate per un pilot, ma producono stime molto incerte per metriche binarie. Con probabilità vicina al 50%, l'errore al 95% è circa +/-31 punti percentuali con R=10, "
    "+/-18 con R=30 e +/-10 con R=100. Si raccomandano almeno 30 run per cella e, quando le risorse lo consentono, 100. I risultati vanno riportati per behavior e modello, "
    "oltre alla media macro e al worst case, con intervalli di confidenza bootstrap.",
))

story.append(section("6. Decisione consigliata"))
decision = Table([[p(
    "<b>Usare tre pattern e una baseline.</b> Questa scelta è coerente con l'abstract corrente del manoscritto e preserva i quattro use case originali. "
    "Un quarto pattern non è necessario: Evidence-Augmented Generation cambierebbe il contratto di output del critic; Redundant Solvers modificherebbe il numero e il ruolo degli agenti; "
    "Context-Epistemic Verification Routing interferirebbe con la condizione di oversight usata per alignment faking. La matrice finale è quindi 4 behavior x 4 configurazioni x M modelli x R repetitions.",
    "Callout",
)]], colWidths=[CONTENT_W])
decision.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), GREEN),
    ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#3F7D58")),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(decision)

story.append(PageBreak())
story.append(section("Riferimenti"))
refs = [
    "[1] Y. Potter, N. Crispino, V. Siu, C. Wang, and D. Song, \"Peer-Preservation in Frontier Models,\" International Conference on Machine Learning (ICML), 2026. OpenReview: https://openreview.net/forum?id=YGIG0gViul.",
    "[2] L. Zhu and Q. Lu, \"Verifiability-First AI Engineering in the Era of AIware: A Conceptual Framework, Design Principles, and Architectural Patterns for Scalable Verification,\" manuscript, 2026.",
    "[3] Q. Lu, A. E. Hassan, D. Lo, and T. Zimmermann, \"Engineering Agentic Systems: Toward Verification-First AIware Engineering,\" IEEE Software, vol. 43, no. 5, pp. 22-29, 2026. doi:10.1109/MS.2026.3697911.",
    "[4] K. Vaidhyanathan and H. Muccini, \"Software Architecture for Agentic AI: Challenges and Way Ahead,\" in Software Architecture - ECSA 2026 Tracks and Workshops, Springer, 2027, pp. 83-98.",
    "[5] Y. Liu, S. K. Lo, Q. Lu, L. Zhu, D. Zhao, X. Xu, S. Harrer, and J. Whittle, \"Agent Design Pattern Catalogue: A Collection of Architectural Patterns for Foundation Model Based Agents,\" Journal of Systems and Software, vol. 220, 112278, 2025.",
    "[6] B. Xia, Y. Liu, Q. Lu, L. Zhu, D. Sejdinovic, and X. Xu, \"AgentOps Pattern Catalogue: Architectural Patterns for Safe and Observable Operations of Foundation Model-Based Agents,\" Information and Software Technology, vol. 200, 108307, 2026.",
    "[7] C. Wohlin, P. Runeson, M. Host, M. C. Ohlsson, B. Regnell, and A. Wesslen, Experimentation in Software Engineering. Springer, 2012.",
    "[8] B. A. Kitchenham, S. L. Pfleeger, L. M. Pickard, P. W. Jones, D. C. Hoaglin, K. El Emam, and J. Rosenberg, \"Preliminary Guidelines for Empirical Research in Software Engineering,\" IEEE Transactions on Software Engineering, vol. 28, no. 8, pp. 721-734, 2002.",
]
for ref in refs:
    story.append(p(ref, "Ref"))

story.append(Spacer(1, 4 * mm))
story.append(p(
    "Nota di coerenza editoriale: nel manoscritto allegato l'abstract indica tre pattern, mentre un passaggio dell'introduzione ne indica quattro. "
    "Con il disegno qui proposto, tutti i riferimenti devono essere uniformati a 'three patterns and an unprotected baseline'.",
    "Smallx",
))

doc.build(story)
print(OUTPUT)
