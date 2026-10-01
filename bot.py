import os
import re
import html
import subprocess
import tempfile
import asyncio
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from openai import OpenAI
from google import genai

# ReportLab Imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# --- CREDENTIALS (REPLACE WITH YOUR ACTUAL KEYS) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

# Clients
or_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
You are a senior university curriculum architect and academic dean serving a multi-disciplinary institution.
Your function is to convert an uploaded textbook outline, syllabus, or topic list from ANY academic discipline (e.g., Computer Science, Mathematics, Statistics, Engineering, Business, Economics, Life Sciences) into an accredited 12-week university course blueprint.

CRITICAL INSTRUCTIONS & STRICT ACADEMIC GUARDRAILS:
1. FIELD-AGNOSTIC RIGOR:
   - Identify the primary field from the source topics.
   - Adjust terminology accordingly:
     * Engineering/CS: Practical coding/hardware labs, test suites, architecture benchmarks.
     * Math/Stats: Proof workshops, problem sets, analytical derivations, statistical modeling.
     * Business/Finance: Case studies, valuation spreadsheets, strategic market analyses.
     * Science/Medicine: Experimental design, empirical analysis, lab protocols.
2. STRICT DOMAIN ISOLATION:
   - Every lecture topic, lab/seminar session, assignment, and exam question MUST be derived directly from the supplied source text.
   - Never inject unrelated domains, generic programming tasks, or sports analytics unless explicitly present in the input.
3. 5-PART ACCREDITED SCHEMA:
   - Part 1: Course Meta (Discipline, Course Code & Title, Level, Prerequisites, Primary Reference).
   - Part 2: Pedagogical Objectives (5 Bloom's Taxonomy outcomes tailored to the field: Analyze, Evaluate, Construct/Formulate, Implement/Apply, Synthesize).
   - Part 3: 12-Week Master Schedule (Week 01 to 12. Include: Weekly Theme, Lecture Topics, Hands-on Lab/Workshop/Case Session, Deliverables/Assessments. Week 06 Midterm Exam, Week 12 Final Capstone/Assessment).
   - Part 4: Assessment Scheme (Quizzes, Assignments, Midterm, Capstone Project/Paper, Final Exam totaling 100%).
   - Part 5: Instructional Staff Blueprint (TA/Facilitator prep hours, weekly grading guidance, and 3 common student learning bottlenecks with exact remediation strategies).
4. CLEAN OUTPUT RULES:
   - Do NOT emit raw HTML tags (no <br>, <div>, <span>). Use clean standard Markdown line breaks and bullet points.
"""

def generate_with_fallback(prompt: str, sys_prompt: str = SYSTEM_PROMPT) -> tuple[str, str]:
    """Fallback chain: Nemotron Ultra -> Nemotron Lightning -> Gemini 2.5 Flash."""
    # 1. Primary
    try:
        response = or_client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content, "nvidia/nemotron-3-ultra-550b-a55b:free"
    except Exception as e:
        print(f"Primary failed: {e}")

    # 2. Fallback 1
    try:
        response = or_client.chat.completions.create(
            model="nvidia/nemotron-3.5-lightning:free",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content, "nvidia/nemotron-3.5-lightning:free"
    except Exception as e:
        print(f"Fallback 1 failed: {e}")

    # 3. Fallback 2
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "system_instruction": sys_prompt,
                "temperature": 0.2
            }
        )
        return response.text, "gemini-2.5-flash"
    except Exception as e:
        print(f"Fallback 2 failed: {e}")
        raise RuntimeError(f"All models in fallback chain failed: {e}")

def sanitize_for_reportlab(text: str) -> str:
    """Escapes XML entities and converts markdown syntax to safe ReportLab tags."""
    if not text:
        return ""
    
    # 1. Normalize any literal <br> variants to a safe placeholder
    text = re.sub(r'<br\s*/?>', '__SAFE_BR__', text, flags=re.IGNORECASE)
    
    # 2. Escape XML entities (&, <, >) to avoid paraparser crashes
    text = html.escape(text, quote=False)
    
    # 3. Restore valid self-closing <br/> tags
    text = text.replace('__SAFE_BR__', '<br/>')
    
    # 4. Convert Markdown formatting to strict ReportLab XML tags
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', text)
    
    return text

def build_pdf_from_text(raw_text: str, output_path: str):
    """Parses markdown text and safely compiles a clean, publication-ready PDF."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155'),
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3
    )

    code_style = ParagraphStyle('CodeBlock', parent=styles['Normal'], fontName='Courier', fontSize=7.5,
                                leading=9.5, backColor=colors.HexColor('#F1F5F9'), leftIndent=6, spaceAfter=6)
    story = []
    in_code, code_lines = False, []

    for line in raw_text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('```'):
            if in_code and code_lines:
                story.append(Preformatted('\n'.join(code_lines), code_style))
                code_lines = []
            in_code = not in_code
            continue
        if in_code:
            ex = line.rstrip().replace('\t', '    ')
            while len(ex) > 100:
                code_lines.append(ex[:100])
                ex = '    ' + ex[100:]
            code_lines.append(ex)
            continue
        if not stripped:
            story.append(Spacer(1, 4))
            continue

        if stripped.startswith('# '):
            clean_text = sanitize_for_reportlab(stripped[2:])
            story.append(Paragraph(clean_text, title_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=8))
        elif stripped.startswith('## '):
            clean_text = sanitize_for_reportlab(stripped[3:])
            story.append(Spacer(1, 6))
            story.append(Paragraph(clean_text, h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=6))
        elif stripped.startswith('### '):
            clean_text = sanitize_for_reportlab(stripped[4:])
            story.append(Paragraph(clean_text, h2_style))
        elif stripped.startswith(('* ', '- ', '• ')):
            bullet_body = stripped[2:].strip()
            clean_text = sanitize_for_reportlab(bullet_body)
            story.append(Paragraph(f"&bull; {clean_text}", bullet_style))
        elif re.match(r'^\d+\.\s+', stripped):
            clean_text = sanitize_for_reportlab(stripped)
            story.append(Paragraph(clean_text, bullet_style))
        elif stripped.startswith('|') and stripped.endswith('|'):
            # Filter out divider lines |---|---|
            if re.match(r'^\|[\s\-:|]+\|$', stripped):
                continue
            cells = [c.strip() for c in stripped.split('|')[1:-1]]
            cell_paragraphs = []
            for c in cells:
                safe_cell = sanitize_for_reportlab(c)
                is_header = 'Week' in stripped or 'Topic' in stripped
                cell_text = f"<b>{safe_cell}</b>" if is_header else safe_cell
                cell_paragraphs.append(Paragraph(cell_text, body_style))
            
            row_table = Table([cell_paragraphs], colWidths=[5.5 * inch / max(len(cells), 1)] * len(cells))
            row_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(row_table)
        else:
            clean_text = sanitize_for_reportlab(stripped)
            story.append(Paragraph(clean_text, body_style))

    if code_lines:
        story.append(Preformatted('\n'.join(code_lines), code_style))
    doc.build(story)

# ---------------------------------------------------------------------
# Assignments / Labs / Tutorials generated from the saved syllabus
# ---------------------------------------------------------------------
SYLLABUS_DIR = Path(__file__).parent / "syllabi"   # saved on disk, so restarts lose nothing
SYLLABUS_DIR.mkdir(exist_ok=True)

DOC_TYPES = {
    "assignment": ("📝 Assignment", """Write a student Assignment. Title: '# Week {w} Assignment'. Sections:
## Overview & Learning Objectives
## Instructions & Submission Format
## Questions / Tasks (5-6 graded tasks, increasing difficulty, marks per task, total 100)
## Grading Rubric (table)
## Model Solutions / Marking Guide"""),
    "lab": ("💻 Lab", """Write a hands-on Lab Handout. Title: '# Week {w} Lab Handout'. Sections:
## Learning Objectives
## Prerequisites & Setup
## Tasks (numbered, step-by-step, each with an expected deliverable)
## Starter Code / Template (fenced block: code for CS/Engineering, otherwise a worksheet, proof skeleton, spreadsheet layout or protocol)
## Grading Rubric (table, total 100)"""),
    "tutorial": ("📖 Tutorial", """Write a Tutorial Sheet for a tutorial / recitation session. Title: '# Week {w} Tutorial'. Sections:
## Key Concepts Recap (short)
## Worked Examples (3, fully solved step by step)
## Practice Problems (6, increasing difficulty)
## Solutions to Practice Problems"""),
}

DOC_SYSTEM_PROMPT = """You are a senior university instructor writing a polished, client-ready teaching document for ONE week of a course.
Use ONLY the supplied week excerpt and course context; never add unrelated topics. Adapt to the discipline.
Output clean Markdown only (no HTML tags), starting with a single '# ' title line. Be complete; never write TBD or placeholders."""

WEEK_RE = re.compile(r'^[\s|#>*\-•_]*\**\s*Week\s*0?(\d{1,2})\b', re.IGNORECASE)


def extract_week(syllabus: str, week: int) -> str:
    """Only this week's lines from the syllabus (keeps prompts short and on-topic)."""
    lines, out, cur = syllabus.split('\n'), [], False
    for line in lines:
        m = WEEK_RE.match(line)
        if m:
            cur = int(m.group(1)) == week
            if cur:
                out.append(line.rstrip())
        elif line.lstrip().startswith('#') or re.match(r'^[\s#*]*Part\s*\d', line, re.I):
            cur = False
        elif cur and line.strip():
            out.append(line.rstrip())
    return '\n'.join(out)[:2500] if out else syllabus[:3000]


def type_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=f"type_{k}") for k, (label, _) in DOC_TYPES.items()]])


def week_keyboard(kind: str) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(f"Week {w}", callback_data=f"gen_{kind}_{w}") for w in range(r, r + 4)] for r in (1, 5, 9)]
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="back")])
    return InlineKeyboardMarkup(rows)


async def handle_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data
    chat_id = q.message.chat_id

    if data.startswith("type_"):
        kind = data[5:]
        await q.answer()
        await q.message.reply_text(f"{DOC_TYPES[kind][0]} - pick a week:", reply_markup=week_keyboard(kind))
        return
    if data == "back":
        await q.answer()
        await q.message.edit_text("What would you like to generate?", reply_markup=type_keyboard())
        return

    _, kind, week = data.split("_")
    week = int(week)
    path = SYLLABUS_DIR / f"{chat_id}.md"
    if not path.exists():
        await q.answer("No syllabus saved yet. Upload a document first.", show_alert=True)
        return
    await q.answer()
    label, task = DOC_TYPES[kind]
    status = await q.message.reply_text(f"⏳ Generating Week {week} {label[2:]}...")
    syllabus = path.read_text(encoding="utf-8")
    prompt = (f"COURSE CONTEXT:\n\"\"\"\n{syllabus[:800]}\n\"\"\"\n\n"
              f"WEEK {week} SYLLABUS EXCERPT:\n\"\"\"\n{extract_week(syllabus, week)}\n\"\"\"\n\n" + task.format(w=f"{week:02d}"))
    try:
        loop = asyncio.get_running_loop()
        text, _ = await loop.run_in_executor(None, generate_with_fallback, prompt, DOC_SYSTEM_PROMPT)
        with tempfile.TemporaryDirectory() as tmp:
            name = f"Week_{week:02d}_{kind.title()}.pdf"
            out = os.path.join(tmp, name)
            await loop.run_in_executor(None, build_pdf_from_text, text, out)
            await status.delete()
            with open(out, "rb") as f:
                await context.bot.send_document(chat_id, f, filename=name, caption=f"{label} - Week {week}\nWant another?",
                                                reply_markup=type_keyboard())
    except Exception as e:
        await status.edit_text(f"❌ Failed: {e}")


async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not (SYLLABUS_DIR / f"{update.effective_chat.id}.md").exists():
        await update.message.reply_text("No syllabus yet. Upload a TOC/topic list first.")
        return
    await update.message.reply_text("What would you like to generate?", reply_markup=type_keyboard())

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📚 **University Curriculum Architect**\n\n"
        "Send any textbook Table of Contents or topic list (PDF or TXT) across **any field** (Engineering, Mathematics, Business, Science, Computing, etc.).\n\n"
        "I will extract the topics and generate an accredited 12-week syllabus with lectures, practical labs/workshops, quizzes, exams, and deliverables packaged in a clean, downloadable PDF.",
        parse_mode="Markdown"
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    file_name = document.file_name.lower()
    
    status_msg = await update.message.reply_text("📥 Downloading document...")

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, document.file_name)
        file = await context.bot.get_file(document.file_id)
        await file.download_to_drive(file_path)

        # 1. Text Extraction
        extracted_text = ""
        if file_name.endswith(".pdf"):
            await status_msg.edit_text("📄 Extracting text layers using pdftotext...")
            try:
                result = subprocess.run(["pdftotext", file_path, "-"], capture_output=True, text=True, check=True)
                extracted_text = result.stdout.strip()
            except Exception as e:
                await status_msg.edit_text(f"❌ Failed to extract PDF: {str(e)}")
                return
        elif file_name.endswith((".txt", ".md")):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read().strip()
        else:
            await status_msg.edit_text("❌ Please upload a PDF, TXT, or Markdown document.")
            return

        if not extracted_text:
            await status_msg.edit_text("❌ Could not extract any readable text from this document.")
            return

        # 2. Model Generation
        await status_msg.edit_text("⚙️ Synthesizing accredited curriculum across discipline...")
        
        prompt = f"""
SOURCE DOCUMENT CONTENT:
\"\"\"
{extracted_text[:14000]}
\"\"\"

Task: Identify the academic discipline and topics from the text above. Generate a complete, accredited 12-week university course blueprint strictly matching the system instructions.
"""
        try:
            loop = asyncio.get_running_loop()
            full_curriculum, model_used = await loop.run_in_executor(None, generate_with_fallback, prompt)
        except Exception as e:
            await status_msg.edit_text(f"❌ Model generation error: {str(e)}")
            return

        # 3. Compile Deliverable PDF with ReportLab
        await status_msg.edit_text("📑 Compiling publication-ready PDF...")
        pdf_out_path = os.path.join(tmpdir, "course_curriculum.pdf")
        try:
            build_pdf_from_text(full_curriculum, pdf_out_path)
        except Exception as e:
            await status_msg.edit_text(f"❌ PDF compilation error: {str(e)}")
            return

        # 4. Generate clean Telegram Summary Card
        summary_prompt = f"""
Summarize this course outline into a clean, mobile-friendly Telegram card (under 1,500 characters).
Rules:
- Identify Discipline, Course Code, and Title.
- List: Target Audience, 3 Major Prerequisites.
- 5 Bloom's Outcomes (one line each).
- Grading Breakdown (Percentages adding to 100%).
- Key Assessment Schedule (Quizzes, Midterm Week 6, Final Exam/Project Week 12).
- DO NOT use markdown tables; use bold bullets.

Text:
\"\"\"
{full_curriculum[:5000]}
\"\"\"
"""
        try:
            summary_card, _ = await loop.run_in_executor(None, generate_with_fallback, summary_prompt)
        except Exception:
            summary_card = full_curriculum[:1200] + "..."

        await status_msg.delete()
        (SYLLABUS_DIR / f"{update.effective_chat.id}.md").write_text(full_curriculum, encoding="utf-8")

        # 5. Send clean text preview in chat
        await update.message.reply_text(
            f"🎓 **Curriculum Generated Successfully**\n"
            f"*Engine:* `{model_used}`\n\n"
            f"{summary_card}",
            parse_mode="Markdown"
        )

        # 6. Send compiled PDF document
        with open(pdf_out_path, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename="course_curriculum.pdf",
                caption="📄 **Accredited 12-Week Course Syllabus (PDF)**\nIncludes full weekly lectures, lab/workshop specs, grading rubric, and instructional blueprints.\n\nNeed assignments, labs or tutorials for this course? Pick one below:",
                reply_markup=type_keyboard()
            )

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", menu))
    app.add_handler(CallbackQueryHandler(handle_buttons, pattern=r"^(type_\w+|gen_\w+_\d+|back)$"))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    print("🚀 Universal Syllabus Bot is listening for documents...")
    app.run_polling()

if __name__ == "__main__":
    main()
