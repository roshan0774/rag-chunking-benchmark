"""Generate a fictional 200-page employee handbook plus a 200-question golden set.

Design: 20 policy topics x 10 regional offices = 200 pages. Pages on the same topic share
most of their wording and differ in the office name and one key value. That gives near-duplicate
distractors, the main difficulty in real handbook and policy RAG: the retriever must find the right
*office's* page, not just the right topic.

"Lakeview Group" is a fictional company. All names and values are generated at random.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

COMPANY = "Lakeview Group"
OFFICES = ["Charlotte", "Atlanta", "Denver", "Austin", "Columbus",
           "Phoenix", "Raleigh", "Nashville", "Tampa", "Richmond"]

FIRST = ["Maya", "Jordan", "Priya", "Luis", "Grace", "Omar", "Elena", "Marcus", "Hannah", "Wei",
         "Tessa", "Andre", "Nina", "Caleb", "Sofia", "Ravi", "Leah", "Tomas", "Ivy", "Derek"]
LAST = ["Brennan", "Okafor", "Lindqvist", "Ramirez", "Sato", "Whitaker", "Novak", "Chandra",
        "Delgado", "Harlow", "Kim", "Mbeki", "Fontaine", "Price", "Varga", "Castillo"]


@dataclass
class Topic:
    title: str
    intro: str
    fact: str        # uses {office} and {value}
    question: str    # paraphrased; uses {office}
    values: list     # candidate values; 10 distinct ones are sampled per topic


def _money(lo, hi, step):
    return [f"${v:,}" for v in range(lo, hi, step)]


TOPICS = [
    Topic("Paid Time Off Accrual",
          "Paid time off (PTO) combines vacation, personal, and short-term sick time into one balance.",
          "Full-time employees assigned to the {office} office accrue {value} hours of PTO each pay period.",
          "How many hours of paid time off do full-time staff in {office} earn per pay period?",
          [f"{h / 4:.2f}" for h in range(16, 33)]),
    Topic("Business Travel Lodging",
          "Employees traveling on company business should book lodging through the corporate travel portal.",
          "For employees based in the {office} office, the nightly hotel rate cap is {value} before taxes.",
          "What is the maximum nightly hotel rate a {office}-based employee can book for a work trip?",
          _money(139, 260, 7)),
    Topic("Meal Per Diem",
          "Meal allowances cover breakfast, lunch, and dinner on travel days and are paid as a flat per diem.",
          "The daily meal per diem for {office} office staff is {value}.",
          "How much is the daily food allowance for travelers from the {office} office?",
          _money(48, 96, 3)),
    Topic("Remote and Hybrid Work",
          "The hybrid work program lets eligible roles split time between home and the office.",
          "Eligible employees in the {office} office may work remotely up to {value} days per week.",
          "How many days a week can eligible {office} employees work from home?",
          ["one", "two", "three", "four", "one and a half", "two and a half", "three and a half",
           "zero", "five", "four and a half"]),
    Topic("Parental Leave",
          "Parental leave is available to all parents following the birth, adoption, or foster placement of a child.",
          "Employees in the {office} office receive {value} weeks of fully paid parental leave.",
          "How many weeks of paid leave does a new parent at the {office} location get?",
          [str(w) for w in range(6, 21)]),
    Topic("Tuition Assistance",
          "Tuition assistance supports job-related degrees, certificates, and professional courses.",
          "The annual tuition reimbursement limit for {office} employees is {value}.",
          "Up to what amount per year will the company reimburse tuition for someone in {office}?",
          _money(2500, 10500, 500)),
    Topic("Building Access Hours",
          "Badge access to company buildings is limited to posted hours outside of approved exceptions.",
          "The {office} office building opens for badge access at {value} on weekdays.",
          "What time can employees first badge into the {office} building on a weekday?",
          ["5:30 AM", "5:45 AM", "6:00 AM", "6:15 AM", "6:30 AM", "6:45 AM", "7:00 AM", "7:15 AM",
           "7:30 AM", "7:45 AM", "8:00 AM"]),
    Topic("Commuter and Parking Benefit",
          "The commuter benefit offsets parking and transit costs for employees who come into the office.",
          "The monthly parking and transit subsidy for the {office} office is {value}.",
          "How much does the company contribute each month toward commuting costs in {office}?",
          _money(40, 300, 15)),
    Topic("Wellness Stipend",
          "The wellness stipend can be used for gym memberships, fitness classes, and approved wellness apps.",
          "{office} office employees receive an annual wellness stipend of {value}.",
          "What is the yearly fitness and wellness allowance for {office} staff?",
          _money(200, 1250, 50)),
    Topic("Overtime Pre-Approval",
          "Non-exempt employees must obtain manager approval before working overtime.",
          "In the {office} office, overtime beyond {value} hours in a week requires director approval.",
          "After how many weekly overtime hours does a {office} employee need a director to sign off?",
          [str(h) for h in range(4, 16)]),
    Topic("Laptop Refresh Cycle",
          "Company laptops are replaced on a fixed cycle to keep hardware secure and supported.",
          "Laptops issued to {office} office employees are refreshed every {value} months.",
          "How often do {office} employees get a new company laptop?",
          [str(m) for m in (18, 24, 27, 30, 33, 36, 42, 48, 54, 60)]),
    Topic("Expense Report Deadlines",
          "Expense reports must include itemized receipts and be submitted through the expense system.",
          "{office} office employees must submit expense reports within {value} days of incurring the expense.",
          "What is the deadline, in days, for {office} staff to file an expense report?",
          [str(d) for d in range(10, 61, 5)]),
    Topic("Mileage Reimbursement",
          "Employees who use a personal vehicle for company business are reimbursed per mile driven.",
          "The mileage reimbursement rate for the {office} office is {value} cents per mile.",
          "How many cents per mile are {office} employees paid for driving their own car on business?",
          [str(c) for c in range(52, 72)]),
    Topic("Bereavement Leave",
          "Bereavement leave allows time away following the death of a family member.",
          "Employees in the {office} office may take up to {value} days of paid bereavement leave.",
          "How many paid days off does a {office} employee get after a death in the family?",
          [str(d) for d in range(3, 15)]),
    Topic("Jury Duty",
          "The company supports employees fulfilling civic obligations such as jury service.",
          "{office} office employees receive full pay for up to {value} days of jury duty per year.",
          "For how many days of jury service per year will a {office} employee keep full pay?",
          [str(d) for d in range(5, 31, 2)]),
    Topic("Company Holidays",
          "The holiday calendar is published each December for the following year.",
          "The {office} office observes {value} paid company holidays per calendar year.",
          "How many paid holidays does the {office} location give each year?",
          [str(d) for d in range(8, 16)] + ["9.5", "10.5", "11.5"]),
    Topic("Introductory Period",
          "New hires complete an introductory period with structured check-ins with their manager.",
          "The introductory period for new hires in the {office} office lasts {value} days.",
          "How long is the probation period for new employees joining in {office}?",
          [str(d) for d in range(30, 181, 15)]),
    Topic("Security Desk Contacts",
          "Each office maintains a staffed security desk for badge issues, visitors, and emergencies.",
          "The {office} office security desk can be reached at internal extension {value}.",
          "What phone extension reaches building security in {office}?",
          [str(x) for x in range(4100, 4999, 37)]),
    Topic("Emergency Assembly Points",
          "During an evacuation, employees must leave by the nearest exit and report to their assembly point.",
          "Employees evacuating the {office} office gather at assembly point {value}.",
          "Where should {office} staff gather after leaving the building in a fire drill?",
          [f"{c}-{n}" for c in "ABCDEFGH" for n in (2, 5, 7)]),
    Topic("IT Help Desk Response Times",
          "IT support requests are logged through the help desk portal and prioritized by impact.",
          "For the {office} office, standard IT tickets receive a first response within {value} business hours.",
          "How quickly should a {office} employee expect a first reply on a normal IT ticket?",
          [str(h) for h in range(2, 25, 2)]),
]

BOILERPLATE = [
    "This policy applies to all regular full-time and part-time employees unless a collective agreement "
    "or local law provides otherwise. Temporary staff and contractors should consult their agency agreement.",
    "Managers are responsible for applying this policy consistently and for documenting any approved "
    "exceptions in the HR system. Exceptions must be reviewed by People Operations at least annually.",
    "Where this policy conflicts with applicable federal, state, or local requirements, the more generous "
    "provision to the employee will apply. People Operations reviews office-level values each fiscal year.",
    "Employees should keep records that support any claim made under this policy. Records may be "
    "requested during internal audits, and misuse of benefits may result in disciplinary action.",
]


@dataclass
class QA:
    id: int
    question: str
    answer: str
    page: int
    topic: str
    office: str


def generate(seed: int = 7) -> tuple[list[dict], list[QA]]:
    """Return (pages, questions). Page dicts: {page, title, office, paragraphs}."""
    rng = random.Random(seed)
    pages, qas = [], []
    for t_i, topic in enumerate(TOPICS):
        values = rng.sample(topic.values, len(OFFICES))
        for o_i, office in enumerate(OFFICES):
            page_no = len(pages) + 1
            contact = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            fact = topic.fact.format(office=office, value=values[o_i])
            bp = rng.sample(BOILERPLATE, 3)
            paragraphs = [
                topic.intro + " " + bp[0],
                f"Office-specific provisions for {office}. " + fact
                + f" Questions about this provision should be directed to {contact}, People Operations partner "
                  f"for the {office} office.",
                bp[1],
                bp[2] + f" Policy reference LV-{t_i + 1:02d}-{o_i + 1:02d}, revision {rng.randint(2, 9)}.",
            ]
            pages.append({"page": page_no, "title": f"Section {t_i + 1}.{o_i + 1}  {topic.title}: {office} Office",
                          "office": office, "paragraphs": paragraphs})
            qas.append(QA(len(qas) + 1, topic.question.format(office=office), values[o_i],
                          page_no, topic.title, office))
    return pages, qas


def write_pdf(pages: list[dict], out: Path) -> None:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    ss = getSampleStyleSheet()
    story = []
    for i, p in enumerate(pages):
        story += [Paragraph(f"{COMPANY} Employee Handbook (fictional)", ss["Italic"]),
                  Paragraph(p["title"], ss["Heading2"]), Spacer(1, 6)]
        for para in p["paragraphs"]:
            story += [Paragraph(para, ss["BodyText"]), Spacer(1, 6)]
        if i < len(pages) - 1:
            story.append(PageBreak())
    out.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(out), pagesize=LETTER, title=f"{COMPANY} Employee Handbook (fictional)").build(story)


def build_dataset(out_dir: str | Path = "data", seed: int = 7) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    pages, qas = generate(seed)
    pdf = out_dir / "lakeview_handbook_200p.pdf"
    golden = out_dir / "golden_200.jsonl"
    write_pdf(pages, pdf)
    out_dir.mkdir(parents=True, exist_ok=True)
    golden.write_text("\n".join(json.dumps(asdict(q)) for q in qas) + "\n")
    return pdf, golden
