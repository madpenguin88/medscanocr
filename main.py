import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_INSTRUCTION = """Ești un sistem automat de tip Document AI, specializat în digitizarea buletinelor de analize medicale din România (Synevo, Regina Maria, Sanador, Bioclinica etc.).

Extrage TOATE datele din document și returnează STRICT un obiect JSON valid cu structura de mai jos. Niciun alt text, nicio introducere, niciun bloc Markdown.

Structura JSON obligatorie:
{
  "pacient": {
    "nume_complet": "string",
    "varsta": "string",
    "sex": "string (M sau F)",
    "data_recoltare": "string (DD.MM.YYYY sau gol)",
    "laborator": "string (numele laboratorului sau clinicii exact cum apare în document, ex: Synevo, Regina Maria, Sanador, Bioclinica; gol dacă nu există)"
  },
  "analize": [
    {
      "categorie": "string (ex: Biochimie, Hematologie, Urina, Hormoni etc.)",
      "denumire": "string (exact din document)",
      "rezultat": "string (valoarea numerica sau textul calitativ)",
      "um": "string (unitatea de masura sau gol)",
      "interval_referinta": "string (copiaza EXACT din document, inclusiv texte lungi; gol daca nu exista)",
      "interval_ref_pacient": "string (intervalul numeric aplicabil DOAR acestui pacient, ex: '0.29 - 1.67' sau '< 4.2' sau '8.64 - 29'; gol daca nu e interval pe varsta/sex)",
      "min_ref": number sau null,
      "max_ref": number sau null,
      "in_afara_limitelor": true sau false
    }
  ]
}

Reguli stricte:
1. INTERVAL_REFERINTA: Copiaza TEXTUL COMPLET din coloana de referinta, inclusiv fraze lungi. Nu trunchia, nu reformata.
2. INTERVAL_REF_PACIENT: Daca intervalul contine subintervale pe varsta/sex/stadii (ex: Testosteron, Prolactina, Estradiol), alege randul care corespunde pacientului si scrie DOAR acea valoare (ex: '0.29 - 1.67'). Daca intervalul e simplu sau unic, lasa gol (va fi preluat din interval_referinta). Nu copia textul complet, nu adauga etichete, scrie doar cifrele.
3. MIN_REF si MAX_REF: Extrage limitele numerice din interval_ref_pacient (daca exista) sau din interval_referinta simplu. Daca exista doar limita superioara (ex: < 100), pune min=null si max=100. Daca nu exista interval numeric, pune null.
3. IN_AFARA_LIMITELOR = true daca: valoarea e bold/colorata diferit, are asterisc (*), sageata (↑↓), H/L, sau depaseste numeric limitele din acelasi rand.
4. Include fiecare analiza din document, fara exceptie.
5. Pentru rezultate calitative (negativ, clara, normal etc.) pune textul exact in campul "rezultat".
6. Daca un camp lipseste din document, foloseste string gol "" pentru string-uri si null pentru numere."""


@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    try:
        content = await file.read()
        mime_type = file.content_type or "application/pdf"

        full_response = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_bytes(data=content, mime_type=mime_type)],
                )
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                media_resolution="MEDIA_RESOLUTION_HIGH",
                response_mime_type="application/json",
                system_instruction=[types.Part.from_text(text=SYSTEM_INSTRUCTION)],
            ),
        ):
            # Skip thinking tokens — only accumulate actual response parts
            if chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    if not getattr(part, "thought", False) and part.text:
                        full_response += part.text
            elif chunk.text:
                full_response += chunk.text

        # Strip markdown code fences if Gemini wrapped the JSON
        stripped = full_response.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
            full_response = stripped

        # Validate JSON before returning
        json.loads(full_response)

        print("=== GEMINI RAW RESPONSE ===")
        print(full_response)
        print("=== END GEMINI RESPONSE ===")
        return {"text": f"=== GEMINI_STRUCTURED ===\n{full_response}"}
    except Exception as e:
        print(f"=== GEMINI ERROR: {e} ===")
        return {"text": f"Eroare procesare: {str(e)}"}


@app.post("/debug")
async def debug_extract(file: UploadFile = File(...)):
    """Same as /extract but without schema enforcement — returns raw Gemini text to see full output."""
    try:
        content = await file.read()
        mime_type = file.content_type or "application/pdf"

        full_response = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-3.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=content, mime_type=mime_type),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
                media_resolution="MEDIA_RESOLUTION_HIGH",
                system_instruction=[types.Part.from_text(text="""Extrage toate analizele medicale din acest PDF și listează-le în formatul:
DENUMIRE | VALOARE | UM | INTERVAL_REFERINTA | IN_AFARA_LIMITELOR

Fii exhaustiv — include fiecare rând din tabelele de rezultate, inclusiv intervalele de referință exact cum apar în PDF.""")],
            ),
        ):
            if chunk.text:
                full_response += chunk.text

        print("=== DEBUG RAW RESPONSE ===")
        print(full_response)
        print("=== END DEBUG RESPONSE ===")
        return {"raw": full_response}
    except Exception as e:
        return {"error": str(e)}


# ── Health Report models ────────────────────────────────────────────────────

class HealthAnalyzeItem(BaseModel):
    denumire: str
    valoare: Optional[float] = None
    valoare_text: Optional[str] = None
    unitate: str = ""
    status: str = ""
    min_ref: Optional[float] = None
    max_ref: Optional[float] = None
    interval_referinta: Optional[str] = None


class HealthRecord(BaseModel):
    data: Optional[str] = None
    laborator: Optional[str] = None
    analize: List[HealthAnalyzeItem]


class HealthReportRequest(BaseModel):
    records: List[HealthRecord]
    varsta: Optional[int] = None
    sex: Optional[str] = None


HEALTH_REPORT_INSTRUCTION = """Ești un asistent medical AI specializat în interpretarea analizelor de laborator. \
Primești date despre analizele medicale ale unui pacient și generezi un raport de sănătate detaliat în limba română.

Returnează STRICT un obiect JSON valid cu structura de mai jos. Niciun alt text, nicio introducere, niciun bloc Markdown.

{
  "stare_generala": "normal | atentie | monitorizare | critica",
  "rezumat": "string (2-3 propoziții rezumat general)",
  "interpretare_analize": [
    {
      "categorie": "string (ex: Metabolism glucidic, Hematologie, Lipide, Tiroidă etc.)",
      "analize": [
        {
          "denumire": "string",
          "valoare_curenta": number sau null,
          "unitate": "string",
          "status": "normal | crescut | scazut | anormal",
          "trend": "stabil | crescator | descrescator | insuficient_date",
          "interpretare_text": "string (explicatie 1-2 propoziții în română simplă)",
          "cauze_posibile": ["string"],
          "riscuri": ["string"],
          "recomandari_specifice": ["string"]
        }
      ]
    }
  ],
  "bune_practici": ["string"],
  "analize_recomandate": [
    {
      "denumire": "string",
      "motiv": "string",
      "urgenta": "urgent | recomandat | optional"
    }
  ],
  "nota_medicala": "string"
}

Reguli stricte:
1. stare_generala: "normal" = toate ok; "atentie" = 1-2 ușor deviate; "monitorizare" = multiple deviate sau tendințe; "critica" = valori semnificativ patologice.
2. trend: calculează dacă există mai multe valori în timp pentru aceeași analiză; altfel "insuficient_date".
3. cauze_posibile, riscuri, recomandari_specifice: completează DOAR pentru analize cu status != "normal" sau trend îngrijorător. Pentru cele normale, lasă liste goale [].
4. analize_recomandate: sugerează analize complementare relevante bazate pe ce ai găsit.
5. Răspunde EXCLUSIV în limba română.
6. nota_medicala trebuie să fie: "Acest raport este generat cu ajutorul inteligenței artificiale și are scop informativ. Nu înlocuiește consultul medical de specialitate."
"""


@app.post("/health-report")
async def generate_health_report(request: HealthReportRequest):
    try:
        profil = f"Pacient: sex={request.sex or 'nespecificat'}, vârsta={request.varsta or 'nespecificată'} ani"

        records_text_parts = []
        for rec in request.records:
            lines = [f"--- Data recoltare: {rec.data or 'nespecificată'} | Laborator: {rec.laborator or 'nespecificat'} ---"]
            for a in rec.analize:
                val_str = str(a.valoare) if a.valoare is not None else (a.valoare_text or "—")
                ref_str = a.interval_referinta or (
                    f"{a.min_ref}–{a.max_ref}" if a.min_ref is not None and a.max_ref is not None
                    else (f"<{a.max_ref}" if a.max_ref is not None else (f">{a.min_ref}" if a.min_ref is not None else ""))
                )
                lines.append(f"  {a.denumire}: {val_str} {a.unitate} | status={a.status} | referinta={ref_str}")
            records_text_parts.append("\n".join(lines))

        prompt = f"""{profil}

Date analize medicale (ordonate cronologic):
{chr(10).join(records_text_parts)}

Generează raportul de sănătate complet pentru acest pacient."""

        full_response = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-3.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
                response_mime_type="application/json",
                system_instruction=[types.Part.from_text(text=HEALTH_REPORT_INSTRUCTION)],
            ),
        ):
            if chunk.text:
                full_response += chunk.text

        print("=== HEALTH REPORT RESPONSE (first 500 chars) ===")
        print(full_response[:500])
        print("=== END ===")

        parsed = json.loads(full_response)
        return parsed
    except json.JSONDecodeError as e:
        print(f"=== HEALTH REPORT JSON ERROR: {e} ===\nRaw: {full_response[:300]}")
        raise HTTPException(status_code=500, detail=f"Gemini returned invalid JSON: {str(e)}")
    except Exception as e:
        print(f"=== HEALTH REPORT ERROR: {e} ===")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
