"""
segmentador.py
==============

Segmentador semántico de textos largos, diseñado para producir EL MISMO
resultado tanto si el texto trae saltos de línea como si viene todo pegado.

Idea central
------------
Los saltos de línea son una pista poco fiable, así que no se usan como criterio
de corte. El texto se normaliza a un flujo continuo de oraciones y las fronteras
se deciden por cambio de tema (cohesión léxica / semántica), no por formato.

Uso rápido
----------
    seg = SegmentadorSemantico()
    for s in seg.segmentar(texto):
        print(s.texto)

Dependencias: solo librería estándar.
Opcional: sentence-transformers (usar_embeddings=True) para mayor precisión.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Recursos léxicos
# --------------------------------------------------------------------------

STOPWORDS = set("""
a an the and or but if while of to in on at by for with from as is are was were
be been being it its this that these those there here he she they them his her
their our your you i we not no nor so than then too very can could would will
shall may might must do does did done have has had am about into over under
again further once
el la los las un una unos unas y e o u pero si de del al en con por para como
que se su sus lo le les es son era eran ser estar este esta estos estas ese esa
eso aquel aquella aquello no ni tan mas muy ya tambien sobre entre sin hasta
desde cuando donde porque cual cuales quien quienes ha han habia fue fueron
""".split())

# Marcadores que indican continuidad discursiva: si la oración siguiente empieza
# con uno de ellos, cortar ahí es mala idea.
CONECTORES_CONTINUIDAD = {
    "however", "therefore", "moreover", "furthermore", "besides", "thus",
    "consequently", "meanwhile", "nevertheless", "additionally", "also",
    "then", "this", "these", "those", "it", "they", "he", "she", "such",
    "sin", "embargo", "ademas", "por", "tanto", "asimismo", "entonces",
    "luego", "esto", "esta", "estos", "estas", "eso", "asi", "aunque",
}

ABREVIATURAS = {
    "sr", "sra", "srta", "dr", "dra", "ing", "lic", "prof", "mr", "mrs", "ms",
    "jr", "st", "inc", "ltd", "vs", "etc", "fig", "num", "no", "vol", "pp",
    "ed", "eds", "cf", "al", "ca", "dept", "aprox", "approx", "ej",
}

_PH_PUNTO = "\x00"        # punto protegido
_PH_ELIPSIS = "\x01"      # puntos suspensivos protegidos

_TRAD_COMILLAS = {
    ord("“"): '"', ord("”"): '"', ord("„"): '"',
    ord("‘"): "'", ord("’"): "'", ord("‚"): "'",
    ord("–"): "-", ord("—"): "-", ord(" "): " ",
}
_TRAD_INVISIBLES = dict.fromkeys(
    map(ord, "​‌‍﻿"), None
)

_RE_URL = re.compile(r"\b(?:https?://|www\.)\S+|\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_RE_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_RE_ABREV = re.compile(
    r"\b(" + "|".join(sorted(ABREVIATURAS, key=len, reverse=True)) + r")\.",
    re.IGNORECASE,
)
_RE_INICIAL = re.compile(r"\b([A-Za-zÁÉÍÓÚÑáéíóúñ])\.")
_RE_FRONTERA = re.compile(
    r"""([.!?]+["'\)\]]*)      # terminador, con comillas o cierres pegados
        (\s+)                  # espacio obligatorio
        (?=["'\(\[¿¡]*[A-ZÁÉÍÓÚÑ0-9])  # la siguiente arranca en mayúscula o dígito
    """,
    re.VERBOSE,
)
_RE_TOKEN = re.compile(r"[a-záéíóúüñ0-9']{2,}")


def limpiar_markdown(texto: str) -> str:
    """
    Convierte [ancla](url) en "ancla" y elimina notas del tipo [[7]](url).

    Se cuentan los paréntesis para no romperse con URLs que ya los contienen,
    por ejemplo .../wiki/Wolf's_Lair_(Los_Angeles)#cite_note-7
    """
    salida: List[str] = []
    i, n = 0, len(texto)
    while i < n:
        c = texto[i]
        if c != "[":
            salida.append(c)
            i += 1
            continue
        # cierre del corchete, contando anidados
        nivel, j = 0, i
        while j < n:
            if texto[j] == "[":
                nivel += 1
            elif texto[j] == "]":
                nivel -= 1
                if nivel == 0:
                    break
            j += 1
        if j >= n or j + 1 >= n or texto[j + 1] != "(":
            salida.append(c)
            i += 1
            continue
        # cierre del paréntesis, contando anidados
        nivel, k = 0, j + 1
        while k < n:
            if texto[k] == "(":
                nivel += 1
            elif texto[k] == ")":
                nivel -= 1
                if nivel == 0:
                    break
            k += 1
        if k >= n:
            salida.append(c)
            i += 1
            continue
        ancla = texto[i + 1:j]
        if re.fullmatch(r"\[?\s*\d+\s*\]?", ancla):   # nota al pie: se elimina
            pass
        else:
            salida.append(limpiar_markdown(ancla))
        i = k + 1
    return "".join(salida)


def es_titulo(linea: str, max_palabras: int = 9) -> bool:
    """Heurística para encabezados sin formato: línea corta y sin punto final."""
    t = linea.strip().strip("#* ").strip()
    if not t or len(t.split()) > max_palabras:
        return False
    if t[-1] in ".!?:;,":
        return False
    if not (t[0].isupper() or t[0].isdigit()):
        return False
    return sum(1 for c in t if c.isalpha()) >= 2


# --------------------------------------------------------------------------
# Estructuras de salida
# --------------------------------------------------------------------------

@dataclass
class Segmento:
    """Un segmento coherente de texto."""
    indice: int
    texto: str
    oracion_inicio: int
    oracion_fin: int          # exclusivo
    oraciones: List[str] = field(default_factory=list, repr=False)

    @property
    def n_oraciones(self) -> int:
        return self.oracion_fin - self.oracion_inicio

    @property
    def n_caracteres(self) -> int:
        return len(self.texto)

    @property
    def n_palabras(self) -> int:
        return len(self.texto.split())

    def metricas(self) -> Dict[str, float]:
        """Rasgos básicos útiles para analizar texto generado por IA."""
        palabras = [p.lower().strip(".,;:!?\"'()") for p in self.texto.split()]
        palabras = [p for p in palabras if p]
        n = len(palabras) or 1
        largos = [len(o.split()) for o in self.oraciones] or [0]
        media = sum(largos) / len(largos)
        var = sum((x - media) ** 2 for x in largos) / len(largos)
        return {
            "n_oraciones": float(len(self.oraciones)),
            "n_palabras": float(n),
            "long_media_oracion": round(media, 2),
            "desv_long_oracion": round(math.sqrt(var), 2),
            "diversidad_lexica": round(len(set(palabras)) / n, 4),
            "densidad_contenido": round(
                sum(1 for p in palabras if p not in STOPWORDS) / n, 4
            ),
        }


# --------------------------------------------------------------------------
# Segmentador
# --------------------------------------------------------------------------

class SegmentadorSemantico:
    """
    Segmenta texto por cambio temático, ignorando el formato de entrada.

    Parámetros
    ----------
    ventana : int
        Número de oraciones a cada lado de la frontera que se comparan.
    sensibilidad : float
        Multiplicador de la desviación estándar en el umbral adaptativo.
        Más bajo = más cortes. Rango útil: -0.5 a 1.0.
    min_oraciones : int
        Mínimo de oraciones por segmento.
    max_caracteres : int | None
        Si un segmento lo excede, se parte por su frontera interna más débil.
        Nunca parte una oración.
    usar_saltos_de_linea : bool
        Si es True, un salto de línea real refuerza la frontera. Déjalo en
        False si necesitas que ambas versiones del texto den lo mismo.
    usar_embeddings : bool
        Si es True usa sentence-transformers en vez de cohesión léxica.
    modelo : str
        Nombre del modelo de sentence-transformers.
    """

    def __init__(
        self,
        ventana: int = 2,
        sensibilidad: float = 0.0,
        min_oraciones: int = 2,
        max_caracteres: Optional[int] = None,
        min_caracteres: int = 0,
        min_palabras_oracion: int = 5,
        usar_lsa: bool = True,
        dim_lsa: int = 48,
        max_vocabulario: int = 20000,
        max_muestra: int = 1200,
        limpiar_marcado: bool = True,
        detectar_titulos: bool = True,
        usar_saltos_de_linea: bool = False,
        usar_embeddings: bool = False,
        modelo: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        self.ventana = max(1, ventana)
        self.sensibilidad = sensibilidad
        self.min_oraciones = max(1, min_oraciones)
        self.max_caracteres = max_caracteres
        self.min_caracteres = min_caracteres
        self.min_palabras_oracion = min_palabras_oracion
        self.usar_lsa = usar_lsa
        self.dim_lsa = dim_lsa
        self.max_vocabulario = max_vocabulario
        self.max_muestra = max_muestra
        self.limpiar_marcado = limpiar_marcado
        self.detectar_titulos = detectar_titulos
        self.usar_saltos_de_linea = usar_saltos_de_linea
        self.usar_embeddings = usar_embeddings
        self._modelo_nombre = modelo
        self._modelo = None

    # -------------------- API pública --------------------

    def segmentar(self, texto: str) -> List[Segmento]:
        plano, marcas, titulos = self._normalizar(texto)
        oraciones, saltos, es_tit = self._dividir_oraciones(plano, marcas, titulos)
        oraciones, saltos, es_tit = self._agrupar_fragmentos(oraciones, saltos, es_tit)
        if not oraciones:
            return []
        forzados = [i for i, t in enumerate(es_tit) if t and i > 0]
        if len(oraciones) <= self.min_oraciones and not forzados:
            return [self._construir(0, oraciones, 0, len(oraciones))]

        vectores = self._vectorizar(oraciones)
        similitudes = self._similitudes(vectores)
        profundidades = self._profundidades(similitudes)
        profundidades = self._ajustar_por_cohesion(
            profundidades, oraciones, saltos
        )
        cortes = self._elegir_cortes(profundidades, forzados)
        segmentos = self._construir_todos(oraciones, cortes)
        segmentos = self._fusionar_pequenos(segmentos, vectores)
        if self.max_caracteres:
            segmentos = self._partir_grandes(segmentos, similitudes)
        return self._reindexar(segmentos)

    def comparar_formatos(self, texto_a: str, texto_b: str) -> Dict[str, object]:
        """Comprueba que dos versiones del mismo texto se segmentan igual."""
        a = [s.texto for s in self.segmentar(texto_a)]
        b = [s.texto for s in self.segmentar(texto_b)]
        return {
            "identicos": a == b,
            "n_a": len(a),
            "n_b": len(b),
            "diferencias": [
                (i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y
            ],
        }

    # -------------------- 1. Normalización --------------------

    def _normalizar(
        self, texto: str
    ) -> Tuple[str, List[int], List[Tuple[int, int]]]:
        """
        Devuelve el texto como flujo continuo, las posiciones donde había un
        salto de línea y los tramos que parecen encabezados.
        """
        t = unicodedata.normalize("NFC", texto)
        t = t.translate(_TRAD_INVISIBLES).translate(_TRAD_COMILLAS)
        t = re.sub(r"\\n\\?", "\n", t)     # saltos pegados como texto literal
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = t.replace("…", "...")
        if self.limpiar_marcado:
            t = limpiar_markdown(t)

        partes: List[str] = []
        marcas: List[int] = []
        titulos: List[Tuple[int, int]] = []
        largo = 0
        for trozo in re.split(r"(\n+)", t):
            if not trozo:
                continue
            if trozo.startswith("\n"):
                if partes and partes[-1].endswith(" "):
                    partes[-1] = partes[-1].rstrip()
                    largo = sum(len(p) for p in partes)
                marcas.append(largo)
                partes.append(" ")
                largo += 1
                continue
            linea = re.sub(r"[ \t]+", " ", trozo).strip()
            if not linea:
                continue
            if self.detectar_titulos and es_titulo(linea):
                titulos.append((largo, largo + len(linea)))
            partes.append(linea)
            largo += len(linea)
            partes.append(" ")
            largo += 1
        plano = "".join(partes).rstrip()
        return plano, marcas, titulos

    # -------------------- 2. Oraciones --------------------

    def _dividir_oraciones(
        self,
        texto: str,
        marcas: Sequence[int],
        titulos: Sequence[Tuple[int, int]] = (),
    ) -> Tuple[List[str], List[bool], List[bool]]:
        protegido = self._proteger(texto)
        limites = {m.end() for m in _RE_FRONTERA.finditer(protegido)}
        inicios_titulo = set()
        for ini, fin in titulos:            # el título es siempre unidad aparte
            limites.add(ini)
            limites.add(fin + 1)
            inicios_titulo.add(ini)
        limites.add(len(protegido))
        limites = sorted(x for x in limites if 0 < x <= len(protegido))

        oraciones, saltos, es_tit = [], [], []
        inicio = 0
        for fin in limites:
            bruto = protegido[inicio:fin]
            texto_or = self._restaurar(bruto).strip()
            if texto_or:
                oraciones.append(texto_or)
                saltos.append(any(inicio <= p <= fin for p in marcas))
                es_tit.append(inicio in inicios_titulo)
            inicio = fin
        return oraciones, saltos, es_tit

    def _agrupar_fragmentos(
        self,
        oraciones: List[str],
        saltos: List[bool],
        es_tit: List[bool],
    ) -> Tuple[List[str], List[bool], List[bool]]:
        """
        Une fragmentos demasiado cortos (un título, "You?", una fecha suelta)
        con la oración siguiente. Evita segmentos huérfanos y vectores vacíos.
        """
        if not oraciones:
            return oraciones, saltos, es_tit

        salida: List[str] = []
        marcas: List[bool] = []
        titulos: List[bool] = []
        buf, buf_salto, buf_tit = "", False, False
        for o, s, t in zip(oraciones, saltos, es_tit):
            actual = (buf + " " + o).strip() if buf else o
            s_actual = buf_salto or s
            t_actual = buf_tit or (t and not buf)
            if len(actual.split()) < self.min_palabras_oracion:
                buf, buf_salto, buf_tit = actual, s_actual, t_actual
                continue
            salida.append(actual)
            marcas.append(s_actual)
            titulos.append(t_actual)
            buf, buf_salto, buf_tit = "", False, False
        if buf:
            if salida:
                salida[-1] = (salida[-1] + " " + buf).strip()
                marcas[-1] = marcas[-1] or buf_salto
            else:
                salida.append(buf)
                marcas.append(buf_salto)
                titulos.append(buf_tit)
        return salida, marcas, titulos

    @staticmethod
    def _proteger(texto: str) -> str:
        def _url(m):
            return m.group(0).replace(".", _PH_PUNTO)

        t = _RE_URL.sub(_url, texto)
        t = t.replace("...", _PH_ELIPSIS)
        t = _RE_DECIMAL.sub(_PH_PUNTO, t)
        t = _RE_ABREV.sub(lambda m: m.group(1) + _PH_PUNTO, t)
        t = _RE_INICIAL.sub(lambda m: m.group(1) + _PH_PUNTO, t)
        return t

    @staticmethod
    def _restaurar(texto: str) -> str:
        return texto.replace(_PH_PUNTO, ".").replace(_PH_ELIPSIS, "...")

    # -------------------- 3. Vectorización --------------------

    def _vectorizar(self, oraciones: Sequence[str]):
        if self.usar_embeddings:
            return self._vectorizar_embeddings(oraciones)
        disperso = self._vectorizar_lexico(oraciones)
        if self.usar_lsa and len(disperso) >= 6:
            denso = self._proyectar_lsa(disperso)
            if denso is not None:
                return denso
        return disperso

    def _vectorizar_lexico(self, oraciones: Sequence[str]) -> List[Dict[str, float]]:
        """TF-IDF disperso por oración, sin palabras vacías."""
        docs = []
        for o in oraciones:
            tokens = [
                t for t in _RE_TOKEN.findall(self._sin_acentos(o.lower()))
                if t not in STOPWORDS
            ]
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0.0) + 1.0
            docs.append(tf)

        n = len(docs)
        df: Dict[str, int] = {}
        for tf in docs:
            for t in tf:
                df[t] = df.get(t, 0) + 1

        vectores = []
        for tf in docs:
            v = {}
            for t, f in tf.items():
                idf = math.log((n + 1) / (df[t] + 1)) + 1.0
                v[t] = (1.0 + math.log(f)) * idf
            norma = math.sqrt(sum(x * x for x in v.values())) or 1.0
            vectores.append({t: x / norma for t, x in v.items()})
        return vectores

    def _proyectar_lsa(self, vectores: List[Dict[str, float]]):
        """
        Proyecta las oraciones a un espacio latente (LSA por SVD truncado).

        Sin esto, dos oraciones del mismo tema que no comparten ninguna palabra
        tienen similitud cero. La descomposición recupera esa relación a través
        de las coocurrencias de todo el documento.

        Se calcula sobre la matriz de Gram (muestra x muestra), que es mucho más
        barata que la SVD directa cuando el vocabulario es grande.
        """
        try:
            import numpy as np
        except ImportError:
            return None

        n = len(vectores)
        frec: Dict[str, int] = {}
        for v in vectores:
            for t in v:
                frec[t] = frec.get(t, 0) + 1
        if len(frec) < 4:
            return None
        vocab = sorted(frec, key=lambda t: (-frec[t], t))[: self.max_vocabulario]
        idx = {t: i for i, t in enumerate(vocab)}

        muestra = list(range(n))
        if n > self.max_muestra:      # documentos enormes: subespacio por muestreo
            paso = n / float(self.max_muestra)
            muestra = [int(i * paso) for i in range(self.max_muestra)]

        M = np.zeros((len(muestra), len(vocab)), dtype=np.float32)
        for fila, r in enumerate(muestra):
            for t, x in vectores[r].items():
                j = idx.get(t)
                if j is not None:
                    M[fila, j] = x

        # k pequeño obliga a agrupar por tema. Si k se acerca al rango, la
        # descomposición solo rota el espacio original y no aporta nada.
        k = int(round(len(muestra) ** 0.5))
        k = max(3, min(self.dim_lsa, k, min(M.shape) - 1))

        try:
            gram = M @ M.T
            valores, U = np.linalg.eigh(gram)            # ascendente
        except np.linalg.LinAlgError:
            return None
        orden = np.argsort(valores)[::-1][:k]
        valores = np.clip(valores[orden], 1e-9, None)
        comp = (U[:, orden].T @ M) / np.sqrt(valores)[:, None]   # k x vocab

        Z = np.zeros((n, k), dtype=np.float32)
        for r, v in enumerate(vectores):
            for t, x in v.items():
                j = idx.get(t)
                if j is not None:
                    Z[r] += x * comp[:, j]
        normas = np.linalg.norm(Z, axis=1, keepdims=True)
        normas[normas == 0] = 1.0
        return list(Z / normas)

    def _vectorizar_embeddings(self, oraciones: Sequence[str]):
        if self._modelo is None:
            from sentence_transformers import SentenceTransformer  # lazy
            self._modelo = SentenceTransformer(self._modelo_nombre)
        import numpy as np

        embs = self._modelo.encode(list(oraciones), normalize_embeddings=True)
        return [np.asarray(e, dtype=float) for e in embs]

    @staticmethod
    def _sin_acentos(texto: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", texto)
            if unicodedata.category(c) != "Mn"
        )

    # -------------------- 4. Similitud y profundidad --------------------

    def _coseno(self, a, b) -> float:
        if isinstance(a, dict):
            if not a or not b:
                return 0.0
            if len(a) > len(b):
                a, b = b, a
            num = sum(v * b.get(k, 0.0) for k, v in a.items())
            na = math.sqrt(sum(v * v for v in a.values())) or 1.0
            nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
            return num / (na * nb)
        import numpy as np

        na = float(np.linalg.norm(a)) or 1.0
        nb = float(np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / (na * nb))

    def _bloque(self, vectores, ini: int, fin: int):
        trozo = vectores[max(0, ini):fin]
        if isinstance(vectores[0], dict):
            acum: Dict[str, float] = {}
            for v in trozo:
                for k, x in v.items():
                    acum[k] = acum.get(k, 0.0) + x
            return acum
        import numpy as np

        return np.sum(np.vstack(trozo), axis=0)

    def _similitudes(self, vectores) -> List[float]:
        n = len(vectores)
        w = self.ventana
        sims = []
        for i in range(n - 1):
            izq = self._bloque(vectores, i - w + 1, i + 1)
            der = self._bloque(vectores, i + 1, i + 1 + w)
            sims.append(self._coseno(izq, der))
        return sims

    @staticmethod
    def _profundidades(sims: Sequence[float]) -> List[float]:
        """Depth score al estilo TextTiling: cuán profundo es el valle."""
        n = len(sims)
        prof = []
        for i in range(n):
            j = i
            while j > 0 and sims[j - 1] >= sims[j]:
                j -= 1
            k = i
            while k < n - 1 and sims[k + 1] >= sims[k]:
                k += 1
            prof.append((sims[j] - sims[i]) + (sims[k] - sims[i]))
        return prof

    def _ajustar_por_cohesion(
        self,
        prof: List[float],
        oraciones: Sequence[str],
        saltos: Sequence[bool],
    ) -> List[float]:
        ajustado = list(prof)
        for i in range(len(ajustado)):
            siguiente = oraciones[i + 1]
            primera = self._sin_acentos(
                siguiente.lstrip("\"'([").split(" ")[0].lower().strip(".,;:")
            )
            if primera in CONECTORES_CONTINUIDAD:
                ajustado[i] -= 0.15          # penaliza cortar tras un conector
            if self.usar_saltos_de_linea and saltos[i]:
                ajustado[i] += 0.20          # el formato solo refuerza, no decide
        return ajustado

    def _elegir_cortes(
        self, prof: Sequence[float], forzados: Sequence[int] = ()
    ) -> List[int]:
        if not prof:
            return sorted(set(forzados))
        media = sum(prof) / len(prof)
        var = sum((x - media) ** 2 for x in prof) / len(prof)
        umbral = media + self.sensibilidad * math.sqrt(var)

        candidatos = sorted(
            (i for i, p in enumerate(prof) if p > umbral and p > 0),
            key=lambda i: prof[i],
            reverse=True,
        )
        elegidos: List[int] = list(dict.fromkeys(forzados))
        n_or = len(prof) + 1
        for i in candidatos:
            corte = i + 1
            if corte < self.min_oraciones or n_or - corte < self.min_oraciones:
                continue
            if any(abs(corte - c) < self.min_oraciones for c in elegidos):
                continue
            elegidos.append(corte)
        return sorted(elegidos)

    # -------------------- 5. Construcción y post proceso --------------------

    @staticmethod
    def _construir(idx: int, oraciones: Sequence[str], ini: int, fin: int) -> Segmento:
        trozo = list(oraciones[ini:fin])
        return Segmento(
            indice=idx,
            texto=" ".join(trozo).strip(),
            oracion_inicio=ini,
            oracion_fin=fin,
            oraciones=trozo,
        )

    def _construir_todos(
        self, oraciones: Sequence[str], cortes: Sequence[int]
    ) -> List[Segmento]:
        limites = [0] + list(cortes) + [len(oraciones)]
        return [
            self._construir(i, oraciones, limites[i], limites[i + 1])
            for i in range(len(limites) - 1)
        ]

    def _fusionar_pequenos(
        self, segmentos: List[Segmento], vectores
    ) -> List[Segmento]:
        """Un segmento corto se une al vecino más parecido, no al de al lado."""
        if len(segmentos) < 2 or self.min_caracteres <= 0:
            return segmentos
        cambio = True
        while cambio and len(segmentos) > 1:
            cambio = False
            for i, s in enumerate(segmentos):
                if s.n_caracteres >= self.min_caracteres:
                    continue
                izq = segmentos[i - 1] if i > 0 else None
                der = segmentos[i + 1] if i < len(segmentos) - 1 else None
                v = self._bloque(vectores, s.oracion_inicio, s.oracion_fin)
                sim_izq = (
                    self._coseno(v, self._bloque(vectores, izq.oracion_inicio, izq.oracion_fin))
                    if izq else -1.0
                )
                sim_der = (
                    self._coseno(v, self._bloque(vectores, der.oracion_inicio, der.oracion_fin))
                    if der else -1.0
                )
                objetivo = i - 1 if sim_izq >= sim_der else i + 1
                a, b = sorted((i, objetivo))
                fusion = Segmento(
                    indice=a,
                    texto=(segmentos[a].texto + " " + segmentos[b].texto).strip(),
                    oracion_inicio=segmentos[a].oracion_inicio,
                    oracion_fin=segmentos[b].oracion_fin,
                    oraciones=segmentos[a].oraciones + segmentos[b].oraciones,
                )
                segmentos = segmentos[:a] + [fusion] + segmentos[b + 1:]
                cambio = True
                break
        return segmentos

    def _partir_grandes(
        self, segmentos: List[Segmento], sims: Sequence[float]
    ) -> List[Segmento]:
        salida: List[Segmento] = []
        pendientes = list(segmentos)
        while pendientes:
            s = pendientes.pop(0)
            if s.n_caracteres <= self.max_caracteres or s.n_oraciones < 2:
                salida.append(s)
                continue
            interiores = range(s.oracion_inicio, s.oracion_fin - 1)
            mejor = min(interiores, key=lambda i: sims[i]) + 1
            izq = Segmento(
                indice=0,
                texto=" ".join(s.oraciones[: mejor - s.oracion_inicio]),
                oracion_inicio=s.oracion_inicio,
                oracion_fin=mejor,
                oraciones=s.oraciones[: mejor - s.oracion_inicio],
            )
            der = Segmento(
                indice=0,
                texto=" ".join(s.oraciones[mejor - s.oracion_inicio:]),
                oracion_inicio=mejor,
                oracion_fin=s.oracion_fin,
                oraciones=s.oraciones[mejor - s.oracion_inicio:],
            )
            pendientes = [izq, der] + pendientes
        return sorted(salida, key=lambda x: x.oracion_inicio)

    @staticmethod
    def _reindexar(segmentos: List[Segmento]) -> List[Segmento]:
        for i, s in enumerate(segmentos):
            s.indice = i
        return segmentos
